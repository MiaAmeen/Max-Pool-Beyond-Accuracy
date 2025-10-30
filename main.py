import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from alexnet import AlexNet
from torchvision.models import alexnet


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# -------------------------
# HYPERPARAMS
# -------------------------
BATCH_SIZE = 128
INITIAL_TRAIN_EPOCHS = 20        # dense training
RETRAIN_EPOCHS = 30               # fine-tune after each pruning
LEARNING_RATE = .05
WEIGHT_DECAY = 1e-4              # L2 regularization (weight decay)
MOMENTUM = 0.9
PRUNE_ITERATIONS = 5             # number of prune->retrain cycles
ALPHA = 0.05                      # quality parameter to multiply stddev (tunable)
ALEXNET_INIT_DROPOUT_RATES = [0.05, 0.15, 0.25, 0.5, 0.5] # ehhh
NUM_CLASSES_CIFAR10 = 10
MAX_SPARSITY = 0.9               # target sparsity level

# -------------------------
# Data (CIFAR-10)
# -------------------------
def get_dataloaders(dataset_name, batch_size=BATCH_SIZE, toy_data=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset_cls = torchvision.datasets.CIFAR10 if dataset_name == 'CIFAR10' else torchvision.datasets.CIFAR100
    trainset = dataset_cls(root='./data', train=True, download=True, transform=transform)
    testset = dataset_cls(root='./data', train=False, download=True, transform=transform)

    if toy_data:
        # Randomly sample smaller subsets
        trainset = Subset(trainset, random.sample(range(len(trainset)), 512))
        testset = Subset(testset, random.sample(range(len(testset)), 128))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=4, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=4, shuffle=False)
    return trainloader, testloader

# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(dataloader, leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)

@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def initial_dense_train(model, trainloader, testloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=INITIAL_TRAIN_EPOCHS)  # match full training duration

    print("=== Initial dense training ===")
    for epoch in range(INITIAL_TRAIN_EPOCHS):
        loss = train_one_epoch(model, trainloader, optimizer, criterion)
        scheduler.step()
        test_acc = evaluate(model, testloader)
        print(f"Epoch {epoch+1}/{INITIAL_TRAIN_EPOCHS}, Loss: {loss:.4f}, Test Acc: {test_acc:.4f}%")

    model_path = f"model-initial-{len(trainloader)}.pth"
    torch.save(model.state_dict(), model_path)
    return model_path


def iterative_prune_train_retrain(model, model_path, trainloader, testloader):
    # Load initial dense model
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # --- 2. Iterative pruning + retraining ---
    for prune_iter in range(PRUNE_ITERATIONS):
        print(f"\n=== Prune iteration {prune_iter+1}/{PRUNE_ITERATIONS} ===")

        # --- 2a. Prune conv layers ---
        print("Pruning conv layers...")
        conv_sparsities = [model.prune(layer, quality_param=ALPHA) for layer in model.conv_layers]
        print(f"Conv layer sparsities: {[f'{s:.4f}, ' for s in conv_sparsities]}")

        # Freeze FC layers during conv retraining
        for layer in model.fc_layers:
            for param in layer.parameters():
                param.requires_grad = False
        for layer in model.conv_layers:
            for param in layer.parameters():
                param.requires_grad = True
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

        # Retrain conv layers
        print("Retraining conv layers...")
        for epoch in range(RETRAIN_EPOCHS):
            loss = train_one_epoch(model, trainloader, optimizer, criterion)
            acc = evaluate(model, testloader)
            print(f"Retrain Epoch {epoch+1}/{RETRAIN_EPOCHS}, Loss: {loss:.4f}, Test Acc: {acc:.2f}%")

        # --- 2b. Prune FC layers ---
        print("Pruning FC layers...")
        fc_sparsities = [model.prune(layer, quality_param=ALPHA) for layer in model.fc_layers]
        print(f"FC layer sparsities: {[f'{s:.4f}, ' for s in fc_sparsities]}")

        # Freeze conv layers during FC retraining
        for layer in model.conv_layers:
            for param in layer.parameters():
                param.requires_grad = False
        for layer in model.fc_layers:
            for param in layer.parameters():
                param.requires_grad = True
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

        # Retrain FC layers
        print("Retraining FC layers...")
        for epoch in range(RETRAIN_EPOCHS):
            loss = train_one_epoch(model, trainloader, optimizer, criterion)
            acc = evaluate(model, testloader)
            print(f"Retrain Epoch {epoch+1}/{RETRAIN_EPOCHS}, Loss: {loss:.4f}, Test Acc: {acc:.2f}%")

        torch.save(model.state_dict(), f"model-{prune_iter}.pth")

    print("=== Pruning + retraining complete ===")
    return model, evaluate(model, testloader)


# -------------------------
# Run experiment
# -------------------------
if __name__ == "__main__":
    trainloader, testloader = get_dataloaders('CIFAR10', toy_data=True)
    model = AlexNet(num_classes=NUM_CLASSES_CIFAR10).to(DEVICE)
    model_path = initial_dense_train(model=model, trainloader=trainloader, testloader=testloader)
    # iterative_prune_train_retrain(model, model_path=model_path, trainloader=trainloader, testloader=testloader)
