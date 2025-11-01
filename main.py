import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from alexnet import AlexNet
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# -------------------------
# HYPERPARAMS
# -------------------------
CRITERION = nn.CrossEntropyLoss()
BATCH_SIZE = 128
INITIAL_TRAIN_EPOCHS = 200        # dense training
MAX_RETRAIN_EPOCHS = 30               # fine-tune after each pruning
LEARNING_RATE = .05
WEIGHT_DECAY = 1e-4              # L2 regularization (weight decay)
MOMENTUM = 0.9
PRUNE_ITERATIONS = 5             # number of prune->retrain cycles
ALPHA = 0.1                      # quality parameter to multiply stddev (tunable)
NUM_CLASSES_CIFAR10 = 10

# -------------------------
# Data (CIFAR-10)
# -------------------------
DATA_PATH = "/share/csc591007f25/fameen/MaxPooling/data/"
# DATA_PATH = "./data/"
MODEL_PATH = "/share/csc591007f25/fameen/MaxPooling/models/"
# MODEL_PATH = "./models/"
def get_dataloaders(dataset, batch_size=BATCH_SIZE, toy_data=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if dataset == 'CIFAR10':
        dataset_cls, exists = datasets.CIFAR10, os.path.exists(os.path.join(DATA_PATH, 'cifar-10-batches-py'))
    else:
        dataset_cls, exists = datasets.CIFAR100, os.path.exists(os.path.join(DATA_PATH, 'cifar-100-python'))

    trainset = dataset_cls(root=DATA_PATH, train=True, download=not exists, transform=transform)
    testset = dataset_cls(root=DATA_PATH, train=False, download=not exists, transform=transform)

    if toy_data:
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


def initial_dense_train(model, dataset, trainloader, testloader):
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=INITIAL_TRAIN_EPOCHS)  # match full training duration

    print("=== Initial dense training ===")
    print("Epoch, Loss, Train Acc, Test Acc")
    for epoch in range(INITIAL_TRAIN_EPOCHS):
        loss = train_one_epoch(model, trainloader, optimizer, CRITERION)
        scheduler.step()
        train_acc = evaluate(model, trainloader)
        test_acc = evaluate(model, testloader)
        print(f"{epoch+1}/{INITIAL_TRAIN_EPOCHS}, {loss:.4f}, {train_acc:.4f}, {test_acc:.4f}")

    model_path = f"model-init-{model.name}{dataset}.pth"
    torch.save(model.state_dict(), model_path)
    return model_path


def iterative_prune_train_retrain(model, model_path, dataset, trainloader, testloader):
    # Printing Header
    print("Prune Iteration, conv1, conv2, conv3, conv4, conv5, fc1, fc2, train accuracy, pruned accuracy, retrain accuracy")

    # Load initial dense model
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    base_acc = evaluate(model, testloader)

    # --- 2. Iterative pruning + retraining ---
    for prune_iter in range(PRUNE_ITERATIONS):        
        # --- Prune conv layers ---
        conv_sparsities = [model.prune(layer, quality_param=ALPHA) for layer in model.conv_layers]
        pruned_acc = evaluate(model, testloader)

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
        for _ in range(MAX_RETRAIN_EPOCHS):
            train_one_epoch(model, trainloader, optimizer, CRITERION)

        # --- Prune FC layers ---
        fc_sparsities = [model.prune(layer, quality_param=ALPHA) for layer in model.fc_layers]

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
        for _ in range(MAX_RETRAIN_EPOCHS):
            train_one_epoch(model, trainloader, optimizer, CRITERION)

        retrain_acc = evaluate(model, testloader)

        print(f"{prune_iter+1}, {', '.join(f'{sparsity:.4f}' for sparsity in conv_sparsities)}, \
              {', '.join(f'{sparsity:.4f}' for sparsity in fc_sparsities)}, \
                {base_acc}, {pruned_acc}, {retrain_acc}")
        
        base_acc = retrain_acc

    torch.save(model.state_dict(), f"prune{model.name}{dataset}{model.pooling_method}.pth")
    return model


# -------------------------
# Run experiment
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', 
        type=str,
        choices=['CIFAR10', 'CIFAR100'], 
    )
    parser.add_argument(
        '--pooling_method', 
        type=str, 
        choices=['max', 'avg'],
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
    )
    args = parser.parse_args()
        
    model_path = MODEL_PATH + args.model_path
    dataset = args.dataset
    pooling_method = args.pooling_method

    trainloader, testloader = get_dataloaders(dataset)
    model = AlexNet(num_classes=NUM_CLASSES_CIFAR10, pooling_method=pooling_method).to(DEVICE)
    iterative_prune_train_retrain(model, dataset=dataset, model_path=model_path, trainloader=trainloader, testloader=testloader)
