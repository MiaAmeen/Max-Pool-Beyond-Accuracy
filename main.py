import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from .alexnet import AlexNet
import math

DEVICE = torch.device("mps" if torch.cuda.is_available() else "cpu")

# -------------------------
# HYPERPARAMS
# -------------------------
BATCH_SIZE = 128
INITIAL_TRAIN_EPOCHS = 20        # dense training
RETRAIN_EPOCHS = 8               # fine-tune after each pruning
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4              # L2 regularization (weight decay)
PRUNE_ITERATIONS = 5             # number of prune->retrain cycles
ALPHA = 0.1                      # quality parameter to multiply stddev (tunable)
# initial dropout rates for the 5 conv layers (you asked to allow different initial rates)
ALEXNET_INIT_DROPOUT_RATES = [0.05, 0.15, 0.25, 0.5, 0.5]
NUM_CLASSES_CIFAR10 = 10

# -------------------------
# Data (CIFAR-10)
# -------------------------
def get_dataloaders(batch_size=BATCH_SIZE, dataset_name='CIFAR10'):
    transform = transforms.Compose([
        transforms.Resize(224),  # for AlexNet/VGG input size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset_cls = torchvision.datasets.CIFAR10 if dataset_name == 'CIFAR10' else torchvision.datasets.CIFAR100

    trainset = dataset_cls(root='./data', train=True, download=True, transform=transform)
    testset = dataset_cls(root='./data', train=False, download=True, transform=transform)
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


def iterative_prune_train_retrain():
    trainloader, testloader = get_dataloaders()

    # Initialize model
    model = AlexNet(num_classes=NUM_CLASSES_CIFAR10, dropout_rates=ALEXNET_INIT_DROPOUT_RATES).to(DEVICE)
    model.disable_dropout()  # no dropout for initial dense training

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

    # --- 1. Initial dense training ---
    print("=== Initial dense training ===")
    for epoch in range(INITIAL_TRAIN_EPOCHS):
        loss = train_one_epoch(model, trainloader, optimizer, criterion)
        acc = evaluate(model, testloader)
        print(f"Epoch {epoch+1}/{INITIAL_TRAIN_EPOCHS}, Loss: {loss:.4f}, Test Acc: {acc:.2f}%")

    # --- 2. Iterative pruning + retraining ---
    target_sparsity = 0.9
    for prune_iter in range(PRUNE_ITERATIONS):
        print(f"\n=== Prune iteration {prune_iter+1}/{PRUNE_ITERATIONS} ===")

        # --- 2a. Prune conv layers ---
        print("Pruning conv layers...")
        for i, layer in enumerate(model.conv_layers):
            model.prune(layer, quality_param=ALPHA)

        # Compute remaining connections per conv layer
        conv_remaining = model.check_sparsity(type="CONV")

        # Scale dropout rates for conv layers
        new_dropout_rates = []
        for i, layer in enumerate(model.conv_layers):
            total = layer.weight.numel()
            orig_p = ALEXNET_INIT_DROPOUT_RATES[i]
            new_p = orig_p * math.sqrt(conv_remaining[i] / total)
            new_dropout_rates.append(new_p)
        # Append current FC dropout rates unchanged
        new_dropout_rates.extend(ALEXNET_INIT_DROPOUT_RATES[5:])  
        model.update_dropout(new_dropout_rates)

        # Freeze FC layers during conv retraining
        for layer in model.fc_layers:
            for param in layer.parameters():
                param.requires_grad = False
        for layer in model.conv_layers:
            for param in layer.parameters():
                param.requires_grad = True

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

        # Retrain conv layers
        print("Retraining conv layers...")
        for epoch in range(RETRAIN_EPOCHS):
            loss = train_one_epoch(model, trainloader, optimizer, criterion)
            acc = evaluate(model, testloader)
            print(f"Retrain Epoch {epoch+1}/{RETRAIN_EPOCHS}, Loss: {loss:.4f}, Test Acc: {acc:.2f}%")

        # --- 2b. Prune FC layers ---
        print("Pruning FC layers...")
        for i, layer in enumerate(model.fc_layers):
            model.prune(layer, quality_param=ALPHA)

        # Compute remaining connections per FC layer
        fc_remaining = model.check_sparsity(type="FC")

        # Scale dropout rates for FC layers
        new_dropout_rates = new_dropout_rates[:5]  # keep conv rates
        for i, layer in enumerate(model.fc_layers):
            total = layer.weight.numel()
            orig_p = ALEXNET_INIT_DROPOUT_RATES[5 + i]
            new_p = orig_p * math.sqrt(fc_remaining[0] / total)
            new_dropout_rates.append(new_p)
        model.update_dropout(new_dropout_rates)

        # Freeze conv layers during FC retraining
        for layer in model.conv_layers:
            for param in layer.parameters():
                param.requires_grad = False
        for layer in model.fc_layers:
            for param in layer.parameters():
                param.requires_grad = True

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

        # Retrain FC layers
        print("Retraining FC layers...")
        for epoch in range(RETRAIN_EPOCHS):
            loss = train_one_epoch(model, trainloader, optimizer, criterion)
            acc = evaluate(model, testloader)
            print(f"Retrain Epoch {epoch+1}/{RETRAIN_EPOCHS}, Loss: {loss:.4f}, Test Acc: {acc:.2f}%")

        # --- 2c. Print sparsity stats ---
        stats = model.check_sparsity()
        print(f"Remaining weights per layer: {stats}")

        # Optional: check overall sparsity and break if target reached
        total_params = sum([layer.weight.numel() for layer in model.conv_layers + model.fc_layers])
        remaining_params = sum(stats)
        overall_sparsity = 1 - remaining_params / total_params
        print(f"Overall sparsity: {overall_sparsity:.4f}")
        if overall_sparsity >= target_sparsity:
            print("Target sparsity reached. Stopping pruning.")
            break

    print("=== Pruning + retraining complete ===")
    return model, evaluate(model, testloader)


# -------------------------
# Run experiment
# -------------------------
if __name__ == "__main__":
    iterative_prune_train_retrain()
