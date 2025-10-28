import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------
# CONFIG
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
BATCH_SIZE = 128
EPOCHS = 10
PRUNING_PERCENTAGES = [0.0, 0.2, 0.4, 0.6, 0.8]

# ---------------------------
# DATA
# ---------------------------
def get_dataloaders(dataset_name='CIFAR10'):
    transform = transforms.Compose([
        transforms.Resize(224),  # for AlexNet/VGG input size
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset_cls = torchvision.datasets.CIFAR10 if dataset_name == 'CIFAR10' else torchvision.datasets.CIFAR100

    trainset = dataset_cls(root='./data', train=True, download=True, transform=transform)
    testset = dataset_cls(root='./data', train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
    return trainloader, testloader


# ---------------------------
# MODEL CONFIGS (POOLING VARIANTS)
# ---------------------------
def replace_pooling(model, pooling_type="max"):
    for name, module in model.named_children():
        if isinstance(module, nn.MaxPool2d):
            if pooling_type == "avg":
                setattr(model, name, nn.AvgPool2d(kernel_size=module.kernel_size, stride=module.stride))
            elif pooling_type == "none":
                setattr(model, name, nn.Identity())
        else:
            replace_pooling(module, pooling_type)
    return model


def get_model(model_name="alexnet", pooling_type="max", num_classes=10):
    if model_name == "alexnet":
        model = torchvision.models.alexnet(num_classes=num_classes)
    elif model_name == "vgg":
        model = torchvision.models.vgg16(num_classes=num_classes)
    else:
        raise ValueError("Unsupported model name.")

    model = replace_pooling(model, pooling_type)
    return model.to(DEVICE)


# ---------------------------
# TRAINING AND EVAL
# ---------------------------
def train(model, trainloader, optimizer, criterion):
    model.train()
    for images, labels in tqdm(trainloader, leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def evaluate(model, testloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# ---------------------------
# PRUNING
# ---------------------------
def prune_model_fc_layers(model, amount=0.2):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)


# ---------------------------
# MAIN EXPERIMENT LOOP
# ---------------------------
def run_experiment(model_name, dataset_name, pooling_type):
    trainloader, testloader = get_dataloaders(dataset_name)
    num_classes = 10 if dataset_name == "CIFAR10" else 100

    model = get_model(model_name, pooling_type, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\n--- Training {model_name} with {pooling_type}-pooling on {dataset_name} ---")

    # Base training
    for epoch in range(EPOCHS):
        train(model, trainloader, optimizer, criterion)
        acc = evaluate(model, testloader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Accuracy: {acc:.2f}%")

    base_acc = evaluate(model, testloader)
    print(f"Base accuracy before pruning: {base_acc:.2f}%")

    results = []
    for p in PRUNING_PERCENTAGES:
        if p > 0:
            prune_model_fc_layers(model, amount=p)
        acc = evaluate(model, testloader)
        results.append((p, acc))
        print(f"Pruned {p*100:.0f}% → Accuracy: {acc:.2f}%")
    return results


# ---------------------------
# ENTRY POINT
# ---------------------------
if __name__ == "__main__":
    configs = [
        ("alexnet", "CIFAR10"),
        ("alexnet", "CIFAR100"),
        ("vgg", "CIFAR10"),
        ("vgg", "CIFAR100"),
    ]

    for model_name, dataset in configs:
        for pooling in ["max", "avg", "none"]:
            run_experiment(model_name, dataset, pooling)
