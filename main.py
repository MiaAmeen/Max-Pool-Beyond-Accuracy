import argparse
import random
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.utils.data import Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from alexnet import AlexNet
import os
from math import floor
import copy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# -------------------------
# HYPERPARAMS
# -------------------------
CRITERION = nn.CrossEntropyLoss()
BATCH_SIZE = 128
INITIAL_TRAIN_EPOCHS = 100        # dense training
MAX_RETRAIN_EPOCHS = 100               # fine-tune after each pruning
LR = .001
RETRAIN_LR = 0.0001
WEIGHT_DECAY = 1e-4              # L2 regularization (weight decay)
MOMENTUM = 0.9
PRUNE_ITERATIONS = 100             # number of prune->retrain cycles
ALPHA = 0.1                      # quality parameter to multiply stddev (tunable)
THRESHOLD = 0.05
THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
NUM_CLASSES_CIFAR10 = 10

# This is for producing deltas... maps the ith pooling layer to jth conv layer
conv_idx_map = {
    1: 0, 2: 1, 3: 4
}

# -------------------------
# Data (CIFAR-10)
# -------------------------
# DATA_PATH = "/share/csc591007f25/fameen/MaxPooling/data/"
# MODEL_PATH = "/share/csc591007f25/fameen/MaxPooling/models/"
DATA_PATH = "./data/"
MODEL_PATH = "./models/"


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

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=DEVICE.type == "cuda")
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=4, shuffle=False, pin_memory=DEVICE.type == "cuda")
    return trainloader, testloader

# -------------------------
# Train / Eval
# -------------------------
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
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
    return round(100.0 * correct / total, 4)


def initial_dense_train(model, dataset, trainloader, testloader, iter=None):
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=INITIAL_TRAIN_EPOCHS)  # match full training duration

    print("ITER, Epoch, Loss, Train Acc, Test Acc")
    for epoch in range(INITIAL_TRAIN_EPOCHS):
        loss = train_one_epoch(model, trainloader, optimizer, CRITERION)
        scheduler.step()
        train_acc = evaluate(model, trainloader)
        test_acc = evaluate(model, testloader)
        print(f"{iter if iter else 1}, {epoch+1}/{INITIAL_TRAIN_EPOCHS}, {loss:.4f}, {train_acc:.4f}, {test_acc:.4f}")

    model_path = MODEL_PATH + f"init{iter if iter else ""}{model.name}{dataset}{model.pooling_method}.pth"
    torch.save(model.state_dict(), model_path)
    return model_path


def iterative_prune_retrain_conv_layer(model, model_path, conv_idx, trainloader, testloader):
    # Printing Header3
    print(f"model version, pooling method, pruning method, prune iteration, conv_layer, sparsity, train accuracy, pruned accuracy, retrain accuracy, retrain loss")

    # Load initial dense model
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model_version = model_path.split("/")[-1][4]
    base_acc = evaluate(model, testloader)

    # --- 2. Iterative pruning + retraining ---
    layer = model.conv_layers[conv_idx]
    structured = "unstr" not in model.pruning_method
    threshold = floor(layer.out_channels * THRESHOLD) if structured else \
        floor(layer.weight.numel() * THRESHOLD)
    channel_density = layer.weight.numel() / layer.out_channels
    
    for prune_iter in range(PRUNE_ITERATIONS): 
        if prune.is_pruned(layer):
            threshold = min(threshold, torch.sum(layer.weight_mask == 1) / channel_density) if structured else \
                min(threshold, layer.weight_mask.numel())
        
        # --- Prune conv layers ---
        conv_sparsity = model.prune(layer, threshold)
        pruned_acc = evaluate(model, testloader)

        # Retrain conv layers
        optimizer = optim.Adam(model.parameters(), lr=RETRAIN_LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_RETRAIN_EPOCHS)  # match full training duration
        for _ in range(MAX_RETRAIN_EPOCHS):
            retrain_loss = train_one_epoch(model, trainloader, optimizer, CRITERION)
            scheduler.step()
        retrain_acc = evaluate(model, testloader)

        print(f"{model_version}, {model.pooling_method}, {model.pruning_method}, {prune_iter+1}, conv{conv_idx}, {conv_sparsity}, {base_acc}, {pruned_acc}, {retrain_acc}, {retrain_loss}")
        base_acc = retrain_acc

    # torch.save(model.state_dict(), f"{model_version}{model.name}{dataset}conv{conv_idx}{model.pooling_method}.pth")
    return model


def sample_dense_training(dataset, trainloader, testloader, pooling_method, model_path=None):
    for iter in range(1):
        model = AlexNet(num_classes=NUM_CLASSES_CIFAR10, pooling_method=pooling_method).to(DEVICE)
        if model_path: model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        initial_dense_train(model, dataset=dataset, trainloader=trainloader, testloader=testloader, iter=iter + 1)


@torch.no_grad()
def evaluate_delta(model_init: AlexNet, model_prune: AlexNet, dataloader, layer_name: str="pool1"):
    model_init.eval()
    model_prune.eval()
    model_init.attach_hooks()
    model_prune.attach_hooks()

    total_samples = 0
    accumulated = None
    for images, _ in dataloader:
        images = images.to(DEVICE)

        model_init(images)
        model_prune(images)

        out_init  = model_init.intermediate_outputs[layer_name]   # [B, C, H, W]
        out_prune = model_prune.intermediate_outputs[layer_name]  # [B, C, H, W]

        delta = (out_init - out_prune).abs()   # [B, C, H, W]
        delta_2d = delta.mean(dim=1)           # [B, H, W]
        batch_size = delta_2d.size(0)
        total_samples += batch_size

        if accumulated is None:
            accumulated = delta_2d.sum(dim=0)        # [H, W]
        else:
            accumulated += delta_2d.sum(dim=0)

    final_delta_image = accumulated / total_samples

    model_init.detach_hooks()
    model_prune.detach_hooks()

    return final_delta_image.cpu()
        
def delta(pruning_method, pool_idx):
    def delta_helper(model_init, pool_idx, test):
        imgs = {}
        layer_name = f"pool{pool_idx}"
        conv_idx = conv_idx_map[pool_idx]
        for thresh in THRESHOLDS:
            model_pruned = copy.deepcopy(model_init).to(DEVICE)
            print(model_pruned.prune(model_pruned.conv_layers[conv_idx], thresh))
            output = evaluate_delta(model_init, model_pruned, test, layer_name=layer_name)
            imgs[f"max-{thresh}"] = output
            torch.save(output, f"deltas/delta-{2}-{model_init.pooling_method}-{model_init.pruning_method}-{layer_name}-{thresh}.pth")
        return imgs

    MODEL_MAX = AlexNet(
        num_classes=NUM_CLASSES_CIFAR10, 
        pooling_method="max", 
        pruning_method=pruning_method).to(DEVICE)
    MODEL_MAX.load_state_dict(torch.load("/Users/destroyerofworlds/Desktop/DL/Max-Pool-Beyond-Accuracy/models/init2AlexNetCIFAR10max.pth", map_location=DEVICE))
    
    MODEL_AVG = AlexNet(
        num_classes=NUM_CLASSES_CIFAR10, 
        pooling_method="avg", 
        pruning_method=pruning_method).to(DEVICE)
    MODEL_AVG.load_state_dict(torch.load("/Users/destroyerofworlds/Desktop/DL/Max-Pool-Beyond-Accuracy/models/init2AlexNetCIFAR10avg.pth", map_location=DEVICE))

    _, test = get_dataloaders("CIFAR10")
    delta_helper(MODEL_MAX, pool_idx, test)
    delta_helper(MODEL_AVG, pool_idx, test)

# -------------------------
# Run experiment
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', 
        type=str,
        default='CIFAR10',
        choices=['CIFAR10', 'CIFAR100'], 
    )
    parser.add_argument(
        '--pooling_method', 
        type=str, 
        default='avg',
        choices=['max', 'min', 'avg'],
    )
    parser.add_argument(
        '--pruning_method', 
        type=str, 
        default='rand-unstr',
        choices=['unstr', 'str', 'rand-unstr', 'rand-str'],
        required=False
    )
    parser.add_argument(
        '--conv_idx',
        type=int, 
        required=False,
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=False,
    )
    parser.add_argument(
        '--sample', 
        action='store_true',
    )
    args = parser.parse_args()
    
    model_path = args.model_path if args.model_path else None
    dataset = args.dataset
    pooling_method = args.pooling_method
    pruning_method = args.pruning_method
    trainloader, testloader = get_dataloaders(dataset)

    if args.sample:
        sample_dense_training(dataset=dataset, trainloader=trainloader, testloader=testloader, pooling_method=pooling_method, model_path=model_path)
        quit(0)

    model = AlexNet(num_classes=NUM_CLASSES_CIFAR10, pooling_method=pooling_method, pruning_method=pruning_method).to(DEVICE)
    if not model_path:
        model_path = initial_dense_train(model, dataset=dataset, trainloader=trainloader, testloader=testloader)

    if args.conv_idx is not None: 
        conv_idx = args.conv_idx - 1
        model_path = MODEL_PATH + args.model_path
        iterative_prune_retrain_conv_layer(model, model_path=model_path, conv_idx=conv_idx, trainloader=trainloader, testloader=testloader)

    

if __name__ == "__main__":
    main()

    import matplotlib.pyplot as plt

    # delta("unstr", pool_idx=3)

    # pool = "pool1"
    # _max = torch.load(f"results/deltas/delta-2-max-unstr-{pool}-0.9.pth") \
    #     + torch.load(f"results/deltas/delta-2-max-unstr-{pool}-0.5.pth") \
    #     + torch.load(f"results/deltas/delta-2-max-unstr-{pool}-0.7.pth") \
    #     + torch.load(f"results/deltas/delta-2-max-unstr-{pool}-0.9.pth")
    
    # avg = torch.load(f"results/deltas/delta-2-avg-unstr-{pool}-0.9.pth") \
    #     + torch.load(f"results/deltas/delta-2-avg-unstr-{pool}-0.5.pth") \
    #     + torch.load(f"results/deltas/delta-2-avg-unstr-{pool}-0.7.pth") \
    #     + torch.load(f"results/deltas/delta-2-avg-unstr-{pool}-0.9.pth")
    
    # print("Tensor shape: ", _max.size())
    # print("Max tensor range:", _max.min(), _max.max())
    # print("Avg tensor min:", avg.min(), avg.max())
    
    # min_val = torch.min(_max.min(), avg.min())
    # max_val = torch.max(_max.max(), avg.max())

    # normalized_max = (_max - min_val) / (max_val - min_val)
    # normalized_avg = (avg - min_val) / (max_val - min_val)

    # # Compute global min and max for shared color scale
    # vmin = min(normalized_max.min(), normalized_avg.min())
    # vmax = max(normalized_max.max(), normalized_avg.max())

    # fig, axes = plt.subplots(2, 1, figsize=(8, 12))

    # im0 = axes[0].imshow(normalized_max, cmap="Blues", aspect="auto", vmin=vmin, vmax=vmax)
    # im1 = axes[1].imshow(normalized_avg, cmap="Blues", aspect="auto", vmin=vmin, vmax=vmax)
    # axes[0].set_xticks([])  # hide x ticks
    # axes[0].set_yticks([])
    # axes[1].set_xticks([])  # hide x ticks
    # axes[1].set_yticks([])

    # # Single shared colorbar
    # cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    # cbar.set_label("Delta")

    # plt.tight_layout()
    # plt.show()


