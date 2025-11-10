import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class MinPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return -self.maxpool(-x)

class AlexNet(nn.Module):
    """
    AlexNet variant for CIFAR-10 with configurable dropout after each convolutional layer.
    Includes dropout placeholders that can be turned off (set to Identity) or have their rates changed dynamically.
    """    
    def __init__(self, num_classes=10, pooling_method="max"):
        super(AlexNet, self).__init__()
        self.name = "AlexNet"
        self.pooling_method = pooling_method

        if pooling_method == "max":
            pool_layer = nn.MaxPool2d
        elif pooling_method == "min":
            pool_layer = MinPool2d
        else:
            pool_layer = nn.AvgPool2d

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            pool_layer(kernel_size=2, stride=2),  # 32 -> 16

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            pool_layer(kernel_size=2, stride=2),  # 16 -> 8

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            pool_layer(kernel_size=2, stride=2)   # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

        self.conv_layers = [m for m in self.features if isinstance(m, nn.Conv2d)]
        self.fc_layers = [m for m in self.classifier if isinstance(m, nn.Linear)][:-1] # exclude final FC !!

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
    
    def unstructured_magnitude_prune(self, layer, quality_param, remove=False):
        if prune.is_pruned(layer):
            weights = layer.weight_orig.data
            old_mask = layer.weight_mask.to(dtype=torch.int32)
        else:
            weights = layer.weight.data
            old_mask = torch.ones_like(weights, dtype=torch.int32)

        threshold = float(quality_param * torch.std(weights).item())
        mask = (torch.abs(weights) > threshold).int()
        prune.custom_from_mask(layer, name="weight", mask=mask & old_mask)
        
        if remove: prune.remove(layer, 'weight')

        return self.check_sparsity(layer)
    
    def l1_unstructured_prune(self, layer, threshold, remove=False):
        prune.l1_unstructured(layer, name="weight", amount=threshold)
        if remove: prune.remove(layer, 'weight')

        return self.check_sparsity(layer)
    
    def random_unstructured1_prune(self, layer, threshold, remove=False):
        prune.random_unstructured(layer, name="weight", amount=threshold)
        if remove: prune.remove(layer, 'weight')

        return self.check_sparsity(layer)
    
    def check_sparsity(self, layer):
        w = layer.weight_mask
        return torch.sum(w == 0) / w.numel()


# if __name__ == "__main__":
#     model = AlexNet()
#     print(model.unstructured_l1_prune(model.conv_layers[0], floor(model.conv_layers[0].weight.numel() * 0.1)))
#     print(model.unstructured_l1_prune(model.conv_layers[0], floor(model.conv_layers[0].weight.numel() * 0.1)))
