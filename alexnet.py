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
    def __init__(self, num_classes=10, pooling_method="max", pruning_method="unstr"):
        super(AlexNet, self).__init__()
        self.name = "AlexNet"
        self.pooling_method = pooling_method
        if pooling_method == "max":
            pool_layer = nn.MaxPool2d
        elif pooling_method == "min":
            pool_layer = MinPool2d
        else:
            pool_layer = nn.AvgPool2d

        self.pruning_method = pruning_method
        self.prune = self.l1_unstructured_prune
        match pruning_method:
            case "rand-unstr":
                self.prune = self.rand_unstructured_prune
            case "str":
                self.prune = self.l1_structured_prune
            case "rand-str":
                self.prune = self.rand_structured_prune  

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), # torch.Size([64, 3, 3, 3])
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
    
    def l1_unstructured_prune(self, layer, threshold, remove=False):
        prune.l1_unstructured(layer, name="weight", amount=threshold)
        if remove: prune.remove(layer, 'weight')

        return self.check_sparsity(layer)
    
    def rand_unstructured_prune(self, layer, threshold, remove=False):
        prune.random_unstructured(layer, name="weight", amount=threshold)
        if remove: prune.remove(layer, 'weight')

        return self.check_sparsity(layer)
    
    def l1_structured_prune(self, layer, threshold, remove=False):
        # Prune entire channels with smallest L1-norms of their weights
        prune.ln_structured(layer, name="weight", amount=threshold, n=1, dim=0)  
        if remove: prune.remove(layer, 'weight') # not recommended to set to True; model will simply set "pruned" parameters to zero
        
        return self.check_sparsity(layer)

    def rand_structured_prune(self, layer, threshold, remove=False):
        # Randomly prune entire channels
        prune.random_structured(layer, name="weight", amount=threshold, dim=0)
        if remove: prune.remove(layer, 'weight')
        
        return self.check_sparsity(layer)
    
    def check_sparsity(self, layer):
        w = layer.weight_mask if prune.is_pruned(layer) else layer.weight
        return torch.sum(w == 0) / w.numel()


# if __name__ == "__main__":
#     input = torch.randn(1, 3, 32, 32)
#     model = AlexNet()
#     test = model.conv_layers[0]
#     print(model.check_sparsity(test))
#     print(model.l1_structured_prune(test))

#     print(model.l1_structured_prune(test))
#     print(model.check_sparsity(prune.remove(test, 'weight')))

#     model(input)