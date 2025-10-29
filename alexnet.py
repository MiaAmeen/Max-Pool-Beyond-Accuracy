import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch import Tensor
import math

class ThresholdPruning(prune.BasePruningMethod):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def compute_mask(self, t, default_mask) -> Tensor:
        mask = (t.abs() > self.threshold).to(t.device).to(dtype=default_mask.dtype)
        return mask

class AlexNet(nn.Module):
    """
    AlexNet variant for CIFAR-10 with configurable dropout after each convolutional layer.
    Includes dropout placeholders that can be turned off (set to Identity) or have their rates changed dynamically.
    """

    def __init__(self, num_classes=10, dropout_rates=None):
        super(AlexNet, self).__init__()

        # Default per-layer dropout rates (5 conv layers, 2 FC layers)
        if dropout_rates is None: dropout_rates = [0.05, 0.15, 0.25, 0.5, 0.5, 0.5, 0.5]

        self.features = nn.Sequential(
            # Conv layer 1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rates[0]),

            # Conv layer 2
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rates[1]),

            # Conv layer 3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rates[2]),

            # Conv layer 4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rates[3]),

            # Conv layer 5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_rates[4]),
        )

        # Classifier (flattened conv output → FC layers)
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rates[5]),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rates[6]),
            nn.Linear(4096, num_classes),
        )
        self.conv_layers = [m for m in self.features if isinstance(m, nn.Conv2d)]
        self.fc_layers = [m for m in self.classifier if isinstance(m, nn.Linear)][:-1] # exclude final FC !!
        self.dropouts = [d for d in self.features if isinstance(d, nn.Dropout2d)]

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def disable_dropout(self):
        """Disable all dropout layers (set dropout probability to 0)."""
        for dropout in self.dropouts: dropout.p = 0.0

    def update_dropout(self, pruned_ratio):
        """Update all dropout layers (conv + FC) with new list of rates."""
        # Expect 7 dropout rates (5 conv + 2 FC)
        assert len(pruned_ratio) == len(self.conv_layers) + len(self.fc_layers)
        for dropout in self.dropouts:
            layer_idx = self.dropouts.index(dropout)
            orig_p = dropout.p
            new_p = orig_p * math.sqrt(pruned_ratio[layer_idx])
            dropout.p = new_p
    
    def compute_layer_stdparam(self, module):
        """Compute std of module weight tensor (Conv2d or Linear)."""
        return torch.std(module.weight.data).item()
    
    def prune(self, layer, quality_param):
        initial = self.check_sparsity(layer)
        std = torch.std(layer.weight).item()
        threshold = quality_param * std
        ThresholdPruning.apply(layer, name="weight", threshold=threshold)
        remaining = self.check_sparsity(layer)

        return remaining/initial
    
    def check_sparsity(self, layer):
        """Return dict of layer name -> (total_params, remaining_params) for pruned layers."""
        mask = getattr(layer, "weight_mask", None)
        remaining_params = int(mask.sum().item()) if mask is not None else layer.weight.numel()
        return remaining_params

# if __name__ == "__main__":
#     model = AlexNet()
#     print(model.dropouts[0].p)
#     model.disable_dropout()
#     print(model.dropouts[0].p)
#     print(model.features[3].p)


