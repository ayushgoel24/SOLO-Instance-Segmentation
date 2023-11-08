import torch.nn as nn
import torch.nn.functional as F

class MaskBranch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, num_layers, num_grids):
        super(MaskBranch, self).__init__()
        self.head = self._make_layers(in_channels, out_channels, kernel_size=kernel_size, padding=padding, num_layers=num_layers)
        self.output = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(in_channels, grid**2, kernel_size=1), nn.Sigmoid()) for grid in num_grids]
        )
        self.output = nn.Sequential(*self.output)

    def _make_layers(self, in_channels, out_channels, kernel_size, padding, num_layers):
        layers = [
            nn.Sequential(
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size, padding=padding, bias=False),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True)
            ) for i in range(num_layers)
        ]
        return nn.Sequential(*layers)

    def forward(self, x, layer_index):
        # return [self.output[i](F.interpolate(self.head(x), scale_factor=2, mode="bilinear")) for i in range(len(self.num_grids))]
        return self.output[layer_index](F.interpolate(self.head(x), scale_factor=2, mode="bilinear"))