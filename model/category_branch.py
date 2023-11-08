import torch.nn as nn

class CategoryBranch(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, num_layers):
        super(CategoryBranch, self).__init__()
        self.head = self._make_layers(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, num_layers=num_layers)
        self.output = nn.Sequential(
            nn.Conv2d(self.in_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def _make_layers(self, in_channels, out_channels, kernel_size, padding, num_layers):
        layers = [
            nn.Sequential(
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size, padding=padding, bias=False),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True)
            ) for i in range(num_layers)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.output(self.head(x))