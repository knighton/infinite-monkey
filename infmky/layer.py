import torch
from torch import nn


class Debug(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        print('%s :: %s' % (self.name or '', tuple(x.shape)))
        return x


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        shape = (x.shape[0],) + self.shape
        return x.view(*shape)


class Flatten(Reshape):
    def __init__(self):
        super().__init__(-1)


class Conv1dBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm1d(out_channels),
        )


class Skip(nn.Module):
    def __init__(self, *inner):
        super().__init__()
        self.inner = nn.Sequential(*inner)
        self.raw_is_used = nn.Parameter(torch.Tensor([-2]))

    def forward(self, x):
        is_used = self.raw_is_used.sigmoid()
        y = self.inner(x)
        return is_used * y + (1 - is_used) * x
