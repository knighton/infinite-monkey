import torch
from torch import nn

from .layer import Conv1dBlock, Reshape, Skip


class Generator(nn.Module):
    def __init__(self, z_dim, out_channels, out_width):
        super().__init__()

        c = z_dim

        self.z_head = nn.Sequential(
            nn.Linear(c, c * 2),
            nn.BatchNorm1d(c * 2),

            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(c * 2, c * 4),
            nn.BatchNorm1d(c * 4),

            Reshape(c, 4),

            Conv1dBlock(c, c),
            Skip(Conv1dBlock(c, c)),
            Conv1dBlock(c, c),
            Skip(Conv1dBlock(c, c)),
        )

        d = 8
        blocks = []
        head = nn.Sequential(
            nn.Conv1d(1, d, 5, 1, 2),
            nn.BatchNorm1d(d),
        )
        blocks.append(head)
        width = out_width
        while 4 < width:
            block = nn.Sequential(
                Conv1dBlock(d, d),
                nn.MaxPool1d(2),
            )
            blocks.append(block)
            width //= 2
        assert width == 4
        self.mask_head = nn.Sequential(*blocks)

        blocks = []

        head = nn.Sequential(
            Conv1dBlock(c + d, c),
            Skip(Conv1dBlock(c, c)),
        )
        blocks.append(head)

        width = 4
        while width < out_width:
            block = nn.Sequential(
                nn.Upsample(scale_factor=2),
                Conv1dBlock(c, c),
                Skip(Conv1dBlock(c, c)),
                Conv1dBlock(c, c),
                Skip(Conv1dBlock(c, c)),
            )
            blocks.append(block)
            width *= 2
        assert width == out_width

        tail = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(c, out_channels, 5, 1, 2),
        )
        blocks.append(tail)

        self.body = nn.Sequential(*blocks)

    def forward(self, z, mask):
        """
        z: (batch size, z dim)
        mask: (batch size, out width)
        y: (batch size, out channels, out width)
        """
        z = self.z_head(z)
        mask = self.mask_head(mask.unsqueeze(1))
        x = torch.cat([z, mask], 1)
        return self.body(x)
