from torch import nn

from .layer import Conv1dBlock, Reshape, Skip


class Generator(nn.Sequential):
    def __init__(self, z_dim, out_channels, out_width):
        blocks = []

        c = z_dim
        head = nn.Sequential(
            nn.Linear(z_dim, c * 2),
            nn.BatchNorm1d(c * 2),

            nn.ReLU(),
            nn.Linear(c * 2, c * 4),
            nn.BatchNorm1d(c * 4),

            Reshape(c, 4),

            Skip(Conv1dBlock(c, c)),
            Skip(Conv1dBlock(c, c)),
        )
        blocks.append(head)

        width = 4
        while width < out_width:
            block = nn.Sequential(
                nn.Upsample(scale_factor=2),
                Skip(Conv1dBlock(c, c)),
                Skip(Conv1dBlock(c, c)),
            )
            blocks.append(block)
            width *= 2
        assert width == out_width

        tail = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(c, out_channels, 5, 1, 2),
        )
        blocks.append(tail)

        super().__init__(*blocks)
