from torch import nn

from .layer import Conv1dBlock, Skip


class Executor(nn.Sequential):
    def __init__(self, in_channels, in_width, out_channels, out_width):
        blocks = []
        c = 128
        mid_width = 4

        head = nn.Sequential(
            nn.Conv1d(in_channels, c, 3, 1, 1),
            nn.BatchNorm1d(c),
        )
        blocks.append(head)

        width = in_width
        while mid_width < width:
            block = nn.Sequential(
                Skip(Conv1dBlock(c, c)),
                Skip(Conv1dBlock(c, c)),
                nn.MaxPool1d(2),
            )
            blocks.append(block)
            width //= 2
        assert width == mid_width

        for i in range(2):
            block = nn.Sequential(
                Skip(Conv1dBlock(c, c)),
                Skip(Conv1dBlock(c, c)),
            )
            blocks.append(block)

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
            nn.Conv1d(c, out_channels, 3, 1, 1),
        )
        blocks.append(tail)

        super().__init__(*blocks)
