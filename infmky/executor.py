from torch import nn

from .layer import Conv1dBlock, Skip


class Executor(nn.Sequential):
    def __init__(self, in_channels, in_width, out_channels, out_width):
        blocks = []
        c = 128
        mid_width = 2

        head = nn.Sequential(
            nn.Conv1d(in_channels, c, 3, 1, 1),
            nn.BatchNorm1d(c),
        )
        blocks.append(head)

        width = in_width
        while mid_width < width:
            block = nn.Sequential(
                Conv1dBlock(c, c),
                Skip(Conv1dBlock(c, c)),
                Conv1dBlock(c, c),
                Skip(Conv1dBlock(c, c)),
                nn.MaxPool1d(2),
            )
            blocks.append(block)
            width //= 2
        assert width == mid_width

        for i in range(4):
            block = nn.Sequential(
                Conv1dBlock(c, c),
                Skip(Conv1dBlock(c, c)),
                Conv1dBlock(c, c),
                Skip(Conv1dBlock(c, c)),
            )
            blocks.append(block)

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
            nn.Conv1d(c, out_channels, 3, 1, 1),
        )
        blocks.append(tail)

        super().__init__(*blocks)
