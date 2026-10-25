from torch import nn

from ..util import GLOBAL_CONFIG


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(self.dropout(out))))
        return out + identity


def build_resnet1d_blocks(base_channels, num_blocks, kernel_size, dropout):
    blocks = []
    in_ch = base_channels
    for block_idx in range(num_blocks):
        widen = block_idx // 4
        out_ch = base_channels * (2 ** widen)
        stride = 2 if (block_idx % 2 == 1) else 1
        blocks.append(
            ResidualBlock1D(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=stride,
                dropout=dropout,
            )
        )
        in_ch = out_ch
    return nn.Sequential(*blocks), in_ch


class AFResNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = GLOBAL_CONFIG["models"]["resnet1d"]
        self.in_channels = cfg["in_channels"]
        self.base_channels = cfg["base_channels"]
        self.num_blocks = cfg["num_blocks"]
        self.output_steps = cfg["output_steps"]
        self.kernel_size = cfg["kernel_size"]
        self.dropout = cfg["dropout"]

        self.stem = nn.Conv1d(
            in_channels=len(self.in_channels),
            out_channels=self.base_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=(self.kernel_size - 1) // 2,
            bias=False,
        )

        self.blocks, in_ch = build_resnet1d_blocks(
            self.base_channels, self.num_blocks, self.kernel_size, self.dropout
        )
        self.final_bn = nn.BatchNorm1d(in_ch)
        self.final_relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool1d(self.output_steps)
        self.head = nn.Conv1d(in_ch, 1, kernel_size=1, bias=True)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.stem(x[:, self.in_channels, :])
        out = self.blocks(out)
        out = self.final_relu(self.final_bn(out))
        out = self.pool(out)
        out = self.head(out).squeeze(1)
        if self.output_steps == 1:
            out = out.squeeze(-1)
        return out
