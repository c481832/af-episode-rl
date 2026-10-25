import torch.nn.functional as F
from torch import nn

from ..util import GLOBAL_CONFIG


class AFCNNLSTM1D(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = GLOBAL_CONFIG["models"]["cnn_lstm"]
        self.in_channels = cfg["in_channels"]
        cnn_channels = cfg["cnn_channels"]
        kernel_size = cfg["cnn_kernel_size"]
        padding = cfg["cnn_padding"]
        pool_kernel_size = cfg["cnn_pool_kernel_size"]
        pool_stride = cfg["cnn_pool_stride"]
        lstm_hidden_size = cfg["lstm_hidden_size"]
        dense_units = cfg["dense_units"]
        dropout = cfg["dropout"]

        cnn_layers, in_ch = [], len(self.in_channels)
        for out_ch in cnn_channels:
            cnn_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        self.lstm = nn.LSTM(input_size=cnn_channels[-1], hidden_size=lstm_hidden_size, batch_first=True)
        self.dense = nn.LazyLinear(dense_units)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(dense_units, 1)

    def forward(self, x):
        out = self.cnn(x[:, self.in_channels, :])
        out, _ = self.lstm(out.transpose(1, 2))
        out = out.flatten(1)
        out = F.relu(self.dense(out))
        out = self.dropout(out)
        return self.head(out).squeeze(-1)
