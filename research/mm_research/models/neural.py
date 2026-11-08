"""秒级因子序列的轻量神经基线，模型只接收截止当前时点的历史窗口。"""

import math

import torch
from torch import nn
from torch.nn import functional as functional


class FactorNetwork(nn.Module):
    """共享回归头的MLP、GRU、LSTM、TCN、Transformer与时间线性基线。"""

    # 【核心逻辑135】按具名实验建立网络，所有模型输出一个标准化未来收益。
    def __init__(self, family: str, features: int, sequence: int, hidden: int):
        """输入模型种类、含缺失标记的列数、历史秒数和隐藏宽度。"""
        super().__init__()
        self.family = family
        self.sequence = sequence
        if family in ["gru", "lstm", "tcn", "transformer"]:
            self.projection = nn.Linear(features, hidden)
        if family == 'mlp':
            self.encoder = nn.Sequential(nn.Linear(features, hidden * 2), nn.GELU(), nn.Dropout(.1),
                                         nn.Linear(hidden * 2, hidden), nn.GELU())
        elif family in ['gru', 'lstm']:
            recurrent = nn.GRU if family == 'gru' else nn.LSTM
            self.encoder = recurrent(hidden, hidden, num_layers=2, batch_first=True, dropout=.1)
        elif family == 'tcn':
            self.encoder = nn.ModuleList([nn.Conv1d(hidden, hidden, kernel_size=3, dilation=dilation)
                                          for dilation in [1, 2, 4, 8]])
        elif family == 'transformer':
            layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=hidden * 2,
                                                dropout=.1, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
            position = torch.arange(sequence, dtype=torch.float32).unsqueeze(1)
            frequency = torch.exp(torch.arange(0, hidden, 2, dtype=torch.float32) * (-math.log(10000) / hidden))
            encoding = torch.zeros(sequence, hidden)
            encoding[:, 0::2] = torch.sin(position * frequency)
            encoding[:, 1::2] = torch.cos(position * frequency)
            self.register_buffer('position', encoding)
        elif family == 'temporal_linear':
            self.encoder = nn.Linear(sequence * features, hidden)
        else:
            raise ValueError(f'未知深度基线 {family}')
        self.head = nn.Linear(hidden, 1)

    # 【核心逻辑137】从完整过去序列提取最后时点表示，输出当前时点预测。
    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """输入batch×过去秒数×列，序列不含任何未来行，返回batch预测。"""
        if self.family == 'mlp':
            encoded = self.encoder(history[:, -1])
        elif self.family == 'temporal_linear':
            encoded = self.encoder(history.flatten(start_dim=1))
        elif self.family in ['gru', 'lstm']:
            encoded, _ = self.encoder(self.projection(history))
            encoded = encoded[:, -1]
        elif self.family == 'tcn':
            encoded = self.projection(history).transpose(1, 2)
            for convolution in self.encoder:
                padding = 2 * convolution.dilation[0]
                encoded = encoded + functional.gelu(convolution(functional.pad(encoded, (padding, 0))))
            encoded = encoded[:, :, -1]
        else:
            encoded = self.encoder(self.projection(history) + self.position)[..., -1, :]
        return self.head(encoded).squeeze(-1)
