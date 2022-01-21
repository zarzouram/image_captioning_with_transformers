import math
import torch
from torch import Tensor
from torch import nn

"""
reference:
https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model
"""


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        param:
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        self.d_model = d_model

        # create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # not a parameter, but should be part of the modules state.
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size(1)]
        return x
