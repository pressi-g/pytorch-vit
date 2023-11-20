""" Attention Layer """

# Machine learning imports
import torch
from torch import nn


class Attention(nn.Module):
    """
    Class that creates an attention layer.

    Parameters:
        dim (int): Dimension of input
        heads (int): Number of heads
        dim_head (int): Dimension of each head
        dropout (float): Dropout probability

    Returns:
        x (Tensor): Output of attention layer
    """

    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float = 0.0):
        super().__init__()

        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5

        # Linear layer to get Q, K, V
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Obtain Q, K, V from input tensor x
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, self.dim_head)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Calculate attention scores and apply scaling
        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale

        # Softmax to get attention weights
        attn = self.attend(dots)

        # Calculate the output tensor
        out = torch.einsum("bhij,bhjd->bhid", attn, v).reshape(B, N, C)
        out = self.proj(out)
        return out
