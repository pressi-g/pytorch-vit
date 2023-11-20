""" Patch Embedding """

# Standard imports
import math

# Machine learning imports
import torch
from torch import nn, Tensor


class PatchEmbedding(nn.Module):
    """
    Class that creates patch embeddings for images
    (image to patch embeddings).

    Parameters:
        image_size (int): Size of image
        patch_size (int): Size of patch
        in_channels (int): Number of input channels
        embed_dim (int): Embedding dimension

    Returns:
        x (Tensor): Patch embeddings
    """

    def __init__(
        self, image_size: int, patch_size: int, in_channels: int, embed_dim: int
    ):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x).flatten(2).transpose(1, 2)
        return x


class DynamicPositionEmbedding(nn.Module):
    """
    Class that creates dynamic position embeddings for images using sinusoidal positional encodings.

    Parameters:
        channels (int): Number of channels
        num_patches (int): Number of patches
        embed_dim (int): Embedding dimension

    Returns:
        x (Tensor): Output of dynamic position embedding layer
    """

    def __init__(self, num_patches, embed_dim):
        super(DynamicPositionEmbedding, self).__init__()

        self.pos_embedding = self.create_pos_embedding(num_patches + 1, embed_dim)

    def create_pos_embedding(self, num_patches, embed_dim):
        position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pos_embedding = torch.zeros(num_patches, embed_dim)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        pos_embedding = pos_embedding.unsqueeze(0).transpose(0, 1)
        return nn.Parameter(pos_embedding, requires_grad=False)

    def forward(self, x):
        x = x + self.pos_embedding[: x.size(1)].squeeze(
            1
        )  # Add position embeddings to both patch embeddings and CLS token.
        return x
