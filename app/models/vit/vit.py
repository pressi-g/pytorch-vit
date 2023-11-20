""" ViT model for MNIST and CIFAR datasets. """

# Standard imports
import numpy as np

# Machine learning imports
import torch
from torch import nn

# Third-party imports
from einops import repeat

# Local imports
from vit.patchembedding import PatchEmbedding, DynamicPositionEmbedding
from vit.transformer_encoder import TransformerEncoderBlock


class VisionTransformer(nn.Module):
    """
    Vision transformer architecture.

    Parameters:
        image_size (int): Size of image
        patch_size (int): Size of patch
        in_channels (int): Number of input channels
        embed_dim (int): Embedding dimension
        depth (int): Depth
        heads (int): Number of heads
        mlp_dim (int): Dimension of MLP
        dropout (float): Dropout probability
        num_classes (int): Number of classes

    Returns:
        x (Tensor): Output of VisionTransformer
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        num_classes: int,
        dropout: float = 0.0,
        mlp_head_units=[2048, 1024],
    ):
        super().__init__()
        # Patch embedding layer
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        # Initializing cls_token and pos_embed with random values
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # Calculate the number of patches
        num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )  # +1 for the cls_token --> initialising to zeros seems to work better than random values

        self.dropout = nn.Dropout(dropout)
        self.transformer_encoder = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    dim=embed_dim,
                    heads=heads,
                    dim_head=embed_dim // heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, mlp_head_units[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_head_units[0], mlp_head_units[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_head_units[1], num_classes),
        )

    def forward(self, x):
        B = x.shape[0]  # batch_size
        x = self.patch_embed(x)

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=B)
        x = torch.cat((cls_tokens, x), dim=1)

        assert (
            x.shape[1] == self.pos_embed.shape[1]
        ), f"Positional embeddings don't match the input patches. x.shape: {x.shape} pos_embed.shape: {self.pos_embed.shape}"

        x = x + self.pos_embed
        x = self.dropout(x)

        for transformer_encoder in self.transformer_encoder:
            x = transformer_encoder(x)

        x = self.norm(x)
        x = x[:, 0]  # Take the cls_token representation
        x = self.mlp_head(x)

        return x


class VisionTransformerDPE(nn.Module):
    """
    Vision transformer architecture with Dynamic Position Embeddings.

    Parameters:
        image_size (int): Size of image
        patch_size (int): Size of patch
        in_channels (int): Number of input channels
        embed_dim (int): Embedding dimension
        depth (int): Depth
        heads (int): Number of heads
        mlp_dim (int): Dimension of MLP
        dropout (float): Dropout probability
        num_classes (int): Number of classes

    Returns:
        x (Tensor): Output of VisionTransformer
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        num_classes: int,
        dropout: float = 0.0,
        mlp_head_units=[2048, 1024],
    ):
        super().__init__()
        # Patch embedding layer
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        # Initializing cls_token and pos_embed with random values
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # Create the Dynamic Positional Embedding layer
        num_patches = (image_size // patch_size) ** 2
        self.dynamic_pos_embed = DynamicPositionEmbedding(num_patches, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.transformer_encoder = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    dim=embed_dim,
                    heads=heads,
                    dim_head=embed_dim // heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, mlp_head_units[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_head_units[0], mlp_head_units[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_head_units[1], num_classes),
        )

    def forward(self, x):
        B = x.shape[0]  # batch_size
        x = self.patch_embed(x)

        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=B)
        x = torch.cat((cls_tokens, x), dim=1)

        # Use the DynamicPositionEmbedding for adding the positional embeddings
        x = self.dynamic_pos_embed(x)

        x = self.dropout(x)

        for transformer_encoder in self.transformer_encoder:
            x = transformer_encoder(x)

        x = self.norm(x)
        x = x[:, 0]  # Take the cls_token representation
        x = self.mlp_head(x)

        return x
