"""
SGNLite Model Architecture

A lightweight transformer for skeleton-based action recognition that treats
(time, joint) pairs as tokens with learnable joint embeddings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, dim: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.pe[:seq_len]


class SGNLiteBlock(nn.Module):
    """
    Transformer block with pre-norm architecture.

    Structure:
        x -> LayerNorm -> MultiHeadAttention -> + -> LayerNorm -> MLP -> +
    """

    def __init__(self, dim: int, num_heads: int = 6, mlp_ratio: float = 2.0, dropout: float = 0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out

        # MLP with pre-norm
        x = x + self.mlp(self.norm2(x))
        return x


class SGNLite(nn.Module):
    """
    SGNLite: Lightweight Skeleton Graph Network using Transformers.

    Architecture:
        1. Linear embedding of joint coordinates
        2. Add learnable joint embeddings (joint identity)
        3. Add sinusoidal positional encoding (temporal position)
        4. Stack of transformer blocks
        5. Global average pooling
        6. Classification head

    Input shape: [N, C, T, V]
        - N: Batch size
        - C: Channels (2 for x,y or 4 for x,y,vx,vy)
        - T: Temporal frames
        - V: Number of joints (vertices)

    Output shape: [N, num_classes]

    Default configuration:
        - embed_dim: 258
        - depth: 6 transformer blocks
        - num_heads: 6 attention heads
        - mlp_ratio: 2.0
        - dropout: 0.3
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_joints: int = 17,
        num_classes: int = 6,
        embed_dim: int = 258,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 2.0,
        dropout: float = 0.3
    ):
        """
        Args:
            in_channels: Number of input channels (2 for x,y; 4 for x,y,vx,vy)
            num_joints: Number of skeleton joints (17 for COCO)
            num_classes: Number of output classes
            embed_dim: Transformer embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension multiplier
            dropout: Dropout rate
        """
        super().__init__()

        self.in_channels = in_channels
        self.num_joints = num_joints
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Coordinate embedding
        self.coord_embed = nn.Linear(in_channels, embed_dim)

        # Learnable joint embeddings (joint identity)
        self.joint_embed = nn.Embedding(num_joints, embed_dim)

        # Temporal positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=1024)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SGNLiteBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [N, C, T, V]

        Returns:
            Logits tensor of shape [N, num_classes]
        """
        N, C, T, V = x.shape

        # Reshape: [N, C, T, V] -> [N, T, V, C]
        x = x.permute(0, 2, 3, 1).contiguous()

        # Coordinate embedding: [N, T, V, C] -> [N, T, V, D]
        x = self.coord_embed(x)

        # Add joint embeddings
        joint_ids = torch.arange(V, device=x.device).view(1, 1, V)
        joint_emb = self.joint_embed(joint_ids).expand(N, T, V, -1)
        x = x + joint_emb

        # Add positional encoding
        pos_enc = self.pos_encoding(T).to(x.device).view(1, T, 1, -1)
        x = x + pos_enc

        # Flatten spatial-temporal: [N, T, V, D] -> [N, T*V, D]
        x = x.view(N, T * V, -1)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Layer norm
        x = self.norm(x)

        # Global average pooling: [N, T*V, D] -> [N, D]
        x = x.mean(dim=1)

        # Classification
        x = self.head(x)

        return x

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = SGNLite(in_channels=2, num_classes=6)
    print(f"SGNLite: {model.get_num_params():,} parameters")

    # Test forward pass
    x = torch.randn(4, 2, 20, 17)  # [batch, channels, frames, joints]
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
