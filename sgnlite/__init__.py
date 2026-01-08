"""
SGNLite: Lightweight Transformer for Skeleton-Based Action Recognition

A lightweight transformer model for tennis swing detection and classification
using 2D pose sequences extracted from video.
"""

__version__ = "0.1.0"
__author__ = "Brian"

from .model import SGNLite, SGNLiteBlock, PositionalEncoding
from .dataset import TennisSwingDataset, normalize_poses, compute_velocity

__all__ = [
    "SGNLite",
    "SGNLiteBlock",
    "PositionalEncoding",
    "TennisSwingDataset",
    "normalize_poses",
    "compute_velocity",
]
