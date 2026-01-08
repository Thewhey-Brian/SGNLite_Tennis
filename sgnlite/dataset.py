"""
Dataset utilities for SGNLite.

Provides data loading, preprocessing, and augmentation for skeleton sequences.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# COCO-17 keypoint indices for left-right swap
COCO17_LR_SWAP = np.array([
    0,   # nose -> nose
    2,   # left_eye -> right_eye
    1,   # right_eye -> left_eye
    4,   # left_ear -> right_ear
    3,   # right_ear -> left_ear
    6,   # left_shoulder -> right_shoulder
    5,   # right_shoulder -> left_shoulder
    8,   # left_elbow -> right_elbow
    7,   # right_elbow -> left_elbow
    10,  # left_wrist -> right_wrist
    9,   # right_wrist -> left_wrist
    12,  # left_hip -> right_hip
    11,  # right_hip -> left_hip
    14,  # left_knee -> right_knee
    13,  # right_knee -> left_knee
    16,  # left_ankle -> right_ankle
    15,  # right_ankle -> left_ankle
], dtype=np.int64)

COCO17_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


def normalize_poses(poses_xy: np.ndarray, method: str = "bbox") -> np.ndarray:
    """
    Normalize pose coordinates.

    Args:
        poses_xy: Pose coordinates of shape [T, V, 2]
        method: Normalization method ("bbox", "std", or "none")

    Returns:
        Normalized poses of shape [T, V, 2]
    """
    if method == "none":
        return poses_xy

    # Get valid (non-zero) points
    valid_mask = (poses_xy[..., 0] != 0) | (poses_xy[..., 1] != 0)
    if valid_mask.sum() == 0:
        return poses_xy

    valid_pts = poses_xy[valid_mask]

    # Center at mean
    center = valid_pts.mean(axis=0, keepdims=True)
    out = poses_xy - center

    if method == "bbox":
        # Scale by bounding box diagonal
        bbox_min = valid_pts.min(axis=0)
        bbox_max = valid_pts.max(axis=0)
        scale = np.linalg.norm(bbox_max - bbox_min) + 1e-6
    elif method == "std":
        # Scale by standard deviation
        scale = valid_pts.std() + 1e-6
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    out = out / scale
    return out


def compute_velocity(poses_xy: np.ndarray) -> np.ndarray:
    """
    Compute velocity (temporal derivative) of poses.

    Args:
        poses_xy: Pose coordinates of shape [T, V, 2]

    Returns:
        Velocity of shape [T, V, 2]
    """
    T = poses_xy.shape[0]
    velocity = np.zeros_like(poses_xy)

    if T > 1:
        velocity[:-1] = poses_xy[1:] - poses_xy[:-1]
        velocity[-1] = velocity[-2]  # Copy last velocity

    return velocity


def pad_or_truncate(seq: np.ndarray, target_len: int) -> np.ndarray:
    """
    Pad or truncate sequence to target length.

    Args:
        seq: Sequence of shape [T, ...]
        target_len: Target length

    Returns:
        Sequence of shape [target_len, ...]
    """
    t = seq.shape[0]

    if t == target_len:
        return seq
    elif t > target_len:
        # Center crop
        start = (t - target_len) // 2
        return seq[start:start + target_len]
    else:
        # Pad by repeating last frame
        pad = np.repeat(seq[-1:], target_len - t, axis=0)
        return np.concatenate([seq, pad], axis=0)


class TennisSwingDataset(Dataset):
    """
    Dataset for tennis swing classification.

    Loads skeleton sequences from .npz files and applies preprocessing/augmentation.
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        target_frames: int = 20,
        use_velocity: bool = True,
        normalize: str = "bbox",
        augment: bool = False,
        flip_prob: float = 0.5,
        noise_std: float = 0.005,
        temporal_jitter: int = 2
    ):
        """
        Args:
            manifest_df: DataFrame with columns ['feature_path', 'label', ...]
            target_frames: Number of frames per sample
            use_velocity: Whether to include velocity features
            normalize: Normalization method ("bbox", "std", "none")
            augment: Whether to apply augmentation
            flip_prob: Probability of horizontal flip
            noise_std: Standard deviation of Gaussian noise (as fraction of scale)
            temporal_jitter: Max frames for temporal shift
        """
        self.df = manifest_df.reset_index(drop=True)
        self.target_frames = target_frames
        self.use_velocity = use_velocity
        self.normalize = normalize
        self.augment = augment
        self.flip_prob = flip_prob
        self.noise_std = noise_std
        self.temporal_jitter = temporal_jitter

    def __len__(self) -> int:
        return len(self.df)

    def _augment(
        self,
        poses: np.ndarray,
        width: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply data augmentation.

        Args:
            poses: Pose array of shape [T, V, 3] (x, y, conf)
            width: Frame width for horizontal flip

        Returns:
            Augmented poses
        """
        T = poses.shape[0]
        out = poses.copy()

        # 1. Horizontal flip
        if width is not None and np.random.rand() < self.flip_prob:
            out[..., 0] = width - out[..., 0]
            out = out[:, COCO17_LR_SWAP, :]

        # 2. Gaussian noise on coordinates
        if width is not None:
            sigma = self.noise_std * width
        else:
            sigma = self.noise_std * (out[..., :2].std() + 1e-6)

        noise = np.random.randn(T, out.shape[1], 2).astype(np.float32) * sigma
        out[..., :2] = out[..., :2] + noise

        # 3. Temporal jitter
        if self.temporal_jitter > 0 and np.random.rand() < 0.3:
            shift = np.random.randint(-self.temporal_jitter, self.temporal_jitter + 1)
            out = np.roll(out, shift, axis=0)

        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with:
                - 'poses': Tensor of shape [C, T, V]
                - 'label': Tensor with class index
                - 'sample_key': Sample identifier (optional)
        """
        row = self.df.iloc[idx]

        # Load data
        data = np.load(row['feature_path'])
        poses_raw = data['poses_raw']  # [T, 17, 3] - x, y, conf
        frame_info = data.get('frame_info', None)

        # Get frame dimensions for augmentation
        width = None
        if frame_info is not None and frame_info.size >= 2:
            width = float(frame_info[0, 0])

        # Augmentation (training only)
        if self.augment:
            poses_raw = self._augment(poses_raw, width)

        # Pad/truncate to target length
        poses_raw = pad_or_truncate(poses_raw, self.target_frames)

        # Extract x, y coordinates
        poses_xy = poses_raw[..., :2]  # [T, V, 2]

        # Normalize
        poses_norm = normalize_poses(poses_xy, method=self.normalize)

        # Optionally add velocity
        if self.use_velocity:
            velocity = compute_velocity(poses_norm)
            features = np.concatenate([poses_norm, velocity], axis=-1)  # [T, V, 4]
        else:
            features = poses_norm  # [T, V, 2]

        # Transpose to [C, T, V]
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)

        result = {
            'poses': torch.from_numpy(features),
            'label': torch.tensor(int(row['label']), dtype=torch.long),
        }

        # Add optional metadata
        if 'sample_key' in row:
            result['sample_key'] = row['sample_key']
        if 'hit_type' in row:
            result['hit_type'] = row['hit_type']

        return result


def create_balanced_sampler(
    labels: np.ndarray,
    num_classes: int,
    power: float = 0.5
) -> WeightedRandomSampler:
    """
    Create a balanced sampler for imbalanced datasets.

    Args:
        labels: Array of class labels
        num_classes: Total number of classes
        power: Power for inverse frequency weighting (0.5 = sqrt)

    Returns:
        WeightedRandomSampler instance
    """
    class_counts = np.bincount(labels, minlength=num_classes)
    class_weights = 1.0 / (class_counts ** power + 1e-6)
    sample_weights = class_weights[labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )


def compute_class_weights(
    labels: np.ndarray,
    num_classes: int,
    scale: float = 1.0
) -> torch.Tensor:
    """
    Compute class weights for loss function.

    Args:
        labels: Array of class labels
        num_classes: Total number of classes
        scale: Scaling factor for weights

    Returns:
        Tensor of class weights
    """
    class_counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)

    # Inverse frequency weighting
    raw_weights = total / (num_classes * class_counts + 1e-6)

    # Apply sqrt to reduce extremes
    weights = np.sqrt(raw_weights)

    # Scale
    weights = 1.0 + scale * (weights - 1.0)

    return torch.tensor(weights, dtype=torch.float32)


def split_by_group(
    df: pd.DataFrame,
    group_col: str = 'group_id',
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by groups (no group leakage between train/val).

    Args:
        df: Input DataFrame
        group_col: Column containing group identifiers
        val_ratio: Validation set ratio
        seed: Random seed

    Returns:
        Tuple of (train_df, val_df)
    """
    rng = np.random.RandomState(seed)

    groups = df[group_col].unique().tolist()
    rng.shuffle(groups)

    total = len(df)
    target_val = int(round(total * val_ratio))

    val_groups = []
    val_count = 0

    for g in groups:
        g_n = len(df[df[group_col] == g])
        if val_count + g_n <= target_val:
            val_groups.append(g)
            val_count += g_n
        if val_count >= target_val * 0.9:
            break

    val_df = df[df[group_col].isin(val_groups)].copy()
    train_df = df[~df[group_col].isin(val_groups)].copy()

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def create_dataloaders(
    manifest_csv: str,
    batch_size: int = 128,
    target_frames: int = 20,
    use_velocity: bool = True,
    val_ratio: float = 0.15,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, torch.Tensor, Dict]:
    """
    Create train and validation dataloaders.

    Args:
        manifest_csv: Path to manifest CSV file
        batch_size: Batch size
        target_frames: Number of frames per sample
        use_velocity: Whether to use velocity features
        val_ratio: Validation set ratio
        num_workers: Number of data loading workers
        seed: Random seed

    Returns:
        Tuple of (train_loader, val_loader, class_weights, label_mapping)
    """
    # Load manifest
    df = pd.read_csv(manifest_csv)
    df = df[df['status'].isin(['processed', 'existed'])].copy()

    # Infer group IDs for stratified split
    def infer_group(clip_path: str) -> str:
        p = Path(clip_path)
        parts = p.stem.split('_')
        if len(parts) >= 3:
            base = '_'.join(parts[:-2])
        elif len(parts) >= 2:
            base = parts[0]
        else:
            base = p.stem
        return f"{p.parent.name}/{base}"

    df['group_id'] = df['clip_path'].apply(infer_group)

    # Split
    train_df, val_df = split_by_group(df, 'group_id', val_ratio, seed)

    # Load label mapping
    features_dir = Path(manifest_csv).parent
    label_mapping_path = features_dir / "label_mapping.json"
    if label_mapping_path.exists():
        with open(label_mapping_path, 'r') as f:
            label_mapping = json.load(f)
    else:
        # Infer from data
        num_classes = df['label'].nunique()
        label_mapping = {
            'num_classes': num_classes,
            'label_to_idx': {str(i): i for i in range(num_classes)},
            'idx_to_label': {str(i): str(i) for i in range(num_classes)}
        }

    num_classes = label_mapping['num_classes']

    # Compute class weights
    class_weights = compute_class_weights(
        train_df['label'].values, num_classes, scale=0.3
    )

    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    print(f"Class distribution (train): {np.bincount(train_df['label'].values, minlength=num_classes)}")

    # Create datasets
    in_channels = 4 if use_velocity else 2

    train_dataset = TennisSwingDataset(
        train_df,
        target_frames=target_frames,
        use_velocity=use_velocity,
        normalize="bbox",
        augment=True
    )

    val_dataset = TennisSwingDataset(
        val_df,
        target_frames=target_frames,
        use_velocity=use_velocity,
        normalize="bbox",
        augment=False
    )

    # Create samplers
    train_sampler = create_balanced_sampler(
        train_df['label'].values, num_classes, power=0.5
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, class_weights, label_mapping


if __name__ == "__main__":
    # Test normalization
    poses = np.random.randn(20, 17, 2).astype(np.float32) * 100 + 500
    normalized = normalize_poses(poses, method="bbox")
    print(f"Original range: [{poses.min():.1f}, {poses.max():.1f}]")
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")

    # Test velocity
    velocity = compute_velocity(normalized)
    print(f"Velocity shape: {velocity.shape}")
