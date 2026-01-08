#!/usr/bin/env python3
"""
Training script for SGNLite.

Usage:
    python scripts/train.py --config configs/sgnlite_base.yaml
    python scripts/train.py --manifest /path/to/index.csv --epochs 50
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from sgnlite.model import SGNLite, create_sgnlite
from sgnlite.dataset import create_dataloaders, compute_class_weights


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def warmup_lr(epoch: int, warmup_epochs: int, base_lr: float) -> float:
    """Compute learning rate with warmup."""
    if epoch >= warmup_epochs:
        return base_lr
    return base_lr * max(0.1, (epoch + 1) / warmup_epochs)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    idx_to_label: Dict
) -> Tuple[float, float, Dict]:
    """
    Evaluate model on a dataset.

    Returns:
        Tuple of (loss, accuracy, per_class_accuracy)
    """
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    all_preds = []
    all_labels = []

    for batch in loader:
        x = batch['poses'].to(device, non_blocking=True)
        y = batch['label'].to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    # Per-class accuracy
    per_class_acc = {}
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    for cls_idx in range(num_classes):
        cls_mask = all_labels == cls_idx
        if cls_mask.sum() > 0:
            cls_correct = (all_preds[cls_mask] == cls_idx).sum()
            label_name = idx_to_label.get(str(cls_idx), str(cls_idx))
            per_class_acc[label_name] = float(cls_correct / cls_mask.sum())

    return total_loss / max(1, total), correct / max(1, total), per_class_acc


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_every: int = 50
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_correct = 0
    seen = 0

    for i, batch in enumerate(loader, 1):
        x = batch['poses'].to(device, non_blocking=True)
        y = batch['label'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        running_correct += (preds == y).sum().item()
        seen += y.size(0)

        if i % log_every == 0:
            print(f"  [{i}/{len(loader)}] loss={running_loss/seen:.4f} acc={running_correct/seen:.4f}")

    return running_loss / seen, running_correct / seen


def train(
    manifest_csv: str,
    output_dir: str,
    model_config: str = "base",
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 5,
    target_frames: int = 20,
    use_velocity: bool = True,
    label_smoothing: float = 0.05,
    seed: int = 42,
    num_workers: int = 4,
    device: str = "cuda"
):
    """
    Main training function.

    Args:
        manifest_csv: Path to manifest CSV
        output_dir: Directory to save checkpoints
        model_config: Model configuration ("tiny", "small", "base", "large")
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        weight_decay: Weight decay
        warmup_epochs: Number of warmup epochs
        target_frames: Number of frames per sample
        use_velocity: Whether to use velocity features
        label_smoothing: Label smoothing factor
        seed: Random seed
        num_workers: Data loading workers
        device: Device to train on
    """
    # Setup
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    print("\nLoading data...")
    train_loader, val_loader, class_weights, label_mapping = create_dataloaders(
        manifest_csv,
        batch_size=batch_size,
        target_frames=target_frames,
        use_velocity=use_velocity,
        num_workers=num_workers,
        seed=seed
    )

    num_classes = label_mapping['num_classes']
    idx_to_label = label_mapping['idx_to_label']
    in_channels = 4 if use_velocity else 2

    print(f"Classes: {label_mapping['label_to_idx']}")

    # Model
    print(f"\nCreating SGNLite-{model_config}...")
    model = create_sgnlite(model_config, num_classes=num_classes, in_channels=in_channels)
    model = model.to(device)
    print(f"Parameters: {model.get_num_params():,}")

    # Loss and optimizer
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs
    )

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    best_val_acc = 0.0
    best_state = None
    history = []

    save_path = os.path.join(output_dir, "sgnlite_best.pt")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Adjust LR for warmup
        if epoch <= warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr(epoch - 1, warmup_epochs, lr)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Step scheduler after warmup
        if epoch > warmup_epochs:
            scheduler.step()

        # Evaluate
        val_loss, val_acc, per_class_acc = evaluate(
            model, val_loader, criterion, device, num_classes, idx_to_label
        )

        dt = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:02d}/{epochs} | {dt:.1f}s | lr={current_lr:.6f}")
        print(f"  Train: loss={train_loss:.4f} acc={train_acc:.4f}")
        print(f"  Val:   loss={val_loss:.4f} acc={val_acc:.4f}")
        print(f"  Per-class: {per_class_acc}")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'per_class_acc': per_class_acc,
            'lr': current_lr
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            torch.save({
                'state_dict': best_state,
                'val_acc': best_val_acc,
                'epoch': epoch,
                'config': {
                    'model_config': model_config,
                    'in_channels': in_channels,
                    'num_joints': 17,
                    'num_classes': num_classes,
                    'target_frames': target_frames,
                    'use_velocity': use_velocity
                },
                'label_mapping': label_mapping
            }, save_path)

            print(f"  -> Saved best model (val_acc={best_val_acc:.4f})")

    # Save training history
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Save final model
    final_path = os.path.join(output_dir, "sgnlite_final.pt")
    torch.save({'state_dict': model.state_dict()}, final_path)

    print(f"\nTraining complete!")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoints saved to: {output_dir}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train SGNLite model")

    # Data
    parser.add_argument("--manifest", required=True, help="Path to manifest CSV")
    parser.add_argument("--output", default="./checkpoints", help="Output directory")

    # Model
    parser.add_argument("--model", default="base",
                       choices=["tiny", "small", "base", "large"],
                       help="Model configuration")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--no-velocity", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.05)

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    train(
        manifest_csv=args.manifest,
        output_dir=args.output,
        model_config=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup,
        target_frames=args.frames,
        use_velocity=not args.no_velocity,
        label_smoothing=args.label_smoothing,
        seed=args.seed,
        num_workers=args.workers,
        device=args.device
    )


if __name__ == "__main__":
    main()
