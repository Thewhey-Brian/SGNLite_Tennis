# SGNLite

**SGNLite: A Lightweight Transformer for Skeleton-Based Tennis Swing Detection**

SGNLite is a lightweight transformer model for skeleton-based action recognition, specifically designed for tennis swing detection and classification. It processes 2D pose sequences extracted from video to detect and classify different types of tennis strokes in real-time.

## Pipeline Overview

![SGNLite Pipeline](assets/pipeline.png)

## Key Advantages

- **Graph-free design**: Learns joint relationships through attention - no predefined skeleton topology needed
- **2D input only**: Works with standard monocular video, no depth sensors required
- **Real-time**: 4,000+ FPS inference on GPU
- **Flexible**: Easily adaptable to other skeleton formats and action recognition tasks
- **End-to-end**: Complete pipeline from raw video to swing detection

## Supported Classes

| Class | Description |
|-------|-------------|
| `ground_stroke` | Forehand and backhand ground strokes |
| `serve` | Tennis serve |
| `volley` | Net volleys |
| `overhead` | Overhead smash |
| `feed` | Ball feed/toss |
| `negative` | No swing detected |

## Installation

```bash
# Clone the repository
git clone https://github.com/Thewhey-Brian/SGNLite_Tennis.git
cd SGNLite_Tennis

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- ultralytics (for YOLO-pose)
- opencv-python
- numpy, scipy, tqdm

## Quick Start

### Inference on Video

```python
from sgnlite.inference import run_inference_pipeline

results = run_inference_pipeline(
    video_path="tennis_match.mp4",
    checkpoint_path="checkpoints/sgnlite_best.pt",
    output_video="output_annotated.mp4"
)

for det in results['detections']:
    print(f"Frame {det['frame']}: {det['class']} ({det['confidence']:.2f})")
```

### Using the Model Directly

```python
import torch
from sgnlite.model import SGNLite

# Create model
model = SGNLite(in_channels=2, num_joints=17, num_classes=6)

# Forward pass
x = torch.randn(4, 2, 20, 17)  # [batch, xy, frames, joints]
logits = model(x)  # [batch, 6]
```

### Training

```bash
python scripts/train.py \
    --manifest /path/to/features/index.csv \
    --output ./checkpoints \
    --epochs 50
```

## Architecture

![SGNLite Architecture](assets/architecture.png)

```
Input: [N, 2, T, 17]  (batch, xy, frames, joints)
         ↓
    Linear Embedding (2 → 258)
         ↓
    + Learnable Joint Embeddings (17 joints)
         ↓
    + Sinusoidal Positional Encoding
         ↓
    6× Transformer Blocks (6 heads each)
         ↓
    Global Average Pooling
         ↓
    Classification Head → 6 classes
```

## Performance

| Metric | Value |
|--------|-------|
| Validation Accuracy | 81.2% |
| Parameters | 3.2M |
| Inference Speed | 4,262 FPS |

## Project Structure

```
SGNLite_release/
├── sgnlite/
│   ├── __init__.py
│   ├── model.py          # SGNLite architecture
│   ├── dataset.py        # Data loading
│   └── inference.py      # Inference pipeline
├── scripts/
│   └── train.py          # Training script
├── configs/
│   └── sgnlite_base.yaml
├── notebooks/
│   └── demo_inference.ipynb
├── assets/               # Figures
└── README.md
```

## Citation

```bibtex
@article{sgnlite2024,
  title={SGNLite: A Lightweight Transformer for Skeleton-Based Tennis Swing Detection},
  author={Your Name},
  year={2024}
}
```

## License

MIT License
