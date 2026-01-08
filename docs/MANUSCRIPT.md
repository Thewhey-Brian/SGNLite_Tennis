# SGNLite: A Lightweight Transformer for Skeleton-Based Tennis Swing Detection

---

## Abstract

We present SGNLite, a lightweight transformer-based model for skeleton-based tennis swing detection and classification. SGNLite treats each (time, joint) pair as a token with learnable joint embeddings, allowing the model to discover relevant joint relationships through attention without requiring predefined skeleton graphs. Combined with YOLO-pose for 2D keypoint extraction, our end-to-end pipeline achieves 81% accuracy on 6-class tennis stroke classification while running at over 4,000 FPS on GPU. All code and trained models are publicly available.

---

## 1. Introduction

Tennis coaching and analytics benefit greatly from automatic swing detection and classification. SGNLite addresses this need with a lightweight, graph-free transformer architecture that:

1. **Works with 2D video**: Only requires standard monocular camera footage
2. **Learns joint relationships**: No predefined skeleton topology needed
3. **Runs in real-time**: Over 4,000 FPS inference speed
4. **Classifies 6 stroke types**: ground_stroke, serve, volley, overhead, feed, negative

---

## 2. Method

### 2.1 Pipeline Overview

```
Video → YOLO-pose → SGNLite → Swing Detection
```

1. **Video Preprocessing**: Standardize to 1080p @ 15fps
2. **Pose Extraction**: YOLO11-pose extracts 17 COCO keypoints per frame
3. **Classification**: SGNLite classifies 20-frame sliding windows
4. **Post-processing**: Gaussian smoothing + peak detection for swing events

### 2.2 SGNLite Architecture

```
Input: [batch, 2, 20, 17]  (xy coordinates, 20 frames, 17 joints)
         ↓
    Linear Embedding (2 → 258)
         ↓
    + Learnable Joint Embeddings (per joint identity)
         ↓
    + Sinusoidal Positional Encoding (temporal position)
         ↓
    6× Transformer Blocks (6 attention heads each)
         ↓
    Global Average Pooling
         ↓
    Classification Head → 6 classes
```

**Key Features:**
- **Joint Embeddings**: Each of 17 joints has a learnable embedding
- **Attention-based**: Discovers which joints matter for each action
- **No graph structure**: Unlike GCN methods, no adjacency matrix needed

### 2.3 Training

- **Dataset**: ~23K labeled tennis clips
- **Augmentation**: Horizontal flip, Gaussian noise, temporal jitter
- **Optimization**: AdamW with cosine annealing
- **Class balancing**: Sqrt-weighted sampling for imbalanced classes

---

## 3. Results

### Performance

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **81.2%** |
| Parameters | 3.2M |
| Inference Speed | 4,262 FPS |

### Per-Class Accuracy

| Class | Accuracy |
|-------|----------|
| negative | 91% |
| ground_stroke | 89% |
| serve | 78% |
| feed | 72% |
| volley | 61% |
| overhead | 45% |

---

## 4. Key Advantages

1. **Graph-free design**: Learns joint relationships through attention - no need to define skeleton connectivity

2. **2D input only**: Works with standard monocular video, no depth sensors or 3D reconstruction required

3. **Real-time capable**: 4,000+ FPS enables live video processing and edge deployment

4. **Flexible architecture**: Easily adaptable to other skeleton formats and action recognition tasks

5. **End-to-end pipeline**: Complete solution from raw video to detected swing events

6. **Learnable joint importance**: Attention mechanism automatically discovers which joints are relevant for each action type

---

## 5. Usage

### Inference on Video

```python
from sgnlite.inference import run_inference_pipeline

results = run_inference_pipeline(
    video_path="tennis_match.mp4",
    checkpoint_path="sgnlite_best.pt",
    output_video="annotated_output.mp4"
)

for detection in results['detections']:
    print(f"Frame {detection['frame']}: {detection['class']} ({detection['confidence']:.2f})")
```

### Training

```bash
python scripts/train.py --manifest data/index.csv --epochs 50
```

---

## 6. Conclusion

SGNLite provides an effective, real-time solution for tennis swing detection using 2D pose sequences. The attention-based architecture learns joint relationships automatically, making it adaptable and easy to deploy.

---

## References

1. Yan et al. (2018). Spatial temporal graph convolutional networks. AAAI.
2. Plizzari et al. (2021). Spatial temporal transformer network. ICPR.
3. Jocher et al. (2023). Ultralytics YOLO.

---

## Appendix: Model Configuration

| Parameter | Value |
|-----------|-------|
| embed_dim | 258 |
| depth | 6 |
| num_heads | 6 |
| mlp_ratio | 2.0 |
| dropout | 0.3 |
| input_frames | 20 |
| num_joints | 17 |
