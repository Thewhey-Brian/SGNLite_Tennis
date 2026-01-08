"""
Inference utilities for SGNLite.

Provides end-to-end video processing: pose extraction -> classification -> visualization.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .model import SGNLite
from .dataset import normalize_poses, compute_velocity, pad_or_truncate


class PoseExtractor:
    """
    Extract 2D poses from video using YOLO-pose.

    Handles video preprocessing (resolution, fps) and pose extraction.
    """

    def __init__(
        self,
        model_name: str = "yolo11l-pose.pt",
        device: str = "cuda",
        conf_threshold: float = 0.1,
        target_fps: int = 15,
        target_resolution: Tuple[int, int] = (1920, 1080)
    ):
        """
        Args:
            model_name: YOLO model name or path
            device: Device to run on ("cuda" or "cpu")
            conf_threshold: Detection confidence threshold
            target_fps: Target video FPS
            target_resolution: Target (width, height)
        """
        self.model_name = model_name
        self.device = device
        self.conf_threshold = conf_threshold
        self.target_fps = target_fps
        self.target_w, self.target_h = target_resolution
        self.num_keypoints = 17  # COCO-17

        self._model = None

    def _load_model(self):
        """Lazy load YOLO model."""
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.model_name)
                if self.device == "cuda" and torch.cuda.is_available():
                    self._model.to("cuda")
            except ImportError:
                raise ImportError(
                    "ultralytics package required for pose extraction. "
                    "Install with: pip install ultralytics"
                )

    def _preprocess_video(self, input_path: str, output_path: str):
        """Preprocess video to target resolution and FPS."""
        try:
            import ffmpeg
            (
                ffmpeg
                .input(input_path)
                .output(
                    output_path,
                    vf=f"scale={self.target_w}:{self.target_h},fps={self.target_fps}",
                    vcodec="libx264",
                    crf=18,
                    pix_fmt="yuv420p",
                    acodec="aac",
                    audio_bitrate="128k",
                    movflags="+faststart"
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except ImportError:
            raise ImportError(
                "ffmpeg-python package required for video preprocessing. "
                "Install with: pip install ffmpeg-python"
            )

    def _extract_best_pose(self, result) -> np.ndarray:
        """Extract pose of most confident detection."""
        pose = np.zeros((self.num_keypoints, 3), dtype=np.float32)

        if (result.boxes is None or len(result.boxes) == 0 or
            result.keypoints is None or len(result.keypoints) == 0):
            return pose

        # Get most confident detection
        confs = result.boxes.conf.detach().cpu().numpy()
        idx = int(confs.argmax())

        # Extract keypoints
        xy = result.keypoints.xy[idx].detach().cpu().numpy()
        kconf = result.keypoints.conf[idx].detach().cpu().numpy()

        pose[:, 0] = xy[:, 0]
        pose[:, 1] = xy[:, 1]
        pose[:, 2] = kconf

        return pose

    @torch.no_grad()
    def extract(
        self,
        video_path: str,
        show_progress: bool = True
    ) -> Dict:
        """
        Extract poses from video.

        Args:
            video_path: Path to input video
            show_progress: Show progress bar

        Returns:
            Dictionary with:
                - 'poses': Array of shape [T, 17, 3]
                - 'frame_info': Array of shape [T, 3] (width, height, timestamp)
                - 'meta': Video metadata
                - 'preprocessed_path': Path to preprocessed video
        """
        self._load_model()

        # Create temp directory for preprocessing
        tmpdir = tempfile.TemporaryDirectory()
        preproc_path = os.path.join(tmpdir.name, "preprocessed.mp4")

        print("Preprocessing video...")
        self._preprocess_video(video_path, preproc_path)

        # Get video info
        cap = cv2.VideoCapture(preproc_path)
        T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Allocate arrays
        poses = np.zeros((T, self.num_keypoints, 3), dtype=np.float32)
        frame_info = np.zeros((T, 3), dtype=np.float32)

        # Extract poses
        print("Extracting poses...")
        cap = cv2.VideoCapture(preproc_path)

        iterator = range(T)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting poses")

        for i in iterator:
            ret, frame = cap.read()
            if not ret:
                break

            results = self._model(frame, conf=self.conf_threshold, verbose=False)
            poses[i] = self._extract_best_pose(results[0])
            frame_info[i] = (W, H, i / self.target_fps)

        cap.release()

        meta = {
            'T': T,
            'width': W,
            'height': H,
            'fps': self.target_fps,
            'original_path': video_path
        }

        return {
            'poses': poses,
            'frame_info': frame_info,
            'meta': meta,
            'preprocessed_path': preproc_path,
            '_tmpdir': tmpdir  # Keep reference to prevent cleanup
        }


class SGNLiteInference:
    """
    Inference pipeline for SGNLite model.

    Supports:
        - Single clip classification
        - Sliding window video inference
        - Video annotation with detections
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        window_size: int = 20,
        stride: int = 1,
        use_velocity: bool = True
    ):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run on
            window_size: Sliding window size (frames)
            stride: Sliding window stride
            use_velocity: Whether to use velocity features
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.window_size = window_size
        self.stride = stride
        self.use_velocity = use_velocity

        # Load model
        self.model, self.config, self.label_mapping = self._load_checkpoint(
            checkpoint_path
        )

        self.num_classes = len(self.label_mapping['idx_to_label'])
        self.idx_to_label = {
            int(k): v for k, v in self.label_mapping['idx_to_label'].items()
        }

    def _load_checkpoint(self, path: str) -> Tuple[SGNLite, Dict, Dict]:
        """Load model from checkpoint."""
        ckpt = torch.load(path, map_location=self.device)

        config = ckpt.get('config', {})
        label_mapping = ckpt.get('label_mapping', {})

        # Create model
        model = SGNLite(
            in_channels=config.get('in_channels', 4 if self.use_velocity else 2),
            num_joints=config.get('num_joints', 17),
            num_classes=config.get('num_classes', 6)
        ).to(self.device)

        # Load weights
        state_dict = ckpt.get('state_dict', ckpt)
        model.load_state_dict(state_dict)
        model.eval()

        print(f"Loaded SGNLite: {model.get_num_params():,} params")

        return model, config, label_mapping

    def _preprocess_window(self, window: np.ndarray) -> torch.Tensor:
        """
        Preprocess a pose window for inference.

        Args:
            window: Pose array of shape [T, V, 3]

        Returns:
            Tensor of shape [1, C, T, V]
        """
        # Pad if needed
        window = pad_or_truncate(window, self.window_size)

        # Get x,y coordinates
        poses_xy = window[..., :2]

        # Normalize
        poses_norm = normalize_poses(poses_xy, method="bbox")

        # Add velocity if needed
        if self.use_velocity:
            velocity = compute_velocity(poses_norm)
            features = np.concatenate([poses_norm, velocity], axis=-1)
        else:
            features = poses_norm

        # Transpose to [C, T, V]
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)

        # Add batch dimension
        return torch.from_numpy(features).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def classify_clip(self, poses: np.ndarray) -> Dict:
        """
        Classify a single pose clip.

        Args:
            poses: Pose array of shape [T, V, 3]

        Returns:
            Dictionary with predictions and probabilities
        """
        x = self._preprocess_window(poses)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(logits.argmax(dim=1).item())

        return {
            'prediction': self.idx_to_label[pred_idx],
            'prediction_idx': pred_idx,
            'confidence': float(probs[pred_idx]),
            'probabilities': {
                self.idx_to_label[i]: float(probs[i])
                for i in range(self.num_classes)
            }
        }

    @torch.no_grad()
    def process_video(
        self,
        poses: np.ndarray,
        smooth_sigma: float = 2.0,
        peak_threshold: float = 0.7,
        peak_distance: int = 15,
        show_progress: bool = True
    ) -> Dict:
        """
        Process full video with sliding window inference.

        Args:
            poses: Pose array of shape [T, V, 3]
            smooth_sigma: Gaussian smoothing sigma
            peak_threshold: Minimum peak height for detection
            peak_distance: Minimum distance between peaks (frames)
            show_progress: Show progress bar

        Returns:
            Dictionary with frame-level predictions and detected peaks
        """
        T = poses.shape[0]

        # Frame-level probabilities
        frame_probs = np.zeros((T, self.num_classes), dtype=np.float32)

        # Sliding window inference
        iterator = range(0, max(1, T - self.window_size + 1), self.stride)
        if show_progress:
            iterator = tqdm(iterator, desc="Running inference")

        for t in iterator:
            window = poses[t:t + self.window_size]
            x = self._preprocess_window(window)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            # Assign to center frame
            center = min(T - 1, t + self.window_size // 2)
            frame_probs[center] = np.maximum(frame_probs[center], probs)

        # Gaussian smoothing
        try:
            from scipy.ndimage import gaussian_filter1d
            frame_probs_smooth = np.array([
                gaussian_filter1d(frame_probs[:, i], sigma=smooth_sigma, mode='nearest')
                for i in range(self.num_classes)
            ]).T
        except ImportError:
            frame_probs_smooth = frame_probs

        # Get predictions
        frame_preds = frame_probs_smooth.argmax(axis=1)

        # Peak detection for action events
        try:
            from scipy.signal import find_peaks

            # Get non-negative class indices
            shot_classes = [
                i for i in range(self.num_classes)
                if self.idx_to_label.get(i, '') != 'negative'
            ]

            if shot_classes:
                max_shot_prob = frame_probs_smooth[:, shot_classes].max(axis=1)
            else:
                max_shot_prob = frame_probs_smooth.max(axis=1)

            peaks_idx, props = find_peaks(
                max_shot_prob,
                height=peak_threshold,
                distance=peak_distance
            )
            peaks_score = props['peak_heights']
        except ImportError:
            peaks_idx = np.array([], dtype=int)
            peaks_score = np.array([], dtype=np.float32)

        return {
            'frame_probs': frame_probs,
            'frame_probs_smooth': frame_probs_smooth,
            'frame_predictions': frame_preds,
            'peaks_idx': peaks_idx.tolist(),
            'peaks_score': peaks_score.tolist(),
            'detections': [
                {
                    'frame': int(idx),
                    'class': self.idx_to_label[frame_preds[idx]],
                    'confidence': float(frame_probs_smooth[idx, frame_preds[idx]])
                }
                for idx in peaks_idx
            ]
        }

    def annotate_video(
        self,
        input_video: str,
        output_video: str,
        poses: np.ndarray,
        results: Dict,
        fps: int = 15
    ):
        """
        Create annotated video with predictions.

        Args:
            input_video: Path to input video (preprocessed)
            output_video: Path to output video
            poses: Pose array of shape [T, V, 3]
            results: Results from process_video()
            fps: Video FPS
        """
        cap = cv2.VideoCapture(input_video)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video, fourcc, fps, (W, H))

        frame_probs = results['frame_probs_smooth']
        frame_preds = results['frame_predictions']
        peak_set = set(results['peaks_idx'])

        for i in tqdm(range(T), desc="Annotating video"):
            ret, frame = cap.read()
            if not ret:
                break

            pred_idx = frame_preds[i]
            hit_type = self.idx_to_label[pred_idx]
            prob = frame_probs[i, pred_idx]
            is_peak = i in peak_set

            # Draw label
            color = (0, 0, 255) if is_peak else (0, 200, 0)
            label = f"{'PEAK ' if is_peak else ''}{hit_type} ({prob:.2f})"
            cv2.putText(
                frame, label, (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4, cv2.LINE_AA
            )

            # Draw peak indicator
            if is_peak:
                cv2.circle(frame, (W // 2, 100), 14, (0, 0, 255), -1)

            writer.write(frame)

        cap.release()
        writer.release()
        print(f"Saved annotated video to: {output_video}")


def run_inference_pipeline(
    video_path: str,
    checkpoint_path: str,
    output_video: Optional[str] = None,
    device: str = "cuda",
    window_size: int = 20,
    smooth_sigma: float = 2.0,
    peak_threshold: float = 0.7
) -> Dict:
    """
    Run full inference pipeline on a video.

    Args:
        video_path: Path to input video
        checkpoint_path: Path to model checkpoint
        output_video: Path to output annotated video (optional)
        device: Device to run on
        window_size: Sliding window size
        smooth_sigma: Gaussian smoothing sigma
        peak_threshold: Peak detection threshold

    Returns:
        Dictionary with detections and metadata
    """
    # Extract poses
    extractor = PoseExtractor(device=device)
    pose_data = extractor.extract(video_path)

    # Run inference
    inference = SGNLiteInference(
        checkpoint_path,
        device=device,
        window_size=window_size
    )

    results = inference.process_video(
        pose_data['poses'],
        smooth_sigma=smooth_sigma,
        peak_threshold=peak_threshold
    )

    # Annotate video if requested
    if output_video:
        inference.annotate_video(
            pose_data['preprocessed_path'],
            output_video,
            pose_data['poses'],
            results,
            fps=pose_data['meta']['fps']
        )

    # Print detections
    print(f"\nDetected {len(results['detections'])} swings:")
    for det in results['detections']:
        time_sec = det['frame'] / pose_data['meta']['fps']
        print(f"  Frame {det['frame']} ({time_sec:.2f}s): {det['class']} ({det['confidence']:.3f})")

    return {
        'detections': results['detections'],
        'meta': pose_data['meta'],
        'frame_predictions': results['frame_predictions'].tolist(),
        'peaks_idx': results['peaks_idx']
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SGNLite Video Inference")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("checkpoint", help="Model checkpoint path")
    parser.add_argument("-o", "--output", help="Output video path")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--window", type=int, default=20, help="Window size")
    parser.add_argument("--threshold", type=float, default=0.7, help="Peak threshold")

    args = parser.parse_args()

    run_inference_pipeline(
        args.video,
        args.checkpoint,
        output_video=args.output,
        device=args.device,
        window_size=args.window,
        peak_threshold=args.threshold
    )
