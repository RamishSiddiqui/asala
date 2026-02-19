"""
Video Physics-based Verification for AI-generated Video Detection.

Implements temporal and per-frame analysis techniques to detect AI-generated
or manipulated video.  Composes with PhysicsVerifier for per-frame image
analysis and adds video-specific temporal methods.

Methods:
  1. Per-Frame Image Forensics Aggregation   — existing 16 image methods on sampled frames
  2. Temporal Noise Consistency              — cross-correlation of noise residuals across frames
  3. Optical Flow Anomaly Detection          — flow field smoothness and temporal consistency
  4. GOP / Double Encoding Analysis          — compression artifact periodicity
  5. Temporal Lighting Consistency           — lighting direction stability across frames
  6. Frame-to-Frame Stability               — global motion and jitter analysis
"""

import concurrent.futures
import io
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import signal as scipy_signal

from .types import LayerResult

logger = logging.getLogger(__name__)


def _decode_video(video_bytes: bytes, max_frames: int = 300) -> Tuple[List[np.ndarray], float, Optional[bytes]]:
    """Decode video bytes to list of BGR frames, fps, and audio bytes (if available).

    Uses a temporary file because OpenCV's VideoCapture doesn't support
    reading from memory buffers directly.

    Args:
        video_bytes: Raw video file bytes.
        max_frames: Maximum frames to extract (0 = all).

    Returns:
        (frames, fps, audio_bytes_or_None)
    """
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        tmp.write(video_bytes)
        tmp.flush()
        tmp.close()

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise ValueError("Failed to open video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample strategy: if too many frames, sample evenly
        if max_frames > 0 and total_frames > max_frames:
            sample_indices = set(np.linspace(0, total_frames - 1, max_frames, dtype=int))
        else:
            sample_indices = None  # Take all

        frames = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if sample_indices is None or idx in sample_indices:
                frames.append(frame)
            idx += 1
            if max_frames > 0 and len(frames) >= max_frames:
                break

        cap.release()

        # Audio extraction not supported via OpenCV alone;
        # would need ffmpeg.  Return None for now.
        return frames, fps, None

    except Exception as e:
        logger.warning(f"Video decode failed: {e}")
        return [], 30.0, None
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


class VideoVerifier:
    """Physics-based video verification layer.

    Detects AI-generated or manipulated video using temporal analysis
    and per-frame image forensics.  Requires no training data.
    """

    WEIGHTS = {
        'frame_analysis': 0.25,        # Per-frame image forensics (strongest)
        'temporal_noise': 0.18,
        'optical_flow': 0.18,
        'encoding_analysis': 0.10,
        'temporal_lighting': 0.14,
        'frame_stability': 0.15,
    }

    def __init__(self, image_verifier=None, max_workers: int = 1):
        """Initialize with optional PhysicsVerifier for per-frame analysis.

        Args:
            image_verifier: Optional PhysicsVerifier for per-frame image analysis.
            max_workers: Number of threads for parallel analysis.
                1 (default) runs sequentially.  Values > 1 use a
                ThreadPoolExecutor to run the 6 analysis methods
                concurrently.
        """
        self._image_verifier = image_verifier
        self._max_workers = max(1, max_workers)

    def verify_video(self, video_bytes: bytes) -> LayerResult:
        """Verify video for physical consistency.

        Args:
            video_bytes: Raw video file bytes (MP4, AVI, etc.).

        Returns:
            LayerResult with passed status, score, and detailed analysis.
        """
        frames, fps, audio = _decode_video(video_bytes)

        if len(frames) < 2:
            return LayerResult(
                name='physics', passed=False, score=0,
                details={'error': 'Video too short or decode failed (< 2 frames)'}
            )

        results: Dict[str, Any] = {
            'frame_count': len(frames),
            'fps': fps,
        }

        # Resize frames to manageable size for analysis
        target_size = (256, 256)
        resized = [cv2.resize(f, target_size) for f in frames]

        # Convert to grayscale for temporal analyses
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                 for f in resized]

        # Define all analysis tasks: (key, callable, args)
        analysis_tasks = [
            ('frame_analysis', self._analyze_frames, (resized,)),
            ('temporal_noise', self._analyze_temporal_noise, (grays,)),
            ('optical_flow', self._analyze_optical_flow, (grays,)),
            ('encoding_analysis', self._analyze_encoding, (grays,)),
            ('temporal_lighting', self._analyze_temporal_lighting, (resized, grays)),
            ('frame_stability', self._analyze_frame_stability, (grays,)),
        ]

        # Run all analyses (parallel or sequential)
        if self._max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = {
                    key: executor.submit(fn, *args)
                    for key, fn, args in analysis_tasks
                }
                for key, future in futures.items():
                    results[key] = future.result()
        else:
            for key, fn, args in analysis_tasks:
                results[key] = fn(*args)

        # Composite score
        final_score = self._calculate_composite_score(results)

        # Count AI indicators
        ai_indicators = self._count_ai_indicators(results)
        total_indicators = 16
        ai_probability = ai_indicators / total_indicators

        passed, final_score, warning = self._determine_result(
            final_score, ai_probability, results
        )

        if warning:
            results['warning'] = warning
        results['ai_probability'] = ai_probability
        results['ai_indicators'] = ai_indicators

        return LayerResult(
            name='physics',
            passed=passed,
            score=final_score,
            details=results,
        )

    # ------------------------------------------------------------------
    # 1. Per-Frame Image Forensics Aggregation
    # ------------------------------------------------------------------
    def _analyze_frames(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Run image forensics on sampled frames and aggregate.

        If no image_verifier is provided, uses lightweight per-frame
        metrics: noise uniformity, DCT statistics, and texture complexity.
        """
        try:
            # Sample up to 10 frames evenly
            n = len(frames)
            indices = np.linspace(0, n - 1, min(10, n), dtype=int)
            sampled = [frames[i] for i in indices]

            if self._image_verifier is not None:
                # Full per-frame analysis
                def _verify_single_frame(frame):
                    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    return self._image_verifier.verify_image(buf.tobytes()).score

                if self._max_workers > 1:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                        scores = list(executor.map(_verify_single_frame, sampled))
                else:
                    scores = [_verify_single_frame(frame) for frame in sampled]

                avg_score = float(np.mean(scores))
                score_std = float(np.std(scores))

                # Unnaturally consistent scores across frames = AI-generated
                # Natural video: std 3-15; AI: std < 2 or > 20
                if score_std < 2:
                    consistency_penalty = 15
                elif score_std > 20:
                    consistency_penalty = 10
                else:
                    consistency_penalty = 0

                frame_score = int(np.clip(avg_score - consistency_penalty, 0, 100))

                return {
                    'avg_frame_score': avg_score,
                    'frame_score_std': score_std,
                    'n_sampled': len(sampled),
                    'frame_analysis_score': frame_score,
                }
            else:
                # Lightweight: per-frame noise + texture
                noise_cvs = []
                texture_scores = []

                for frame in sampled:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Laplacian variance (noise/detail indicator)
                    lap = cv2.Laplacian(gray, cv2.CV_64F)
                    lap_var = float(np.var(lap))

                    # Texture: gradient magnitude
                    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
                    texture_scores.append(float(np.mean(grad_mag)))

                    # Noise CV across 4x4 grid
                    h, w = gray.shape
                    grid_vars = []
                    for i in range(4):
                        for j in range(4):
                            region = gray[i * h // 4:(i + 1) * h // 4,
                                          j * w // 4:(j + 1) * w // 4]
                            region_lap = cv2.Laplacian(region.astype(np.float64), cv2.CV_64F)
                            grid_vars.append(np.var(region_lap))
                    gv = np.array(grid_vars)
                    cv_val = float(np.std(gv) / (np.mean(gv) + 1e-10))
                    noise_cvs.append(cv_val)

                avg_noise_cv = float(np.mean(noise_cvs))
                avg_texture = float(np.mean(texture_scores))
                texture_std = float(np.std(texture_scores))

                # Score: moderate noise CV + good texture = natural
                noise_score = int(np.clip(avg_noise_cv * 100, 20, 90))
                texture_s = int(np.clip(avg_texture / 30 * 60 + 20, 20, 90))
                frame_score = int(0.5 * noise_score + 0.5 * texture_s)

                return {
                    'avg_noise_cv': avg_noise_cv,
                    'avg_texture': avg_texture,
                    'texture_std': texture_std,
                    'n_sampled': len(sampled),
                    'frame_analysis_score': frame_score,
                }
        except Exception as e:
            logger.warning(f"Frame analysis failed: {e}")
            return {'frame_analysis_score': 50, 'error': str(e)}

    # ------------------------------------------------------------------
    # 2. Temporal Noise Consistency
    # ------------------------------------------------------------------
    def _analyze_temporal_noise(self, grays: List[np.ndarray]) -> Dict[str, Any]:
        """Check temporal correlation of noise residuals between frames.

        Real cameras produce temporally correlated sensor noise.
        AI-generated video has either no temporal noise correlation
        (independent frame synthesis) or unnaturally high correlation
        (copied noise patterns).

        Natural: correlation 0.15-0.6 (depends on ISO/shutter)
        Synthetic: <0.05 or >0.85
        """
        try:
            # Extract noise residuals via median filter subtraction
            residuals = []
            for gray in grays[:60]:  # Limit for performance
                blurred = cv2.medianBlur(
                    (gray * 255).astype(np.uint8), 5
                ).astype(np.float32) / 255.0
                residual = gray - blurred
                residuals.append(residual)

            if len(residuals) < 3:
                return {'temporal_noise_score': 50, 'note': 'Too few frames'}

            # Cross-correlation between consecutive frame residuals
            correlations = []
            for i in range(len(residuals) - 1):
                r1 = residuals[i].flatten()
                r2 = residuals[i + 1].flatten()
                s1, s2 = np.std(r1), np.std(r2)
                if s1 < 1e-10 or s2 < 1e-10:
                    correlations.append(0.0)
                    continue
                corr = np.corrcoef(r1, r2)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                correlations.append(float(corr))

            avg_corr = float(np.mean(correlations))
            corr_std = float(np.std(correlations))

            # Also check non-adjacent frame correlation (should decay)
            if len(residuals) > 5:
                far_corrs = []
                for i in range(0, len(residuals) - 3, 3):
                    r1 = residuals[i].flatten()
                    r2 = residuals[i + 3].flatten()
                    s1, s2 = np.std(r1), np.std(r2)
                    if s1 < 1e-10 or s2 < 1e-10:
                        continue
                    corr = np.corrcoef(r1, r2)[0, 1]
                    if not np.isnan(corr):
                        far_corrs.append(float(corr))
                avg_far_corr = float(np.mean(far_corrs)) if far_corrs else avg_corr
                decay = avg_corr - avg_far_corr
            else:
                avg_far_corr = avg_corr
                decay = 0.0

            # Score: natural range (0.1-0.6 with some decay) = high score
            if avg_corr < 0.02:
                score = 25  # No temporal correlation = independent synthesis
            elif avg_corr < 0.15:
                score = int(25 + (avg_corr - 0.02) / 0.13 * 30)
            elif avg_corr < 0.65:
                score = int(55 + (avg_corr - 0.15) / 0.5 * 35)
            elif avg_corr < 0.85:
                score = int(90 - (avg_corr - 0.65) / 0.2 * 30)
            else:
                score = int(np.clip(60 - (avg_corr - 0.85) * 200, 15, 60))

            # Bonus for natural decay pattern
            if 0.01 < decay < 0.3:
                score = min(100, score + 5)

            return {
                'avg_noise_corr': avg_corr,
                'noise_corr_std': corr_std,
                'noise_corr_decay': decay,
                'temporal_noise_score': score,
            }
        except Exception as e:
            logger.warning(f"Temporal noise analysis failed: {e}")
            return {'temporal_noise_score': 50, 'error': str(e)}

    # ------------------------------------------------------------------
    # 3. Optical Flow Anomaly Detection
    # ------------------------------------------------------------------
    def _analyze_optical_flow(self, grays: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze optical flow for physical plausibility.

        Natural motion produces smooth, consistent flow fields.
        AI-generated video shows: temporal jitter, flow discontinuities,
        and physically impossible motion patterns.
        """
        try:
            if len(grays) < 3:
                return {'optical_flow_score': 50, 'note': 'Too few frames'}

            # Compute Farneback dense optical flow between consecutive frames
            # Sample up to 30 frame pairs
            n = len(grays)
            step = max(1, n // 30)
            flow_magnitudes = []
            flow_angles = []
            flow_smoothness = []

            prev_gray_u8 = (grays[0] * 255).astype(np.uint8)
            for i in range(step, n, step):
                curr_gray_u8 = (grays[i] * 255).astype(np.uint8)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray_u8, curr_gray_u8, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )

                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                avg_mag = float(np.mean(mag))
                flow_magnitudes.append(avg_mag)
                flow_angles.append(float(np.mean(ang)))

                # Flow smoothness: Laplacian of flow field
                # Use CV_32F since flow is float32 (CV_64F not supported for float32 input)
                flow_lap_x = cv2.Laplacian(flow[..., 0], cv2.CV_32F)
                flow_lap_y = cv2.Laplacian(flow[..., 1], cv2.CV_32F)
                smoothness = float(np.mean(np.sqrt(flow_lap_x ** 2 + flow_lap_y ** 2)))
                flow_smoothness.append(smoothness)

                prev_gray_u8 = curr_gray_u8

            if not flow_magnitudes:
                return {'optical_flow_score': 50, 'note': 'No flow computed'}

            flow_magnitudes = np.array(flow_magnitudes)
            flow_smoothness_arr = np.array(flow_smoothness)

            # Temporal consistency: how much does flow magnitude vary
            mag_cv = float(np.std(flow_magnitudes) / (np.mean(flow_magnitudes) + 1e-10))

            # Average flow smoothness (lower = smoother = more natural)
            avg_smoothness = float(np.mean(flow_smoothness_arr))

            # Jitter: frame-to-frame changes in flow magnitude
            if len(flow_magnitudes) > 2:
                mag_diffs = np.abs(np.diff(flow_magnitudes))
                avg_jitter = float(np.mean(mag_diffs))
                max_jitter = float(np.max(mag_diffs))
            else:
                avg_jitter = 0.0
                max_jitter = 0.0

            # Score components
            # Smooth flow = natural
            smoothness_score = int(np.clip(100 - avg_smoothness * 50, 20, 95))

            # Moderate jitter = natural; very low (static) or very high = suspect
            if avg_jitter < 0.01:
                jitter_score = 40  # Too static
            elif avg_jitter < 2.0:
                jitter_score = int(np.clip(40 + avg_jitter / 2.0 * 50, 40, 90))
            else:
                jitter_score = int(np.clip(90 - (avg_jitter - 2.0) * 15, 20, 90))

            optical_flow_score = int(0.6 * smoothness_score + 0.4 * jitter_score)

            return {
                'avg_flow_magnitude': float(np.mean(flow_magnitudes)),
                'flow_mag_cv': mag_cv,
                'avg_flow_smoothness': avg_smoothness,
                'avg_jitter': avg_jitter,
                'max_jitter': max_jitter,
                'optical_flow_score': optical_flow_score,
            }
        except Exception as e:
            logger.warning(f"Optical flow analysis failed: {e}")
            return {'optical_flow_score': 50, 'error': str(e)}

    # ------------------------------------------------------------------
    # 4. GOP / Double Encoding Analysis
    # ------------------------------------------------------------------
    def _analyze_encoding(self, grays: List[np.ndarray]) -> Dict[str, Any]:
        """Detect double encoding artifacts via blockiness periodicity.

        Re-encoded video shows periodic variations in blockiness aligned
        with the original GOP structure.  Also detects unnaturally uniform
        compression (AI-generated frames have no real encoding history).
        """
        try:
            # Measure blockiness per frame
            blockiness_per_frame = []
            for gray in grays[:60]:
                gray_u8 = (gray * 255).astype(np.uint8)
                # Blockiness: average gradient at 8-pixel boundaries
                h, w = gray_u8.shape
                # Horizontal block boundaries
                h_blocks = []
                for x in range(8, w - 1, 8):
                    boundary_diff = np.mean(np.abs(
                        gray_u8[:, x].astype(float) - gray_u8[:, x - 1].astype(float)
                    ))
                    h_blocks.append(boundary_diff)

                # Vertical block boundaries
                v_blocks = []
                for y in range(8, h - 1, 8):
                    boundary_diff = np.mean(np.abs(
                        gray_u8[y, :].astype(float) - gray_u8[y - 1, :].astype(float)
                    ))
                    v_blocks.append(boundary_diff)

                avg_blockiness = float(np.mean(h_blocks + v_blocks)) if (h_blocks or v_blocks) else 0
                blockiness_per_frame.append(avg_blockiness)

            if len(blockiness_per_frame) < 5:
                return {'encoding_score': 50, 'note': 'Too few frames'}

            blockiness = np.array(blockiness_per_frame)
            avg_blockiness = float(np.mean(blockiness))
            blockiness_cv = float(np.std(blockiness) / (np.mean(blockiness) + 1e-10))

            # Check for periodic pattern in blockiness (GOP structure)
            # Autocorrelation of blockiness sequence
            bk_centered = blockiness - np.mean(blockiness)
            autocorr = np.correlate(bk_centered, bk_centered, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)

            # Find periodic peaks (skip lag 0)
            peaks, _ = scipy_signal.find_peaks(autocorr[1:min(30, len(autocorr))], height=0.2)
            n_periodic_peaks = len(peaks)

            # Score: natural video has moderate blockiness with some variation
            # AI-generated: very uniform blockiness (low CV)
            # Double encoded: periodic blockiness pattern
            if blockiness_cv < 0.05:
                cv_score = 30  # Too uniform = synthetic
            elif blockiness_cv < 0.3:
                cv_score = int(30 + (blockiness_cv - 0.05) / 0.25 * 50)
            else:
                cv_score = int(np.clip(80 - (blockiness_cv - 0.3) * 30, 40, 80))

            periodic_penalty = min(25, n_periodic_peaks * 10)
            encoding_score = int(np.clip(cv_score - periodic_penalty, 0, 100))

            return {
                'avg_blockiness': avg_blockiness,
                'blockiness_cv': blockiness_cv,
                'n_periodic_peaks': n_periodic_peaks,
                'encoding_score': encoding_score,
            }
        except Exception as e:
            logger.warning(f"Encoding analysis failed: {e}")
            return {'encoding_score': 50, 'error': str(e)}

    # ------------------------------------------------------------------
    # 5. Temporal Lighting Consistency
    # ------------------------------------------------------------------
    def _analyze_temporal_lighting(self, frames: List[np.ndarray],
                                    grays: List[np.ndarray] = None) -> Dict[str, Any]:
        """Check temporal consistency of lighting direction across frames.

        In real video, lighting changes smoothly.  AI-generated video
        shows: abrupt lighting direction changes, flickering brightness,
        and inconsistent shadow positions.
        """
        try:
            # Track brightness and gradient direction per frame
            brightness_values = []
            gradient_directions = []

            for idx, frame in enumerate(frames[:60]):
                gray = grays[idx] if grays is not None else (
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0)
                brightness_values.append(float(np.mean(gray)))

                # Dominant gradient direction (proxy for lighting)
                gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
                gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
                # Weighted average direction
                mag = np.sqrt(gx ** 2 + gy ** 2)
                weight_sum = np.sum(mag) + 1e-10
                avg_dir = np.arctan2(
                    np.sum(gy * mag) / weight_sum,
                    np.sum(gx * mag) / weight_sum
                )
                gradient_directions.append(float(avg_dir))

            if len(brightness_values) < 3:
                return {'temporal_lighting_score': 50, 'note': 'Too few frames'}

            brightness = np.array(brightness_values)
            directions = np.array(gradient_directions)

            # Brightness stability
            brightness_cv = float(np.std(brightness) / (np.mean(brightness) + 1e-10))

            # Brightness flickering: frame-to-frame changes
            brightness_diffs = np.abs(np.diff(brightness))
            avg_flicker = float(np.mean(brightness_diffs))
            max_flicker = float(np.max(brightness_diffs))

            # Direction stability: frame-to-frame changes
            dir_diffs = np.abs(np.diff(directions))
            # Wrap around pi
            dir_diffs = np.minimum(dir_diffs, 2 * np.pi - dir_diffs)
            avg_dir_change = float(np.mean(dir_diffs))

            # Score: smooth changes = natural; abrupt changes or no changes = suspicious
            # Brightness flicker
            if max_flicker > 30:
                flicker_score = int(np.clip(100 - (max_flicker - 30) * 3, 20, 80))
            else:
                flicker_score = 80

            # Direction consistency
            if avg_dir_change < 0.01:
                dir_score = 50  # Perfectly static lighting (neutral — could be indoor)
            elif avg_dir_change < 0.3:
                dir_score = int(50 + (avg_dir_change - 0.01) / 0.29 * 35)
            else:
                dir_score = int(np.clip(85 - (avg_dir_change - 0.3) * 50, 20, 85))

            temporal_lighting_score = int(0.5 * flicker_score + 0.5 * dir_score)

            return {
                'brightness_cv': brightness_cv,
                'avg_flicker': avg_flicker,
                'max_flicker': max_flicker,
                'avg_dir_change': avg_dir_change,
                'temporal_lighting_score': temporal_lighting_score,
            }
        except Exception as e:
            logger.warning(f"Temporal lighting analysis failed: {e}")
            return {'temporal_lighting_score': 50, 'error': str(e)}

    # ------------------------------------------------------------------
    # 6. Frame-to-Frame Stability
    # ------------------------------------------------------------------
    def _analyze_frame_stability(self, grays: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze global frame-to-frame stability and jitter.

        Real video has natural camera motion or stability.  AI-generated
        video shows micro-jitter (per-frame synthesis errors) or
        unnaturally perfect stability (no camera motion at all).
        """
        try:
            if len(grays) < 3:
                return {'stability_score': 50, 'note': 'Too few frames'}

            # Frame-to-frame SSIM-like metric: normalized cross-correlation
            ncc_values = []
            mse_values = []

            for i in range(len(grays) - 1):
                f1 = grays[i]
                f2 = grays[i + 1]
                # NCC
                m1, m2 = np.mean(f1), np.mean(f2)
                s1, s2 = np.std(f1), np.std(f2)
                if s1 < 1e-10 or s2 < 1e-10:
                    ncc_values.append(1.0)
                    mse_values.append(0.0)
                    continue
                ncc = float(np.mean((f1 - m1) * (f2 - m2)) / (s1 * s2))
                ncc_values.append(ncc)
                mse_values.append(float(np.mean((f1 - f2) ** 2)))

            ncc_arr = np.array(ncc_values)
            mse_arr = np.array(mse_values)

            avg_ncc = float(np.mean(ncc_arr))
            ncc_std = float(np.std(ncc_arr))
            avg_mse = float(np.mean(mse_arr))

            # Detect micro-jitter: high-frequency oscillation in NCC
            if len(ncc_arr) > 5:
                ncc_diffs = np.abs(np.diff(ncc_arr))
                avg_ncc_jitter = float(np.mean(ncc_diffs))
            else:
                avg_ncc_jitter = 0.0

            # Score:
            # Very high NCC (>0.999) = too stable (static image or loop)
            # Very low NCC (<0.5) = extreme instability
            # Natural: 0.85-0.99 with moderate variation
            if avg_ncc > 0.999:
                ncc_score = 40  # Suspiciously static
            elif avg_ncc > 0.95:
                ncc_score = int(40 + (0.999 - avg_ncc) / 0.049 * 50)
            elif avg_ncc > 0.7:
                ncc_score = 80  # Good — natural motion
            else:
                ncc_score = int(np.clip(80 - (0.7 - avg_ncc) * 100, 20, 80))

            # Micro-jitter penalty
            if avg_ncc_jitter > 0.05:
                jitter_penalty = min(20, int((avg_ncc_jitter - 0.05) * 200))
            else:
                jitter_penalty = 0

            stability_score = int(np.clip(ncc_score - jitter_penalty, 0, 100))

            return {
                'avg_ncc': avg_ncc,
                'ncc_std': ncc_std,
                'avg_mse': avg_mse,
                'avg_ncc_jitter': avg_ncc_jitter,
                'stability_score': stability_score,
            }
        except Exception as e:
            logger.warning(f"Frame stability analysis failed: {e}")
            return {'stability_score': 50, 'error': str(e)}

    # ------------------------------------------------------------------
    # Composite scoring
    # ------------------------------------------------------------------
    def _calculate_composite_score(self, results: Dict[str, Any]) -> int:
        score_key_map = {
            'frame_analysis': 'frame_analysis_score',
            'temporal_noise': 'temporal_noise_score',
            'optical_flow': 'optical_flow_score',
            'encoding_analysis': 'encoding_score',
            'temporal_lighting': 'temporal_lighting_score',
            'frame_stability': 'stability_score',
        }

        total_score = 0.0
        total_weight = 0.0

        for method, weight in self.WEIGHTS.items():
            sub = results.get(method, {})
            if isinstance(sub, dict):
                score_key = score_key_map.get(method)
                if score_key and score_key in sub:
                    score = sub[score_key]
                    if isinstance(score, (int, float)):
                        total_score += score * weight
                        total_weight += weight

        if total_weight > 0:
            return int(np.clip(total_score / total_weight, 0, 100))
        return 50

    # ------------------------------------------------------------------
    # AI indicator counting
    # ------------------------------------------------------------------
    def _count_ai_indicators(self, results: Dict[str, Any]) -> int:
        indicators = 0

        # 1. Low per-frame image score
        frame_score = results.get('frame_analysis', {}).get('frame_analysis_score', 50)
        if frame_score < 40:
            indicators += 1

        # 2. Very low per-frame score
        if frame_score < 25:
            indicators += 1

        # 3. No temporal noise correlation (independent synthesis)
        noise_corr = results.get('temporal_noise', {}).get('avg_noise_corr', 0.3)
        if noise_corr < 0.05:
            indicators += 1

        # 3b. Zero noise correlation (strong signal — splice or fully independent)
        if noise_corr < 0.01:
            indicators += 1

        # 4. Too high noise correlation (copied noise)
        if noise_corr > 0.75:
            indicators += 1

        # 5. Flow smoothness anomaly
        flow_smooth = results.get('optical_flow', {}).get('avg_flow_smoothness', 0.5)
        if flow_smooth > 3.0:
            indicators += 1

        # 6. Flow jitter (deepfake micro-jitter)
        max_jitter = results.get('optical_flow', {}).get('max_jitter', 0.5)
        if max_jitter > 5.0:
            indicators += 1

        # 7. Too-uniform blockiness (AI-generated)
        block_cv = results.get('encoding_analysis', {}).get('blockiness_cv', 0.15)
        if block_cv < 0.05:
            indicators += 1

        # 8. Periodic blockiness (double encoding)
        periodic = results.get('encoding_analysis', {}).get('n_periodic_peaks', 0)
        if periodic > 2:
            indicators += 1

        # 9. Brightness flickering
        max_flicker = results.get('temporal_lighting', {}).get('max_flicker', 5.0)
        if max_flicker > 40:
            indicators += 1

        # 10. Lighting direction instability
        dir_change = results.get('temporal_lighting', {}).get('avg_dir_change', 0.05)
        if dir_change > 0.5:
            indicators += 1

        # 11. Suspiciously static video
        avg_ncc = results.get('frame_stability', {}).get('avg_ncc', 0.95)
        if avg_ncc > 0.996:
            indicators += 1

        # 12. Frame-level micro-jitter
        ncc_jitter = results.get('frame_stability', {}).get('avg_ncc_jitter', 0.01)
        if ncc_jitter > 0.08:
            indicators += 1

        # 13. Combined: low frame score + low noise correlation = AI video
        if frame_score < 40 and noise_corr < 0.1:
            indicators += 1

        # 14. Combined: perfect stability + uniform encoding = synthetic
        if avg_ncc > 0.996 and block_cv < 0.05:
            indicators += 1

        # 15. High NCC variance (splice indicator — sudden similarity change)
        ncc_std = results.get('frame_stability', {}).get('ncc_std', 0.01)
        if ncc_std > 0.08:
            indicators += 1

        return indicators

    # ------------------------------------------------------------------
    # Result determination
    # ------------------------------------------------------------------
    def _determine_result(
        self,
        score: int,
        ai_probability: float,
        results: Dict[str, Any]
    ) -> Tuple[bool, int, Optional[str]]:

        # Splice detection: zero temporal correlation + unstable similarity
        noise_corr = results.get('temporal_noise', {}).get('avg_noise_corr', 0.3)
        ncc_std = results.get('frame_stability', {}).get('ncc_std', 0.01)
        if noise_corr < 0.02 and ncc_std > 0.05:
            warning = 'Video splice detected — temporal discontinuity'
            adjusted = int(score * 0.6)
            return False, adjusted, warning

        if ai_probability >= 0.4:
            warning = 'Strong indicators of AI-generated video'
            adjusted = int(score * 0.5)
            return False, adjusted, warning

        elif ai_probability >= 0.3:
            warning = 'Some indicators of AI-generated or manipulated video'
            adjusted = int(score * 0.7)
            return adjusted >= 50, adjusted, warning

        elif ai_probability >= 0.15:
            warning = 'Minor video inconsistencies detected'
            passed = score >= 50
            return passed, score, warning if not passed else None

        else:
            passed = score >= 40
            return passed, score, None
