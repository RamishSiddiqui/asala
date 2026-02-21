"""
Enhanced Physics-based Verification for AI-generated Image Detection.

This module implements multiple mathematical analysis techniques to detect
AI-generated or manipulated images based on physical inconsistencies.
"""
import concurrent.futures
import io
import logging
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np
import pywt
from PIL import Image, ImageChops

from .types import LayerResult

# Configure logging
logger = logging.getLogger(__name__)


class PhysicsVerifier:
    """
    Enhanced physics-based verification layer for detecting AI-generated images.
    
    ⚠️ EXPERIMENTAL: This is an experimental feature that uses mathematical and 
    physics-based analysis to detect AI-generated images without machine learning.
    
    Current accuracy may vary based on image characteristics and quality.
    Optimized for consumer devices with CPU-only processing.
    4. Lighting Analysis - Shadow and light direction consistency
    5. Texture Analysis - Gradient-based texture metrics
    6. Color Distribution - HSV/LAB color space analysis
    7. Compression Artifacts - ELA (Error Level Analysis)

    Each analysis returns a score from 0-100 where:
    - Higher scores indicate more "natural" characteristics
    - Lower scores indicate potential AI generation
    """

    def __init__(self, max_workers: int = 1):
        """Initialize PhysicsVerifier.

        Args:
            max_workers: Number of threads for parallel analysis.
                1 (default) runs sequentially.  Values > 1 use a
                ThreadPoolExecutor to run the 15 analysis methods
                concurrently.
        """
        self._max_workers = max(1, max_workers)

    # AI detection thresholds (recalibrated from benchmark data)
    THRESHOLDS = {
        # Noise: combined metric instead of standalone variance_mean
        'noise_cv_low': 0.25,         # CV below this + variance > 30 → suspicious
        'noise_variance_floor': 30,   # Minimum variance for noise suspicion
        'dct_low': 35,                # Lower DCT score = more AI-like
        'lighting_low': 35,           # Lower = inconsistent (AI-like, raised for StyleGAN detection)
        'texture_low': 49,            # Lower = synthetic texture (raised after gradient-based fix)
        'geometric_low': 31,          # Lower = inconsistent geometry
        'color_low': 20,              # Lower = unnatural colors
        'compression_low': 40,        # Lower = unusual compression
        'regional_ela_cv_high': 0.5,  # High regional ELA CV = likely splice
        'regional_ela_cv_low': 0.25,  # Low regional ELA CV = AI-generated (uniform compression)
    }

    def verify_image(self, image_bytes: bytes) -> LayerResult:
        """Verify image for physical consistency using multiple mathematical approaches.
        
        Args:
            image_bytes: Raw image bytes to analyze
            
        Returns:
            LayerResult with passed status, score, and detailed analysis results
        """
        try:
            # Convert bytes to numpy array for OpenCV
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                logger.error("Failed to decode image data")
                return LayerResult(
                    name="Physics Verification (Image)",
                    passed=False,
                    score=0,
                    details={"error": "Invalid image data"}
                )

            # Pre-compute shared intermediate arrays once
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

            # Define all analysis tasks: (key, callable, args)
            analysis_tasks = [
                ('noise_uniformity', self._analyze_noise_uniformity, (img, gray)),
                ('noise_frequency', self._analyze_noise_frequency, (img, gray)),
                ('frequency_analysis', self._analyze_frequency_domain, (img, gray)),
                ('geometric_consistency', self._analyze_geometric_consistency, (img, gray)),
                ('lighting_analysis', self._analyze_lighting_consistency, (img, lab)),
                ('texture_analysis', self._analyze_texture_patterns, (img, gray)),
                ('color_analysis', self._analyze_color_distribution, (img, lab)),
                ('compression_analysis', self._analyze_compression_artifacts, (image_bytes,)),
                ('noise_consistency_map', self._analyze_noise_consistency_map, (img, gray)),
                ('jpeg_ghost', self._analyze_jpeg_ghosts, (image_bytes,)),
                ('spectral_fingerprint', self._analyze_spectral_fingerprint, (img, gray)),
                ('channel_correlation', self._analyze_channel_correlation, (img,)),
                ('benford_dct', self._analyze_benford_dct, (img, gray)),
                ('wavelet_ratio', self._analyze_wavelet_spectral_ratio, (img, gray)),
                ('blocking_grid', self._analyze_blocking_artifact_grid, (img, gray)),
                ('cfa_demosaicing', self._analyze_cfa_demosaicing, (img,)),
            ]

            # Run all analysis modules (parallel or sequential)
            results: Dict[str, Any] = {}
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

            # Calculate weighted composite score
            final_score = self._calculate_composite_score(results)

            # Count AI indicators based on calibrated thresholds
            ai_indicators = self._count_ai_indicators(results)
            total_indicators = 20  # Phase 1: 10, Phase 2: 6, Phase 3: 4 (benford, wavelet, bag, cfa)

            # Calculate AI probability
            ai_probability = ai_indicators / total_indicators
            
            # Determine pass/fail based on score and AI indicators
            passed, final_score, warning = self._determine_result(
                final_score, ai_probability, results
            )
            
            if warning:
                results['warning'] = warning
            results['ai_probability'] = ai_probability
            results['ai_indicators'] = ai_indicators

            return LayerResult(
                name="Physics Verification (Image)",
                passed=passed,
                score=final_score,
                details=results
            )

        except Exception as e:
            logger.exception("Physics verification failed with exception")
            return LayerResult(
                name="Physics Verification (Image)",
                passed=False,
                score=0,
                details={"error": str(e)}
            )

    # Declarative indicator table for simple threshold checks.
    # Each entry: (result_key, metric_key, op, threshold, default, weight)
    INDICATOR_TABLE = [
        # Phase 1: simple threshold indicators
        ('noise_uniformity', 'residual_cv', '<', 0.10, 0.5, 1),
        ('noise_uniformity', 'uniformity_score', '<', 15, 50, 1),
        ('frequency_analysis', 'dct_score', '<', None, 50, 1),  # None → use THRESHOLDS['dct_low']
        ('lighting_analysis', 'lighting_score', '<', None, 50, 1),  # → THRESHOLDS['lighting_low']
        ('texture_analysis', 'texture_score', '<', None, 50, 1),  # → THRESHOLDS['texture_low']
        ('texture_analysis', 'texture_score', '<', 25, 50, 1),
        ('geometric_consistency', 'geometric_score', '<', None, 50, 1),  # → THRESHOLDS['geometric_low']
        ('color_analysis', 'color_score', '<', None, 50, 1),  # → THRESHOLDS['color_low']
        ('compression_analysis', 'compression_score', '<', None, 50, 1),  # → THRESHOLDS['compression_low']
        # Phase 2
        ('noise_consistency_map', 'noise_map_cv', '>', 0.6, 0.3, 1),
        ('jpeg_ghost', 'ghost_quality_spread', '>', 3, 0, 2),
        ('spectral_fingerprint', 'spectral_residual_energy', '>', 0.40, 0, 1),
        # Phase 3
        ('benford_dct', 'benford_kl_divergence', '>', 0.8, 0, 1),
        ('wavelet_ratio', 'cross_scale_decay', '<', 0.10, 0.5, 1),
        ('cfa_demosaicing', 'cfa_avg_snr', '<', 1.5, 2.5, 1),
    ]

    # Map from None threshold to THRESHOLDS key
    _THRESHOLD_KEY_MAP = {
        ('frequency_analysis', 'dct_score'): 'dct_low',
        ('lighting_analysis', 'lighting_score'): 'lighting_low',
        ('texture_analysis', 'texture_score'): 'texture_low',
        ('geometric_consistency', 'geometric_score'): 'geometric_low',
        ('color_analysis', 'color_score'): 'color_low',
        ('compression_analysis', 'compression_score'): 'compression_low',
    }

    def _count_ai_indicators(self, results: Dict[str, Any]) -> int:
        """Count the number of AI indicators based on recalibrated thresholds."""
        indicators = 0

        # Simple threshold checks from declarative table
        for result_key, metric_key, op, threshold, default, weight in self.INDICATOR_TABLE:
            val = results.get(result_key, {}).get(metric_key, default)
            thresh = threshold
            if thresh is None:
                thresh_key = self._THRESHOLD_KEY_MAP.get((result_key, metric_key))
                thresh = self.THRESHOLDS.get(thresh_key, 0) if thresh_key else 0
            if op == '<' and val < thresh:
                indicators += weight
            elif op == '>' and val > thresh:
                indicators += weight

        # Compound rule: noise CV + variance floor
        noise = results.get('noise_uniformity', {})
        cv_ratio = noise.get('cv_ratio', 0.5)
        variance_mean = noise.get('variance_mean', 50)
        if cv_ratio < self.THRESHOLDS['noise_cv_low'] and variance_mean > self.THRESHOLDS['noise_variance_floor']:
            indicators += 1

        # Two-branch rule: regional ELA CV
        regional_cv = results.get('compression_analysis', {}).get('regional_ela_cv', 0.35)
        if regional_cv < self.THRESHOLDS['regional_ela_cv_low']:
            indicators += 3
        elif regional_cv > self.THRESHOLDS['regional_ela_cv_high']:
            indicators += 1

        # Derived metric: cross-channel correlation
        channel_corrs = results.get('channel_correlation', {})
        avg_corr = (
            abs(channel_corrs.get('channel_noise_corr_rg', 0.5))
            + abs(channel_corrs.get('channel_noise_corr_rb', 0.5))
            + abs(channel_corrs.get('channel_noise_corr_gb', 0.5))
        ) / 3
        if avg_corr < 0.25:
            indicators += 2

        # Boolean indicator: BAG dual grid
        dual_grid = results.get('blocking_grid', {}).get('dual_grid_detected', False)
        if dual_grid:
            indicators += 1

        # Combined GAN rule: low ELA CV (uniform compression) + low texture
        # is a signature specific to production GANs. No real camera image
        # has ela_cv < 0.25, so this never fires for authentic content.
        texture_score = results.get('texture_analysis', {}).get('texture_score', 50)
        if regional_cv < self.THRESHOLDS['regional_ela_cv_low'] and texture_score < self.THRESHOLDS['texture_low']:
            indicators += 1

        return indicators

    def _determine_result(
        self,
        score: int,
        ai_probability: float,
        results: Dict[str, Any]
    ) -> Tuple[bool, int, Optional[str]]:
        """Determine verification result based on score and AI probability.

        In addition to the general probability brackets, a targeted
        manipulation rule fires when **both** high regional ELA CV (>0.5,
        meaning compression artifacts vary across the image) and non-zero
        JPEG ghost spread (>=3, meaning regions converge at different
        quality levels) are present.  This combination is specific to
        copy-paste / splice manipulation and does not occur in
        unmanipulated single-source images.
        """

        # --- Targeted manipulation detection ---
        regional_cv = results.get('compression_analysis', {}).get('regional_ela_cv', 0.35)
        ghost_spread = results.get('jpeg_ghost', {}).get('ghost_quality_spread', 0)

        if regional_cv > 0.5 and ghost_spread >= 3:
            warning = 'Compression inconsistencies detected - likely manipulated (splice/edit)'
            adjusted_score = int(score * 0.7)
            passed = adjusted_score >= 50
            return passed, adjusted_score, warning

        # --- General probability-based brackets ---
        if ai_probability >= 0.4:
            warning = 'Strong physical inconsistencies detected - likely AI-generated'
            adjusted_score = int(score * 0.5)
            return False, adjusted_score, warning

        elif ai_probability >= 0.3:
            warning = 'Some physical inconsistencies detected - possibly AI-generated'
            adjusted_score = int(score * 0.7)
            passed = adjusted_score >= 50
            return passed, adjusted_score, warning

        elif ai_probability >= 0.1:
            warning = 'Minor physical inconsistencies detected - review recommended'
            passed = score >= 45
            return passed, score, warning if not passed else None

        else:  # No AI indicators
            passed = score >= 40
            return passed, score, None

    def _analyze_noise_uniformity(self, img: np.ndarray, gray: np.ndarray) -> Dict[str, Any]:
        """Analyze noise uniformity using both Laplacian CV and noise-residual CV.

        Two complementary metrics:
        1. **Laplacian CV** — coefficient of variation of Laplacian variance
           across a 4×4 grid.  Captures whether edge/detail energy varies
           spatially (natural) or is uniform (synthetic smooth images).
        2. **Noise-residual CV** — median-filter the grayscale image to remove
           structure, then measure how the residual noise standard deviation
           varies across regions.  This isolates sensor noise from scene
           content.  Real cameras have spatially varying noise; AI generators
           produce perfectly uniform noise.

        The final score blends both, with residual CV weighted higher because
        it is more robust to structural content (e.g. GAN faces with strong
        edges but uniform noise).
        """
        try:
            h, w = gray.shape

            # --- Laplacian-based CV ratio ---
            variances = self._compute_grid_metric(
                gray, 4, 4,
                lambda r: cv2.Laplacian(r, cv2.CV_64F).var()
            )
            variance_mean = np.mean(variances)
            variance_std = np.std(variances)
            cv_ratio = variance_std / (variance_mean + 1e-10)

            # Map cv_ratio [0.1, 0.8] → [20, 100]
            cv_score = float(np.clip(
                (cv_ratio - 0.1) / (0.8 - 0.1) * 80 + 20, 0, 100
            ))

            # --- Noise-residual CV ---
            median = cv2.medianBlur(gray, 5)
            residual = gray.astype(np.float64) - median.astype(np.float64)

            residual_stds = self._compute_grid_metric(
                residual, 4, 4, lambda r: np.std(r)
            )
            residual_mean_std = np.mean(residual_stds)
            residual_cv = float(np.std(residual_stds) / (residual_mean_std + 1e-10))

            # Map residual_cv [0.05, 0.5] → [20, 100]
            residual_score = float(np.clip(
                (residual_cv - 0.05) / (0.5 - 0.05) * 80 + 20, 0, 100
            ))

            # --- Variance penalty ---
            if variance_mean < 20:
                variance_penalty = (20 - variance_mean) / 20 * 30
            elif variance_mean > 5000:
                variance_penalty = min(30, (variance_mean - 5000) / 5000 * 30)
            else:
                variance_penalty = 0

            # Blend: residual CV is more robust to structural variation
            uniformity_score = int(np.clip(
                cv_score * 0.35 + residual_score * 0.65 - variance_penalty,
                0, 100
            ))

            return {
                'variance_mean': float(variance_mean),
                'variance_std': float(variance_std),
                'cv_ratio': float(cv_ratio),
                'residual_cv': residual_cv,
                'uniformity_score': uniformity_score,
                'cv_score': cv_score,
                'residual_score': residual_score,
                'variance_penalty': float(variance_penalty),
            }

        except Exception as e:
            logger.warning(f"Noise uniformity analysis failed: {e}")
            return {'uniformity_score': 50, 'error': str(e)}

    def _analyze_noise_frequency(self, img: np.ndarray, gray: np.ndarray) -> Dict[str, Any]:
        """Analyze noise frequency characteristics using FFT.

        Real images have characteristic frequency distributions based on:
        - Lens characteristics
        - Sensor frequency response
        - Natural scene statistics
        """
        try:
            
            # Apply FFT
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            
            # Normalize magnitude
            magnitude_log = np.log2(magnitude + 1)
            
            h, w = magnitude.shape
            center_h, center_w = h // 2, w // 2
            
            # Analyze frequency distribution in concentric rings
            # This is more robust than corner-based analysis
            max_radius = min(h, w) // 2
            low_freq_radius = max_radius // 4
            high_freq_radius = max_radius // 2
            
            # Create masks for different frequency bands
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((y - center_h)**2 + (x - center_w)**2)
            
            # Low frequency (center)
            low_freq_mask = dist_from_center <= low_freq_radius
            low_freq_energy = np.sum(magnitude_log[low_freq_mask])
            
            # Mid frequency
            mid_freq_mask = (dist_from_center > low_freq_radius) & (dist_from_center <= high_freq_radius)
            mid_freq_energy = np.sum(magnitude_log[mid_freq_mask])
            
            # High frequency (outer)
            high_freq_mask = dist_from_center > high_freq_radius
            high_freq_energy = np.sum(magnitude_log[high_freq_mask])
            
            total_energy = low_freq_energy + mid_freq_energy + high_freq_energy + 1e-10
            
            # Calculate ratios
            low_freq_ratio = low_freq_energy / total_energy
            mid_freq_ratio = mid_freq_energy / total_energy
            high_freq_ratio = high_freq_energy / total_energy
            
            # Natural images typically have:
            # - High low-frequency content (60-80%)
            # - Moderate mid-frequency (15-30%)
            # - Low but non-zero high-frequency (5-15%)
            
            # Score based on how close to natural distribution
            low_score = 100 - abs(low_freq_ratio * 100 - 70) * 2  # Optimal around 70%
            mid_score = 100 - abs(mid_freq_ratio * 100 - 20) * 3  # Optimal around 20%
            high_score = min(100, high_freq_ratio * 500)  # Higher is better, but rare
            
            frequency_score = int(np.clip((low_score * 0.4 + mid_score * 0.3 + high_score * 0.3), 0, 100))
            
            return {
                'low_freq_ratio': float(low_freq_ratio),
                'mid_freq_ratio': float(mid_freq_ratio),
                'high_freq_ratio': float(high_freq_ratio),
                'frequency_score': frequency_score
            }
            
        except Exception as e:
            logger.warning(f"Noise frequency analysis failed: {e}")
            return {'frequency_score': 50, 'error': str(e)}

    def _analyze_frequency_domain(self, img: np.ndarray, gray: np.ndarray) -> Dict[str, Any]:
        """Analyze DCT coefficients - AI images have characteristic DCT patterns.

        DCT analysis reveals:
        - Compression artifacts
        - Frequency distribution anomalies
        - AI-specific spectral signatures
        """
        try:
            
            # Resize to standard size for consistent analysis
            standard_size = 256
            gray_resized = cv2.resize(gray, (standard_size, standard_size))
            
            # Apply DCT
            dct = cv2.dct(gray_resized.astype(np.float32))
            
            # Analyze coefficient distribution
            dct_abs = np.abs(dct)
            
            h, w = dct_abs.shape
            
            # Calculate energy in different frequency bands
            # DC component (top-left)
            dc_energy = dct_abs[0, 0]
            
            # Low frequency (top-left 8x8)
            low_freq_energy = np.sum(dct_abs[:8, :8]) - dc_energy
            
            # Mid frequency (8-32 range)
            mid_freq_energy = np.sum(dct_abs[8:32, 8:32])
            
            # High frequency (rest)
            high_freq_energy = np.sum(dct_abs) - dc_energy - low_freq_energy - mid_freq_energy
            
            total_energy = np.sum(dct_abs) + 1e-10
            
            # Calculate proper Shannon entropy on normalized DCT
            dct_normalized = dct_abs / total_energy
            dct_entropy = -np.sum(dct_normalized * np.log2(dct_normalized + 1e-10))
            
            # High frequency ratio
            high_freq_ratio = high_freq_energy / total_energy
            
            # AC energy ratio (all non-DC)
            ac_energy_ratio = (total_energy - dc_energy) / total_energy
            
            # Score calculation
            # Natural images have:
            # - High DC component
            # - Moderate high-frequency content
            # - Entropy typically 4-8 bits
            
            # High frequency score (natural images have some high-freq content)
            hf_score = min(100, high_freq_ratio * 300)
            
            # Entropy score (natural entropy is moderate)
            entropy_score = 100 - abs(dct_entropy - 6) * 10  # Optimal around 6 bits
            
            # AC energy score
            ac_score = min(100, ac_energy_ratio * 120)
            
            dct_score = int(np.clip((hf_score * 0.4 + entropy_score * 0.3 + ac_score * 0.3), 0, 100))
            
            return {
                'dc_energy': float(dc_energy),
                'high_freq_ratio': float(high_freq_ratio),
                'ac_energy_ratio': float(ac_energy_ratio),
                'dct_entropy': float(dct_entropy),
                'dct_score': dct_score
            }
            
        except Exception as e:
            logger.warning(f"Frequency domain analysis failed: {e}")
            return {'dct_score': 50, 'error': str(e)}

    def _analyze_geometric_consistency(self, img: np.ndarray, gray: np.ndarray) -> Dict[str, Any]:
        """Analyze geometric consistency - lines, perspectives, shapes.

        AI-generated images often have:
        - Inconsistent line angles
        - Warped perspectives
        - Irregular shapes
        """
        try:
            
            # Detect edges with multiple thresholds for robustness
            edges1 = cv2.Canny(gray, 50, 150)
            edges2 = cv2.Canny(gray, 100, 200)
            edges = cv2.bitwise_or(edges1, edges2)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(
                edges, 1, np.pi/180, 
                threshold=50, 
                minLineLength=20, 
                maxLineGap=10
            )
            
            # Analyze line consistency
            if lines is not None and len(lines) > 5:
                # Calculate angle distribution
                angles = []
                lengths = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    angles.append(angle)
                    lengths.append(length)
                
                angles = np.array(angles)
                lengths = np.array(lengths)
                
                # Angle consistency (natural images have clustered angles)
                angle_std = np.std(angles)
                
                # Check for dominant angles (should have some structure)
                angle_hist, _ = np.histogram(angles, bins=36, range=(-180, 180))
                max_angle_count = np.max(angle_hist)
                angle_dominance = max_angle_count / len(angles)
                
                # Length distribution
                length_mean = np.mean(lengths)
                length_std = np.std(lengths)
                
            else:
                angle_std = 50
                angle_dominance = 0.1
                length_mean = 0
                length_std = 0
            
            # Detect corners
            corners = cv2.goodFeaturesToTrack(
                gray, 
                maxCorners=100, 
                qualityLevel=0.01, 
                minDistance=10
            )
            
            if corners is not None and len(corners) > 10:
                corner_positions = np.array(corners).reshape(-1, 2)
                corner_density = self._calculate_point_density(corner_positions, gray.shape)
                corner_count = len(corners)
            else:
                corner_density = 0.5
                corner_count = 0
            
            # Calculate geometric consistency score
            # Lower angle_std = more consistent = better
            angle_score = max(0, 100 - angle_std * 1.5)
            
            # Higher dominance = more structure = better
            dominance_score = angle_dominance * 100
            
            # Corner density score
            corner_score = corner_density
            
            # Line count score (some lines are good, too many is noise)
            line_count = len(lines) if lines is not None else 0
            line_count_score = min(100, line_count * 2) if line_count < 50 else max(0, 100 - (line_count - 50))
            
            geo_score = int(np.clip(
                (angle_score * 0.3 + dominance_score * 0.2 + corner_score * 0.2 + line_count_score * 0.3),
                0, 100
            ))
            
            return {
                'line_count': line_count,
                'angle_std': float(angle_std),
                'angle_dominance': float(angle_dominance),
                'corner_count': corner_count,
                'corner_density': float(corner_density),
                'geometric_score': geo_score
            }
            
        except Exception as e:
            logger.warning(f"Geometric consistency analysis failed: {e}")
            return {'geometric_score': 50, 'error': str(e)}

    def _analyze_lighting_consistency(self, img: np.ndarray, lab: np.ndarray) -> Dict[str, Any]:
        """Enhanced lighting and shadow analysis.

        Real images have:
        - Consistent light direction
        - Natural shadow gradients
        - Realistic highlight distribution
        """
        try:
            
            h, w = img.shape[:2]
            
            # Analyze lighting using L channel (lightness)
            l_channel = lab[:, :, 0].astype(np.float32)
            
            # Split into regions and analyze lighting variation
            regions = self._divide_image_into_grid(img, 4, 4)
            
            brightness_values = []
            contrast_values = []
            
            for region in regions:
                gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32)
                brightness = np.mean(gray_region)
                contrast = np.std(gray_region)
                brightness_values.append(brightness)
                contrast_values.append(contrast)
            
            brightness_values = np.array(brightness_values)
            contrast_values = np.array(contrast_values)
            
            # Calculate statistics
            brightness_std = np.std(brightness_values)
            brightness_range = np.max(brightness_values) - np.min(brightness_values)
            contrast_std = np.std(contrast_values)
            
            # Analyze shadow and highlight distribution
            shadow_threshold = np.percentile(l_channel, 15)
            highlight_threshold = np.percentile(l_channel, 85)
            
            shadow_mask = l_channel <= shadow_threshold
            highlight_mask = l_channel >= highlight_threshold
            
            shadow_coverage = np.sum(shadow_mask) / (h * w)
            highlight_coverage = np.sum(highlight_mask) / (h * w)
            
            # Analyze gradient direction (light direction consistency)
            # Use Sobel to find gradient
            grad_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
            
            # Gradient magnitude and direction
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            grad_dir = np.arctan2(grad_y, grad_x)
            
            # Filter for significant gradients
            significant_grads = grad_mag > np.percentile(grad_mag, 75)
            
            if np.sum(significant_grads) > 100:
                # Calculate direction consistency
                significant_dirs = grad_dir[significant_grads]
                dir_std = np.std(significant_dirs)
                # Convert to circular std (0-1, lower is more consistent)
                dir_consistency = 1 - min(1, dir_std / np.pi)
            else:
                dir_consistency = 0.5
            
            # Calculate lighting score
            # Natural images have:
            # - Moderate brightness variation (not too flat, not too extreme)
            # - Consistent gradient directions
            # - Reasonable shadow/highlight balance
            
            brightness_score = 100 - min(100, brightness_std * 3)
            contrast_score = 100 - min(100, contrast_std * 5)
            direction_score = dir_consistency * 100
            
            # Shadow/highlight balance score
            balance_score = 100 - abs(shadow_coverage - highlight_coverage) * 200
            balance_score = max(0, balance_score)
            
            lighting_score = int(np.clip(
                (brightness_score * 0.25 + contrast_score * 0.25 + 
                 direction_score * 0.3 + balance_score * 0.2),
                0, 100
            ))
            
            return {
                'brightness_std': float(brightness_std),
                'brightness_range': float(brightness_range),
                'contrast_std': float(contrast_std),
                'shadow_coverage': float(shadow_coverage),
                'highlight_coverage': float(highlight_coverage),
                'gradient_consistency': float(dir_consistency),
                'lighting_score': lighting_score
            }
            
        except Exception as e:
            logger.warning(f"Lighting consistency analysis failed: {e}")
            return {'lighting_score': 50, 'error': str(e)}

    def _analyze_texture_patterns(self, img: np.ndarray, gray: np.ndarray) -> Dict[str, Any]:
        """Gradient-based texture analysis.

        Uses gradient magnitude statistics instead of LBP, which is
        confounded by JPEG compression artifacts on smooth regions.

        Three metrics:
        1. **Gradient CV** (std/mean of gradient magnitude) — natural images
           have high CV because they mix smooth and detailed areas.  Random
           noise has CV ≈ 0.52 (Rayleigh).  Smooth synthetics have low CV.
        2. **Regional texture variation** — CV of per-region mean gradient
           energy across a 4×4 grid.  Natural scenes vary; synthetics are
           more uniform.
        3. **Gradient energy** — mean gradient magnitude.  Natural photos
           have moderate-to-high energy from fine detail.
        """
        try:
            h, w = gray.shape

            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)

            texture_energy = float(np.mean(grad_mag))
            texture_contrast = float(np.std(grad_mag))
            grad_cv = texture_contrast / (texture_energy + 1e-10)

            # Regional texture variation (4×4 grid)
            region_energies = self._compute_grid_metric(
                grad_mag, 4, 4, lambda r: np.mean(r)
            )
            region_cv = float(
                np.std(region_energies) / (np.mean(region_energies) + 1e-10)
            )

            # --- Scoring ---
            # Gradient CV: natural ~1.5-3, random noise ~0.5, smooth AI ~0.5-1
            cv_texture_score = float(np.clip(
                (grad_cv - 0.8) / (2.5 - 0.8) * 100, 0, 100
            ))

            # Regional variation: natural ~0.3-1.0, synthetic ~0.05-0.3
            region_score = float(np.clip(
                (region_cv - 0.1) / (0.7 - 0.1) * 100, 0, 100
            ))

            # Energy: map [5, 50] → [0, 100]
            energy_score = float(np.clip(
                (texture_energy - 5) / (50 - 5) * 100, 0, 100
            ))

            # Weights: CV 0.35, regional 0.35, energy 0.30
            texture_score = int(np.clip(
                cv_texture_score * 0.35 + region_score * 0.35 + energy_score * 0.30,
                0, 100
            ))

            return {
                'texture_energy': texture_energy,
                'texture_contrast': texture_contrast,
                'grad_cv': float(grad_cv),
                'region_cv': region_cv,
                'texture_score': texture_score,
            }

        except Exception as e:
            logger.warning(f"Texture pattern analysis failed: {e}")
            return {'texture_score': 50, 'error': str(e)}

    def _analyze_color_distribution(self, img: np.ndarray, lab: np.ndarray) -> Dict[str, Any]:
        """Analyze color distribution and consistency.

        AI-generated images often have:
        - Unnatural color distributions
        - Missing or excessive saturation
        - Incorrect color relationships
        """
        try:
            # Convert to HSV (LAB is pre-computed and passed in)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Analyze HSV distributions
            h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
            
            # Normalize histograms
            h_hist = h_hist / (np.sum(h_hist) + 1e-10)
            s_hist = s_hist / (np.sum(s_hist) + 1e-10)
            v_hist = v_hist / (np.sum(v_hist) + 1e-10)
            
            # Calculate color entropy
            h_entropy = -np.sum(h_hist * np.log2(h_hist + 1e-10))
            s_entropy = -np.sum(s_hist * np.log2(s_hist + 1e-10))
            v_entropy = -np.sum(v_hist * np.log2(v_hist + 1e-10))
            
            # Analyze color consistency across regions
            regions = self._divide_image_into_grid(img, 4, 4)
            
            color_variations = []
            for region in regions:
                region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                mean_hue = np.mean(region_hsv[:, :, 0])
                mean_sat = np.mean(region_hsv[:, :, 1])
                mean_val = np.mean(region_hsv[:, :, 2])
                color_variations.append([mean_hue, mean_sat, mean_val])
            
            color_variations = np.array(color_variations)
            hue_std = np.std(color_variations[:, 0])
            sat_std = np.std(color_variations[:, 1])
            val_std = np.std(color_variations[:, 2])
            
            # Analyze LAB color distribution
            a_channel = lab[:, :, 1].astype(np.float32) - 128  # Green-Red
            b_channel = lab[:, :, 2].astype(np.float32) - 128  # Blue-Yellow
            
            a_mean = np.mean(np.abs(a_channel))
            b_mean = np.mean(np.abs(b_channel))
            
            # Natural images typically have:
            # - Moderate hue entropy (3-6 bits)
            # - High saturation entropy (5-8 bits)
            # - High value entropy (5-8 bits)
            # - Some color variation across regions
            
            # Entropy scores
            h_entropy_score = 100 - abs(h_entropy - 4.5) * 15  # Optimal around 4.5
            s_entropy_score = min(100, s_entropy * 12)
            v_entropy_score = min(100, v_entropy * 12)
            
            # Variation scores (some variation is natural)
            hue_var_score = 100 - min(100, hue_std * 2)
            sat_var_score = 100 - min(100, sat_std * 1.5)
            
            # Color richness (LAB)
            color_richness = min(100, (a_mean + b_mean) / 2)
            
            color_score = int(np.clip(
                (h_entropy_score * 0.15 + s_entropy_score * 0.15 + v_entropy_score * 0.15 +
                 hue_var_score * 0.15 + sat_var_score * 0.2 + color_richness * 0.2),
                0, 100
            ))
            
            return {
                'hue_entropy': float(h_entropy),
                'saturation_entropy': float(s_entropy),
                'value_entropy': float(v_entropy),
                'hue_std': float(hue_std),
                'saturation_std': float(sat_std),
                'value_std': float(val_std),
                'color_richness': float(color_richness),
                'color_score': color_score
            }
            
        except Exception as e:
            logger.warning(f"Color distribution analysis failed: {e}")
            return {'color_score': 50, 'error': str(e)}

    def _analyze_compression_artifacts(self, image_bytes: bytes) -> Dict[str, Any]:
        """Analyze compression artifacts and ELA patterns with regional awareness.

        In addition to global ELA statistics, this divides the ELA difference
        image into an NxN grid and computes per-region means.  A high
        coefficient of variation across regions indicates different compression
        levels in different areas — a hallmark of splicing / manipulation.
        """
        try:
            original = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            # --- Global multi-quality ELA ---
            qualities = [95, 85, 75, 65]
            ela_scores = []

            for quality in qualities:
                buffer = io.BytesIO()
                original.save(buffer, 'JPEG', quality=quality)
                buffer.seek(0)
                recompressed = Image.open(buffer)
                diff = ImageChops.difference(original, recompressed)
                diff_array = np.array(diff)
                ela_scores.append(np.mean(diff_array))

            ela_scores = np.array(ela_scores)
            ela_variance = np.var(ela_scores)
            ela_mean = np.mean(ela_scores)
            ela_range = np.max(ela_scores) - np.min(ela_scores)
            ela_gradient = (ela_scores[-1] - ela_scores[0]) / (qualities[-1] - qualities[0])

            # --- Regional ELA at quality=90 ---
            buffer90 = io.BytesIO()
            original.save(buffer90, 'JPEG', quality=90)
            buffer90.seek(0)
            recomp90 = Image.open(buffer90)
            diff90 = np.array(ImageChops.difference(original, recomp90)).astype(np.float32)
            # Collapse to single-channel mean across RGB
            diff90_gray = np.mean(diff90, axis=2)

            grid_n = 8  # 8x8 grid = 64 regions
            rh, rw = diff90_gray.shape
            region_means = []
            for i in range(grid_n):
                for j in range(grid_n):
                    region = diff90_gray[
                        i * rh // grid_n : (i + 1) * rh // grid_n,
                        j * rw // grid_n : (j + 1) * rw // grid_n,
                    ]
                    region_means.append(np.mean(region))

            region_means = np.array(region_means)
            global_ela_mean = np.mean(region_means)
            regional_ela_cv = float(
                np.std(region_means) / (global_ela_mean + 1e-10)
            )
            # Count outlier regions (both high AND low — spliced patches that
            # were already heavily compressed show LOWER ELA than pristine areas)
            suspicious_high = int(np.sum(region_means > 2 * global_ela_mean))
            suspicious_low = int(np.sum(region_means < 0.5 * global_ela_mean))
            suspicious_regions = suspicious_high + suspicious_low

            # --- Scoring ---
            variance_score = 100 - min(100, ela_variance * 20)
            mean_score = 100 - min(100, ela_mean * 5)
            gradient_score = min(100, ela_gradient * 50)
            range_score = min(100, ela_range * 30)

            # Regional CV score: high CV → likely manipulation → lower score
            regional_score = float(np.clip(100 - regional_ela_cv * 100, 0, 100))

            compression_score = int(np.clip(
                (variance_score * 0.20 + mean_score * 0.20 +
                 gradient_score * 0.20 + range_score * 0.20 +
                 regional_score * 0.20),
                0, 100
            ))

            return {
                'ela_scores': [float(x) for x in ela_scores],
                'ela_variance': float(ela_variance),
                'ela_mean': float(ela_mean),
                'ela_range': float(ela_range),
                'ela_gradient': float(ela_gradient),
                'regional_ela_cv': regional_ela_cv,
                'suspicious_regions': suspicious_regions,
                'compression_score': compression_score,
            }

        except Exception as e:
            logger.warning(f"Compression artifact analysis failed: {e}")
            return {'compression_score': 50, 'error': str(e)}

    # =======================================================================
    # Phase 2 — Advanced Detection Enhancements
    # =======================================================================

    def _analyze_noise_consistency_map(self, img: np.ndarray, gray: np.ndarray) -> Dict[str, Any]:
        """Spatial-aware splice detection via noise inconsistency mapping.

        Computes per-block noise level estimation using a sliding window and
        measures spatial variation.  Spliced regions from different sources
        have different noise levels → high local CV in the heatmap.
        """
        try:
            h, w = gray.shape
            block_size = 32
            stride = 8
            noise_levels = []

            for y in range(0, h - block_size + 1, stride):
                for x in range(0, w - block_size + 1, stride):
                    block = gray[y:y + block_size, x:x + block_size]
                    lap = cv2.Laplacian(block, cv2.CV_64F)
                    noise_levels.append(np.std(lap))

            if len(noise_levels) < 4:
                return {'noise_map_score': 50}

            noise_levels = np.array(noise_levels)
            noise_map_cv = float(np.std(noise_levels) / (np.mean(noise_levels) + 1e-10))
            noise_level_range = float(np.max(noise_levels) - np.min(noise_levels))

            # Connected component analysis on thresholded heatmap
            threshold = np.mean(noise_levels) + 1.5 * np.std(noise_levels)
            cols = max(1, (w - block_size) // stride + 1)
            rows_n = max(1, (h - block_size) // stride + 1)
            heatmap = (noise_levels > threshold).astype(np.uint8)

            # The loop produces exactly rows_n * cols elements; guard
            # against edge cases where rounding differs.
            expected_size = rows_n * cols
            if len(heatmap) == expected_size:
                heatmap_2d = heatmap.reshape(rows_n, cols)
            elif len(heatmap) > expected_size:
                heatmap_2d = heatmap[:expected_size].reshape(rows_n, cols)
            else:
                # Fewer elements than expected — pad with zeros
                padded = np.zeros(expected_size, dtype=np.uint8)
                padded[:len(heatmap)] = heatmap
                heatmap_2d = padded.reshape(rows_n, cols)

            num_islands, _ = cv2.connectedComponents(heatmap_2d)
            island_count = max(0, num_islands - 1)  # subtract background

            cv_penalty = min(60, noise_map_cv * 100)
            island_penalty = min(40, island_count * 10)
            noise_map_score = int(np.clip(100 - cv_penalty - island_penalty, 0, 100))

            return {
                'noise_map_cv': noise_map_cv,
                'noise_island_count': island_count,
                'noise_level_range': noise_level_range,
                'noise_map_score': noise_map_score,
            }
        except Exception as e:
            logger.warning(f"Noise consistency map analysis failed: {e}")
            return {'noise_map_score': 50, 'error': str(e)}

    def _analyze_jpeg_ghosts(self, image_bytes: bytes) -> Dict[str, Any]:
        """JPEG ghost detection.

        Recompresses the image at multiple quality levels and finds regions
        that converge at different quality levels — indicating copy-paste from
        a differently-compressed source.
        """
        try:
            original = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            w_img, h_img = original.size
            grid_n = 8
            qualities = list(range(50, 96, 5))

            region_optimal_q = np.zeros(grid_n * grid_n)
            region_min_diff = np.full(grid_n * grid_n, np.inf)

            for q in qualities:
                buf = io.BytesIO()
                original.save(buf, 'JPEG', quality=q)
                buf.seek(0)
                recomp = Image.open(buf)
                diff = np.array(ImageChops.difference(original, recomp)).astype(np.float32)
                diff_gray = np.mean(diff, axis=2)

                region_means = self._compute_grid_metric(
                    diff_gray, grid_n, grid_n, lambda r: np.mean(r)
                )
                improved = region_means < region_min_diff
                region_min_diff[improved] = region_means[improved]
                region_optimal_q[improved] = q

            quality_spread = float(np.max(region_optimal_q) - np.min(region_optimal_q))

            # Count regions differing from the mode
            unique, counts = np.unique(region_optimal_q, return_counts=True)
            mode_q = unique[np.argmax(counts)]
            ghost_region_count = int(np.sum(np.abs(region_optimal_q - mode_q) > 10))

            spread_penalty = min(60, quality_spread * 2)
            ghost_penalty = min(40, ghost_region_count * 5)
            # When no ghost is found (spread=0, no outlier regions), score is
            # neutral (50) — absence of JPEG ghosts doesn't confirm authenticity.
            raw_ghost = 100 - spread_penalty - ghost_penalty
            ghost_score = int(np.clip(min(raw_ghost, 50) if quality_spread == 0 and ghost_region_count == 0 else raw_ghost, 0, 100))

            return {
                'ghost_quality_spread': quality_spread,
                'ghost_region_count': ghost_region_count,
                'ghost_score': ghost_score,
            }
        except Exception as e:
            logger.warning(f"JPEG ghost analysis failed: {e}")
            return {'ghost_score': 50, 'error': str(e)}

    def _analyze_spectral_fingerprint(self, img: np.ndarray, gray: np.ndarray) -> Dict[str, Any]:
        """GAN spectral fingerprinting.

        Computes the 2D power spectrum, averages it azimuthally to get a
        radial profile, fits a power-law model, and measures residual peaks
        characteristic of GAN upsampling artifacts.
        """
        try:
            gray_resized = cv2.resize(gray, (256, 256)).astype(np.float64)

            fft = np.fft.fft2(gray_resized)
            fft_shift = np.fft.fftshift(fft)
            power = np.abs(fft_shift) ** 2

            h, w = power.shape
            cx, cy = w // 2, h // 2
            max_r = min(cx, cy)

            radial_sum = np.zeros(max_r)
            radial_count = np.zeros(max_r)

            y_grid, x_grid = np.ogrid[:h, :w]
            dist = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2).astype(int)

            for r in range(1, max_r):
                mask = dist == r
                radial_sum[r] = np.sum(power[mask])
                radial_count[r] = np.sum(mask)

            radial_profile = np.zeros(max_r)
            for r in range(1, max_r):
                if radial_count[r] > 0:
                    radial_profile[r] = np.log10(radial_sum[r] / radial_count[r] + 1e-10)

            # Linear regression on log-log for power law fit
            valid = radial_profile[2:] > 0
            r_vals = np.arange(2, max_r)
            if np.sum(valid) > 2:
                log_r = np.log10(r_vals[valid])
                log_p = radial_profile[2:][valid]
                coeffs = np.polyfit(log_r, log_p, 1)
                predicted = np.polyval(coeffs, log_r)
                residuals = log_p - predicted

                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((log_p - np.mean(log_p)) ** 2) + 1e-10
                power_law_fit_r2 = float(1 - ss_res / ss_tot)
                spectral_residual_energy = float(ss_res / len(residuals))

                # Count peaks
                res_std = np.std(residuals)
                peak_count = 0
                for i in range(1, len(residuals) - 1):
                    if (residuals[i] > 1.5 * res_std and
                            residuals[i] > residuals[i - 1] and
                            residuals[i] > residuals[i + 1]):
                        peak_count += 1
                spectral_peak_count = peak_count
            else:
                power_law_fit_r2 = 0.0
                spectral_residual_energy = 0.0
                spectral_peak_count = 0

            residual_penalty = min(40, spectral_residual_energy * 200)
            fit_penalty = min(30, (1 - power_law_fit_r2) * 30)
            peak_penalty = min(30, spectral_peak_count * 5)
            spectral_score = int(np.clip(100 - residual_penalty - fit_penalty - peak_penalty, 0, 100))

            return {
                'spectral_residual_energy': spectral_residual_energy,
                'spectral_peak_count': spectral_peak_count,
                'power_law_fit_r2': power_law_fit_r2,
                'spectral_score': spectral_score,
            }
        except Exception as e:
            logger.warning(f"Spectral fingerprint analysis failed: {e}")
            return {'spectral_score': 50, 'error': str(e)}

    def _analyze_channel_correlation(self, img: np.ndarray) -> Dict[str, Any]:
        """Cross-channel correlation analysis.

        Real cameras produce correlated noise across R/G/B channels (from
        Bayer demosaicing). GAN-generated images often have independently
        generated channel noise.
        """
        try:
            b_ch, g_ch, r_ch = cv2.split(img)

            # Noise residual = original - median filtered
            r_med = cv2.medianBlur(r_ch, 5)
            g_med = cv2.medianBlur(g_ch, 5)
            b_med = cv2.medianBlur(b_ch, 5)

            r_noise = r_ch.astype(np.float64) - r_med.astype(np.float64)
            g_noise = g_ch.astype(np.float64) - g_med.astype(np.float64)
            b_noise = b_ch.astype(np.float64) - b_med.astype(np.float64)

            # Pairwise Pearson correlation (guard against constant arrays
            # which produce NaN from np.corrcoef due to zero std)
            def pearson(a, b):
                a_flat, b_flat = a.ravel(), b.ravel()
                if np.std(a_flat) < 1e-10 or np.std(b_flat) < 1e-10:
                    return 0.0
                val = np.corrcoef(a_flat, b_flat)[0, 1]
                return 0.0 if np.isnan(val) else float(val)

            corr_rg = pearson(r_noise, g_noise)
            corr_rb = pearson(r_noise, b_noise)
            corr_gb = pearson(g_noise, b_noise)

            # Bayer pattern periodicity (vectorized)
            g_flat = g_noise
            h, w = g_flat.shape
            tl = g_flat[0::2, 0::2]
            tr = g_flat[0::2, 1::2]
            bl = g_flat[1::2, 0::2]
            br = g_flat[1::2, 1::2]
            # Ensure shapes match (truncate to common size)
            min_h = min(tl.shape[0], bl.shape[0])
            min_w = min(tl.shape[1], tr.shape[1])
            tl = tl[:min_h, :min_w]
            tr = tr[:min_h, :min_w]
            bl = bl[:min_h, :min_w]
            br = br[:min_h, :min_w]
            bayer_periodicity = float(np.mean(np.abs((tl + br) - (tr + bl))))

            avg_corr = (abs(corr_rg) + abs(corr_rb) + abs(corr_gb)) / 3
            corr_score = float(np.clip((avg_corr - 0.1) / (0.6 - 0.1) * 80 + 20, 0, 100))
            bayer_score = float(np.clip(bayer_periodicity * 20, 0, 30))
            correlation_score = int(np.clip(corr_score * 0.8 + bayer_score * 0.2, 0, 100))

            return {
                'channel_noise_corr_rg': corr_rg,
                'channel_noise_corr_rb': corr_rb,
                'channel_noise_corr_gb': corr_gb,
                'bayer_periodicity': bayer_periodicity,
                'correlation_score': correlation_score,
            }
        except Exception as e:
            logger.warning(f"Channel correlation analysis failed: {e}")
            return {'correlation_score': 50, 'error': str(e)}

    # =======================================================================
    # Phase 3 — Advanced Image Enhancements
    # =======================================================================

    def _analyze_benford_dct(self, img: np.ndarray, gray: np.ndarray) -> Dict[str, Any]:
        """Benford's Law analysis of DCT coefficients.

        The first significant digits of AC coefficients in 8×8 block DCTs
        of natural images follow a generalized Benford distribution:
            P(d) = log10(1 + 1/d),  d = 1..9
        Any manipulation (splicing, AI generation, contrast edits) disrupts
        this distribution.  We measure divergence via KL-divergence and a
        chi-squared goodness-of-fit test.
        """
        try:
            gray = gray.astype(np.float32)
            h, w = gray.shape

            # Collect first significant digits from 8×8 block DCT AC coefficients
            digits = []
            for y in range(0, h - 7, 8):
                for x in range(0, w - 7, 8):
                    block = gray[y:y + 8, x:x + 8]
                    dct_block = cv2.dct(block)
                    # Flatten and skip DC component (index 0)
                    ac = dct_block.ravel()[1:]
                    for val in ac:
                        absval = abs(val)
                        if absval >= 1.0:
                            # Extract first significant digit
                            d = int(str(f'{absval:.6e}')[0])
                            if 1 <= d <= 9:
                                digits.append(d)

            if len(digits) < 100:
                return {'benford_score': 50}

            # Observed distribution
            counts = np.zeros(9)
            for d in digits:
                counts[d - 1] += 1
            observed = counts / counts.sum()

            # Expected Benford distribution
            expected = np.array([np.log10(1 + 1 / d) for d in range(1, 10)])

            # KL-divergence: D_KL(observed || expected)
            kl_div = float(np.sum(observed * np.log((observed + 1e-10) / (expected + 1e-10))))

            # Chi-squared statistic
            n = len(digits)
            chi2 = float(np.sum((counts - n * expected) ** 2 / (n * expected + 1e-10)))

            # Score: JPEG compression already disrupts the Benford distribution,
            # so real JPEG images typically have KL 0.05-0.5.  Only extreme
            # deviations (KL > 0.8, e.g. pure gradients) are diagnostic.
            # Map KL [0, 1.0] → [100, 20] with gentle slope.
            # Cap at 60: absence of Benford violation is neutral, not positive.
            # All JPEGs (real and AI) have low KL; only extreme deviation is diagnostic.
            benford_score = int(np.clip(60 - kl_div * 50, 20, 60))

            return {
                'benford_kl_divergence': kl_div,
                'benford_chi_squared': chi2,
                'benford_digit_count': len(digits),
                'benford_score': benford_score,
            }
        except Exception as e:
            logger.warning(f"Benford DCT analysis failed: {e}")
            return {'benford_score': 50, 'error': str(e)}

    def _analyze_wavelet_spectral_ratio(self, img: np.ndarray, gray: np.ndarray) -> Dict[str, Any]:
        """Wavelet spectral ratio for diffusion model detection.

        Diffusion models leave a characteristic signature: unnaturally low
        energy in the finest-scale wavelet detail subbands (HH1, HL1, LH1)
        relative to the approximation (LL1).  The denoising process
        progressively removes fine-scale noise.

        Natural images:  WSR = E(HH1)/E(LL1) > 0.01
        Diffusion:       WSR = E(HH1)/E(LL1) < 0.005

        We also check the cross-scale energy decay rate — natural images
        follow an approximate power-law decay while diffusion outputs
        show faster-than-expected falloff.
        """
        try:
            gray = gray.astype(np.float64)
            # Resize for consistent analysis
            gray = cv2.resize(gray, (256, 256))

            # 3-level DWT using Daubechies-4 wavelet
            coeffs = pywt.wavedec2(gray, 'db4', level=3)
            # coeffs = [cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)]

            def energy(arr):
                return float(np.mean(arr ** 2))

            ll_energy = energy(coeffs[0])  # LL3 (coarsest approximation)

            # Per-level detail energies
            level_energies = []
            for level_idx in range(1, len(coeffs)):
                cH, cV, cD = coeffs[level_idx]
                e_h = energy(cH)
                e_v = energy(cV)
                e_d = energy(cD)
                level_energies.append({
                    'horizontal': e_h,
                    'vertical': e_v,
                    'diagonal': e_d,
                    'total': e_h + e_v + e_d,
                })

            # Finest-scale detail energy (level 1 = last element)
            finest = level_energies[-1]
            finest_total = finest['total']

            # Wavelet Spectral Ratio: finest detail / coarsest approximation
            wsr = finest_total / (ll_energy + 1e-10)

            # Cross-scale energy decay: ratio of adjacent levels
            # level_energies[0] = level 3 (coarsest detail)
            # level_energies[2] = level 1 (finest detail)
            decay_rates = []
            for i in range(len(level_energies) - 1):
                coarser = level_energies[i]['total']
                finer = level_energies[i + 1]['total']
                if coarser > 1e-10:
                    decay_rates.append(finer / coarser)

            avg_decay = float(np.mean(decay_rates)) if decay_rates else 1.0

            # Diagonal-to-total ratio at finest scale
            diag_ratio = finest['diagonal'] / (finest_total + 1e-10)

            # Score: JPEG compression + resize destroys absolute WSR, so we
            # use cross-scale decay as the primary discriminator.
            # Real images: decay 0.2-1.5 (fine detail present at all scales)
            # Smooth synthetics: decay < 0.05 (no fine-scale structure)
            # Map decay [0.01, 0.8] → [20, 100]
            decay_score = float(np.clip(
                (avg_decay - 0.01) / (0.8 - 0.01) * 80 + 20, 0, 100
            ))

            # WSR as secondary signal (small contribution)
            wsr_score = float(np.clip(wsr * 5000, 0, 40))

            wavelet_score = int(np.clip(decay_score * 0.8 + wsr_score * 0.2, 0, 100))

            return {
                'wavelet_spectral_ratio': float(wsr),
                'finest_detail_energy': float(finest_total),
                'approx_energy': float(ll_energy),
                'cross_scale_decay': avg_decay,
                'diag_ratio': float(diag_ratio),
                'wavelet_score': wavelet_score,
            }
        except Exception as e:
            logger.warning(f"Wavelet spectral ratio analysis failed: {e}")
            return {'wavelet_score': 50, 'error': str(e)}

    def _analyze_blocking_artifact_grid(self, img: np.ndarray, gray: np.ndarray) -> Dict[str, Any]:
        """Blocking Artifact Grid (BAG) analysis for double JPEG detection.

        JPEG compression creates 8×8 block boundaries with measurable
        discontinuities.  When an image is re-saved (or a region pasted
        from a differently-aligned JPEG), there may be *two* 8-pixel grids
        — the current grid and the previous grid.

        Detection: compute blockiness B(offset) for all 8 possible offsets,
        find primary (strongest) and secondary grids.  A strong secondary
        grid indicates double compression with a different alignment.
        """
        try:
            gray = gray.astype(np.float64)
            h, w = gray.shape

            if h < 16 or w < 16:
                return {'bag_score': 50}

            # Compute blockiness for each offset (0..7) in both directions (vectorized)
            h_blockiness = np.zeros(8)
            v_blockiness = np.zeros(8)

            for offset in range(8):
                # Horizontal blockiness: average |I(x, y) - I(x+1, y)|
                # at columns x = offset, offset+8, offset+16, ...
                cols = np.arange(offset, w - 1, 8)
                if len(cols) > 0:
                    h_blockiness[offset] = float(np.mean(np.abs(
                        gray[:, cols] - gray[:, cols + 1])))

                # Vertical blockiness
                rows = np.arange(offset, h - 1, 8)
                if len(rows) > 0:
                    v_blockiness[offset] = float(np.mean(np.abs(
                        gray[rows, :] - gray[rows + 1, :])))

            # Combined blockiness per offset
            blockiness = h_blockiness + v_blockiness

            # Primary grid: offset with highest blockiness
            primary_offset = int(np.argmax(blockiness))
            primary_strength = float(blockiness[primary_offset])

            # Secondary grid: highest blockiness excluding primary
            mask = np.ones(8, dtype=bool)
            mask[primary_offset] = False
            remaining = blockiness[mask]
            remaining_offsets = np.arange(8)[mask]

            secondary_idx = int(np.argmax(remaining))
            secondary_offset = int(remaining_offsets[secondary_idx])
            secondary_strength = float(remaining[secondary_idx])

            # Average non-grid blockiness (baseline)
            baseline_mask = mask.copy()
            baseline_mask[remaining_offsets[secondary_idx]] = False
            baseline = float(np.mean(blockiness[baseline_mask])) if np.sum(baseline_mask) > 0 else 0

            # Grid strength ratio: primary over baseline
            primary_snr = primary_strength / (baseline + 1e-10)

            # Secondary strength ratio: secondary over baseline
            secondary_snr = secondary_strength / (baseline + 1e-10)

            # Dual grid: significant secondary grid indicates double JPEG.
            # Conservative detection: require the secondary grid at a
            # DIFFERENT offset from primary AND both primary and secondary
            # must stand out clearly from the remaining offsets.
            grid_diff = abs(primary_offset - secondary_offset)

            # The secondary must be significantly above the average of
            # the remaining 6 offsets (excluding primary and secondary)
            tertiary_mask = np.ones(8, dtype=bool)
            tertiary_mask[primary_offset] = False
            tertiary_mask[secondary_offset] = False
            tertiary_avg = float(np.mean(blockiness[tertiary_mask])) if np.sum(tertiary_mask) > 0 else baseline
            secondary_above_tertiary = secondary_strength / (tertiary_avg + 1e-10)

            dual_grid = (
                grid_diff > 0 and
                secondary_above_tertiary > 1.5 and
                primary_snr > 2.0 and
                secondary_snr > 1.8
            )

            # Score: no dual grid = high score (natural)
            # Strong dual grid = low score (double compression)
            if dual_grid:
                dual_penalty = min(50, (secondary_above_tertiary - 1.0) * 30)
                bag_score = int(np.clip(100 - dual_penalty - 20, 0, 100))
            else:
                # No dual grid → neutral score (50). Absence of double-compression
                # artifacts does not indicate authenticity — AI images also lack them.
                bag_score = 50

            return {
                'primary_grid_offset': primary_offset,
                'primary_grid_snr': float(primary_snr),
                'secondary_grid_offset': secondary_offset,
                'secondary_grid_snr': float(secondary_snr),
                'dual_grid_detected': dual_grid,
                'grid_alignment_diff': grid_diff,
                'bag_score': bag_score,
            }
        except Exception as e:
            logger.warning(f"BAG analysis failed: {e}")
            return {'bag_score': 50, 'error': str(e)}

    def _analyze_cfa_demosaicing(self, img: np.ndarray) -> Dict[str, Any]:
        """CFA / demosaicing peak detection.

        Real cameras use a Bayer Color Filter Array where each pixel
        captures only one color channel.  Demosaicing interpolation creates
        periodic correlation peaks in the 2D FFT of inter-channel
        difference images at the Bayer frequencies:
            (π, π), (π, 0), (0, π)
        i.e. at the Nyquist frequency along each axis.

        AI-generated images lack these peaks.  Resampled images destroy
        the CFA pattern.  This is the rigorous version of the existing
        Bayer periodicity check in channel_correlation.
        """
        try:
            b_ch = img[:, :, 0].astype(np.float64)
            g_ch = img[:, :, 1].astype(np.float64)
            r_ch = img[:, :, 2].astype(np.float64)
            h, w = g_ch.shape

            if h < 32 or w < 32:
                return {'cfa_score': 50}

            # Inter-channel difference images
            diff_gr = g_ch - r_ch
            diff_gb = g_ch - b_ch

            peak_snrs = []
            for diff_img in [diff_gr, diff_gb]:
                # 2D FFT
                fft = np.fft.fft2(diff_img)
                fft_shift = np.fft.fftshift(fft)
                magnitude = np.abs(fft_shift)

                cy, cx = h // 2, w // 2

                # Expected Bayer peak locations (relative to shifted center):
                # (π,π) → corner quadrants of FFT
                # (π,0) → left/right edges of FFT
                # (0,π) → top/bottom edges of FFT
                # After fftshift these are at the edges/corners of the array

                # Sample peak regions (small windows around expected frequencies)
                # (π,π) → corners: (0,0), (0,w-1), (h-1,0), (h-1,w-1) in shifted
                # (π,0) → (0, cx), (h-1, cx)
                # (0,π) → (cy, 0), (cy, w-1)
                peak_window = 3  # radius around expected location

                def peak_energy(cy_p, cx_p):
                    y0 = max(0, cy_p - peak_window)
                    y1 = min(h, cy_p + peak_window + 1)
                    x0 = max(0, cx_p - peak_window)
                    x1 = min(w, cx_p + peak_window + 1)
                    return float(np.max(magnitude[y0:y1, x0:x1]))

                # Background energy (median of magnitude, excluding DC and peaks)
                # Use a ring around center (mid-frequencies) as baseline
                r_inner = min(h, w) // 6
                r_outer = min(h, w) // 3
                y_grid, x_grid = np.ogrid[:h, :w]
                dist = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
                ring_mask = (dist > r_inner) & (dist < r_outer)
                background = float(np.median(magnitude[ring_mask])) if np.any(ring_mask) else 1.0

                # Check all Bayer peak locations
                peaks = []
                # (π,π) corners
                for py, px in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
                    peaks.append(peak_energy(py, px))
                # (π,0) top/bottom center
                for py, px in [(0, cx), (h - 1, cx)]:
                    peaks.append(peak_energy(py, px))
                # (0,π) left/right center
                for py, px in [(cy, 0), (cy, w - 1)]:
                    peaks.append(peak_energy(py, px))

                # SNR: max peak over background
                max_peak = max(peaks) if peaks else 0
                snr = max_peak / (background + 1e-10)
                peak_snrs.append(snr)

            avg_snr = float(np.mean(peak_snrs))
            max_snr = float(max(peak_snrs))

            # CFA present if both G-R and G-B show clear peaks
            cfa_present = avg_snr > 2.0

            # Cap at 55: CFA peaks can be mimicked by geometric structure in
            # synthetic images, so high SNR is only weakly indicative.
            # Low SNR (no CFA) is more reliably suspicious.
            snr_score = float(np.clip((avg_snr - 1.0) / (5.0 - 1.0) * 35 + 20, 0, 55))
            cfa_score = int(np.clip(snr_score, 0, 55))

            return {
                'cfa_peak_snr_gr': float(peak_snrs[0]) if len(peak_snrs) > 0 else 0,
                'cfa_peak_snr_gb': float(peak_snrs[1]) if len(peak_snrs) > 1 else 0,
                'cfa_avg_snr': avg_snr,
                'cfa_present': cfa_present,
                'cfa_score': cfa_score,
            }
        except Exception as e:
            logger.warning(f"CFA demosaicing analysis failed: {e}")
            return {'cfa_score': 50, 'error': str(e)}

    def _calculate_composite_score(self, results: Dict[str, Any]) -> int:
        """Calculate final composite score from all analyses.

        Weights are based on discriminative power of each analysis:
        - Noise analysis: High discriminative power
        - Frequency analysis: High discriminative power
        - Lighting: Moderate discriminative power
        - Texture: Moderate discriminative power
        - Geometric: Lower discriminative power (depends on image content)
        - Color: Moderate discriminative power
        - Compression: Lower discriminative power (depends on source)
        """
        try:
            # Weights recalibrated based on discriminative power.
            # Phase 3 methods get conservative weights (0.02-0.03) so they
            # add signal without diluting the proven Phase 1+2 methods.
            weights = {
                # Phase 1 (8 core methods) — largely unchanged
                'noise_uniformity': 0.12,
                'noise_frequency': 0.07,
                'frequency_analysis': 0.14,
                'geometric_consistency': 0.09,
                'lighting_analysis': 0.11,
                'texture_analysis': 0.11,
                'color_analysis': 0.05,
                'compression_analysis': 0.05,
                # Phase 2 (4 methods)
                'noise_consistency_map': 0.04,
                'jpeg_ghost': 0.04,
                'spectral_fingerprint': 0.04,
                'channel_correlation': 0.04,
                # Phase 3 (4 new methods) — conservative weights
                'benford_dct': 0.02,
                'wavelet_ratio': 0.03,
                'blocking_grid': 0.02,
                'cfa_demosaicing': 0.03,
            }
            
            total_score = 0
            total_weight = 0
            
            for analysis_type, weight in weights.items():
                if analysis_type in results:
                    result = results[analysis_type]
                    if isinstance(result, dict):
                        # Map analysis types to their score keys
                        score_key_map = {
                            'noise_uniformity': 'uniformity_score',
                            'noise_frequency': 'frequency_score',
                            'frequency_analysis': 'dct_score',
                            'geometric_consistency': 'geometric_score',
                            'lighting_analysis': 'lighting_score',
                            'texture_analysis': 'texture_score',
                            'color_analysis': 'color_score',
                            'compression_analysis': 'compression_score',
                            'noise_consistency_map': 'noise_map_score',
                            'jpeg_ghost': 'ghost_score',
                            'spectral_fingerprint': 'spectral_score',
                            'channel_correlation': 'correlation_score',
                            # Phase 3
                            'benford_dct': 'benford_score',
                            'wavelet_ratio': 'wavelet_score',
                            'blocking_grid': 'bag_score',
                            'cfa_demosaicing': 'cfa_score',
                        }
                        
                        score_key = score_key_map.get(analysis_type, 'score')
                        score = result.get(score_key, 50)
                        
                        if isinstance(score, (int, float)):
                            total_score += score * weight
                            total_weight += weight
            
            if total_weight > 0:
                final_score = int(total_score / total_weight)
            else:
                final_score = 50
            
            return min(100, max(0, final_score))
            
        except Exception as e:
            logger.warning(f"Composite score calculation failed: {e}")
            return 50

    def _compute_grid_metric(self, data: np.ndarray, rows: int, cols: int,
                              metric_fn) -> np.ndarray:
        """Apply metric_fn to each cell of a rows×cols grid over 2D data."""
        h, w = data.shape[:2]
        values = []
        for i in range(rows):
            for j in range(cols):
                region = data[i * h // rows:(i + 1) * h // rows,
                              j * w // cols:(j + 1) * w // cols]
                values.append(metric_fn(region))
        return np.array(values)

    def _divide_image_into_grid(self, img: np.ndarray, rows: int, cols: int) -> list:
        """Divide image into grid of regions."""
        h, w = img.shape[:2]
        regions = []
        
        for i in range(rows):
            for j in range(cols):
                start_y = i * h // rows
                end_y = (i + 1) * h // rows
                start_x = j * w // cols
                end_x = (j + 1) * w // cols
                
                region = img[start_y:end_y, start_x:end_x]
                regions.append(region)
        
        return regions

    def _calculate_lbp(self, gray_img: np.ndarray, radius: int = 1, neighbors: int = 8) -> np.ndarray:
        """Calculate Local Binary Pattern using vectorized numpy shifts.

        Uses np.roll for each of the 8 neighbours at *radius* distance,
        replacing the previous O(h*w*8) pure-Python loop.
        """
        gray = gray_img.astype(np.int16)
        lbp = np.zeros(gray.shape, dtype=np.uint8)

        # 8 neighbours: N, NE, E, SE, S, SW, W, NW
        shifts = [
            (-radius, 0),
            (-radius, radius),
            (0, radius),
            (radius, radius),
            (radius, 0),
            (radius, -radius),
            (0, -radius),
            (-radius, -radius),
        ]

        for bit, (dy, dx) in enumerate(shifts):
            neighbor = np.roll(np.roll(gray, -dy, axis=0), -dx, axis=1)
            lbp |= ((neighbor >= gray).astype(np.uint8) << bit)

        return lbp

    def _calculate_point_density(self, points: np.ndarray, shape: tuple) -> float:
        """Calculate point density consistency in image."""
        if len(points) == 0:
            return 0
        
        h, w = shape[:2]
        
        # Divide image into grid and count points per cell
        grid_size = 10
        grid_h = max(1, h // grid_size)
        grid_w = max(1, w // grid_size)
        
        point_counts = np.zeros((grid_h, grid_w))
        
        for point in points:
            grid_i = min(int(point[1] / grid_size), grid_h - 1)
            grid_j = min(int(point[0] / grid_size), grid_w - 1)
            point_counts[grid_i, grid_j] += 1
        
        # Calculate density variance (lower = more uniform = more natural)
        non_empty = point_counts[point_counts > 0]
        if len(non_empty) > 0:
            density_variance = np.var(non_empty)
            density_score = max(0, 100 - density_variance * 5)
        else:
            density_score = 0
        
        return density_score