"""
Enhanced Physics-based Verification for AI-generated Image Detection.

This module implements multiple mathematical analysis techniques to detect
AI-generated or manipulated images based on physical inconsistencies.
"""
import cv2
import numpy as np
from PIL import Image, ImageChops
import io
import logging
from typing import Dict, Any, Optional, Tuple

from .types import LayerResult

# Configure logging
logger = logging.getLogger(__name__)


class PhysicsVerifier:
    """Enhanced physics-based verification layer for detecting AI-generated images.
    
    This class implements multiple analysis techniques:
    1. Noise Pattern Analysis - Real camera sensors produce characteristic noise
    2. Frequency Domain Analysis - DCT/FFT patterns differ in AI images
    3. Geometric Consistency - Lines, perspectives, and shapes
    4. Lighting Analysis - Shadow and light direction consistency
    5. Texture Analysis - LBP and GLCM features
    6. Color Distribution - HSV/LAB color space analysis
    7. Compression Artifacts - ELA (Error Level Analysis)
    
    Each analysis returns a score from 0-100 where:
    - Higher scores indicate more "natural" characteristics
    - Lower scores indicate potential AI generation
    """

    # AI detection thresholds (calibrated based on test data)
    THRESHOLDS = {
        'noise_uniformity_high': 70,  # Above this = suspicious (too uniform)
        'noise_uniformity_low': 30,   # Below this = good (natural variation)
        'variance_mean_high': 80,     # Above this variance_mean = AI artifact (very important!)
        'dct_high': 65,               # Higher DCT score = more natural
        'dct_low': 35,                # Lower DCT score = more AI-like
        'lighting_high': 55,          # Higher = more consistent lighting
        'lighting_low': 25,           # Lower = inconsistent (AI-like)
        'texture_high': 60,           # Higher = more natural texture
        'texture_low': 30,            # Lower = synthetic texture
        'geometric_high': 55,         # Higher = more consistent geometry
        'geometric_low': 25,          # Lower = inconsistent geometry
        'color_high': 50,             # Higher = natural color distribution
        'color_low': 20,              # Lower = unnatural colors
        'compression_high': 85,       # Higher = natural compression pattern
        'compression_low': 40,       # Lower = unusual compression
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

            # Run all analysis modules
            results: Dict[str, Any] = {}
            
            # 1. Noise Pattern Analysis
            results['noise_uniformity'] = self._analyze_noise_uniformity(img)
            results['noise_frequency'] = self._analyze_noise_frequency(img)
            
            # 2. Frequency Domain Analysis (DCT/FFT)
            results['frequency_analysis'] = self._analyze_frequency_domain(img)
            
            # 3. Geometric Consistency Analysis
            results['geometric_consistency'] = self._analyze_geometric_consistency(img)
            
            # 4. Lighting and Shadow Analysis
            results['lighting_analysis'] = self._analyze_lighting_consistency(img)
            
            # 5. Texture and Pattern Analysis
            results['texture_analysis'] = self._analyze_texture_patterns(img)
            
            # 6. Color Distribution Analysis
            results['color_analysis'] = self._analyze_color_distribution(img)
            
            # 7. Compression Artifact Analysis
            results['compression_analysis'] = self._analyze_compression_artifacts(image_bytes)

            # Calculate weighted composite score
            final_score = self._calculate_composite_score(results)
            
            # Count AI indicators based on calibrated thresholds
            ai_indicators = self._count_ai_indicators(results)
            total_indicators = 9  # 7 original + 2 for variance_mean (double weight)
            
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

    def _count_ai_indicators(self, results: Dict[str, Any]) -> int:
        """Count the number of AI indicators based on analysis results."""
        indicators = 0
        
        # CRITICAL: Variance mean - AI images have unnaturally high variance
        # This is the strongest indicator from our test data
        variance_mean = results.get('noise_uniformity', {}).get('variance_mean', 0)
        if variance_mean > self.THRESHOLDS['variance_mean_high']:
            indicators += 2  # Double weight - this is very important
        
        # Noise uniformity - too uniform is suspicious
        noise_score = results.get('noise_uniformity', {}).get('uniformity_score', 50)
        if noise_score > self.THRESHOLDS['noise_uniformity_high']:
            indicators += 1
        
        # DCT score - low is suspicious
        dct_score = results.get('frequency_analysis', {}).get('dct_score', 50)
        if dct_score < self.THRESHOLDS['dct_low']:
            indicators += 1
        
        # Lighting - low consistency is suspicious
        lighting_score = results.get('lighting_analysis', {}).get('lighting_score', 50)
        if lighting_score < self.THRESHOLDS['lighting_low']:
            indicators += 1
        
        # Texture - low complexity is suspicious
        texture_score = results.get('texture_analysis', {}).get('texture_score', 50)
        if texture_score < self.THRESHOLDS['texture_low']:
            indicators += 1
        
        # Geometric - low consistency is suspicious
        geo_score = results.get('geometric_consistency', {}).get('geometric_score', 50)
        if geo_score < self.THRESHOLDS['geometric_low']:
            indicators += 1
        
        # Color - unnatural distribution is suspicious
        color_score = results.get('color_analysis', {}).get('color_score', 50)
        if color_score < self.THRESHOLDS['color_low']:
            indicators += 1
        
        # Compression - unusual patterns are suspicious
        compression_score = results.get('compression_analysis', {}).get('compression_score', 50)
        if compression_score < self.THRESHOLDS['compression_low']:
            indicators += 1
        
        return indicators

    def _determine_result(
        self, 
        score: int, 
        ai_probability: float,
        results: Dict[str, Any]
    ) -> Tuple[bool, int, Optional[str]]:
        """Determine verification result based on score and AI probability."""
        
        if ai_probability >= 0.5:  # Strong AI indicators (4.5+ out of 9)
            warning = 'Strong physical inconsistencies detected - likely AI-generated'
            adjusted_score = int(score * 0.5)
            return False, adjusted_score, warning
        
        elif ai_probability >= 0.33:  # Moderate AI indicators (3 out of 9)
            warning = 'Some physical inconsistencies detected - possibly AI-generated'
            adjusted_score = int(score * 0.7)
            passed = adjusted_score >= 50
            return passed, adjusted_score, warning
        
        elif ai_probability >= 0.15:  # Mild indicators (1-2 out of 9)
            warning = 'Minor physical inconsistencies detected - review recommended'
            passed = score >= 50
            return passed, score, warning if not passed else None
        
        else:  # Low AI indicators (0 out of 9)
            passed = score >= 45
            return passed, score, None

    def _analyze_noise_uniformity(self, img: np.ndarray) -> Dict[str, Any]:
        """Analyze noise uniformity - AI images often have unnaturally uniform noise.
        
        Real camera sensors produce noise that varies across the image due to:
        - Different ISO sensitivity regions
        - Temperature variations
        - Sensor imperfections
        
        AI-generated images often have uniform noise patterns.
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Divide image into 16 regions (4x4 grid) for better analysis
            h, w = gray.shape
            regions = []
            for i in range(4):
                for j in range(4):
                    region = gray[i*h//4:(i+1)*h//4, j*w//4:(j+1)*w//4]
                    regions.append(region)
            
            # Calculate noise variance in each region using Laplacian
            variances = []
            for region in regions:
                laplacian = cv2.Laplacian(region, cv2.CV_64F)
                variance = laplacian.var()
                variances.append(variance)
            
            variances = np.array(variances)
            variance_mean = np.mean(variances)
            variance_std = np.std(variances)
            
            # Calculate coefficient of variation (CV)
            # Lower CV = more uniform = more likely AI
            cv_ratio = variance_std / (variance_mean + 1e-10)
            
            # Score: Higher CV (more variation) = more natural = higher score
            # CV typically ranges from 0.1 to 1.0 for natural images
            uniformity_score = int(min(100, max(0, cv_ratio * 150)))
            
            # Also check absolute variance - very high variance can indicate AI artifacts
            variance_score = int(min(100, max(0, 100 - (variance_mean / 1000) * 50)))
            
            # Combined score
            combined_score = int((uniformity_score * 0.6 + variance_score * 0.4))
            
            return {
                'variance_mean': float(variance_mean),
                'variance_std': float(variance_std),
                'cv_ratio': float(cv_ratio),
                'uniformity_score': combined_score,
                'variation_score': uniformity_score,
                'variance_quality_score': variance_score
            }
            
        except Exception as e:
            logger.warning(f"Noise uniformity analysis failed: {e}")
            return {'uniformity_score': 50, 'error': str(e)}

    def _analyze_noise_frequency(self, img: np.ndarray) -> Dict[str, Any]:
        """Analyze noise frequency characteristics using FFT.
        
        Real images have characteristic frequency distributions based on:
        - Lens characteristics
        - Sensor frequency response
        - Natural scene statistics
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
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

    def _analyze_frequency_domain(self, img: np.ndarray) -> Dict[str, Any]:
        """Analyze DCT coefficients - AI images have characteristic DCT patterns.
        
        DCT analysis reveals:
        - Compression artifacts
        - Frequency distribution anomalies
        - AI-specific spectral signatures
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
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

    def _analyze_geometric_consistency(self, img: np.ndarray) -> Dict[str, Any]:
        """Analyze geometric consistency - lines, perspectives, shapes.
        
        AI-generated images often have:
        - Inconsistent line angles
        - Warped perspectives
        - Irregular shapes
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
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

    def _analyze_lighting_consistency(self, img: np.ndarray) -> Dict[str, Any]:
        """Enhanced lighting and shadow analysis.
        
        Real images have:
        - Consistent light direction
        - Natural shadow gradients
        - Realistic highlight distribution
        """
        try:
            # Convert to different color spaces
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
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

    def _analyze_texture_patterns(self, img: np.ndarray) -> Dict[str, Any]:
        """Analyze texture patterns using statistical methods.
        
        AI-generated images often have:
        - Unnatural texture regularity
        - Missing fine details
        - Incorrect texture scaling
        """
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate Local Binary Patterns (LBP)
            lbp = self._calculate_lbp(gray)
            
            # Analyze texture statistics
            lbp_hist = np.histogram(lbp.flatten(), bins=256, range=(0, 256))[0]
            lbp_hist = lbp_hist / (np.sum(lbp_hist) + 1e-10)
            
            # Calculate texture entropy
            texture_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))
            
            # Calculate LBP uniformity (measure of texture regularity)
            lbp_uniformity = np.sum(lbp_hist ** 2)
            
            # Use a simplified GLCM approach
            # Calculate gray-level co-occurrence features
            gray_uint8 = gray.astype(np.uint8)
            
            # Calculate gradient-based texture features
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Texture energy (mean gradient magnitude)
            texture_energy = np.mean(grad_mag)
            
            # Texture contrast (std of gradient magnitude)
            texture_contrast = np.std(grad_mag)
            
            # Homogeneity (inverse of gradient magnitude)
            homogeneity = 1.0 / (1.0 + np.mean(grad_mag))
            
            # Score calculation
            # Natural images have:
            # - High entropy (diverse textures)
            # - Moderate uniformity
            # - Good texture energy
            
            entropy_score = min(100, texture_entropy * 12)  # Entropy ~5-8 for natural
            uniformity_penalty = lbp_uniformity * 100  # High uniformity is bad
            energy_score = min(100, texture_energy * 2)
            
            texture_score = int(np.clip(
                (entropy_score * 0.4 + (100 - uniformity_penalty) * 0.3 + energy_score * 0.3),
                0, 100
            ))
            
            return {
                'texture_entropy': float(texture_entropy),
                'texture_uniformity': float(lbp_uniformity),
                'texture_energy': float(texture_energy),
                'texture_contrast': float(texture_contrast),
                'homogeneity': float(homogeneity),
                'texture_score': texture_score
            }
            
        except Exception as e:
            logger.warning(f"Texture pattern analysis failed: {e}")
            return {'texture_score': 50, 'error': str(e)}

    def _analyze_color_distribution(self, img: np.ndarray) -> Dict[str, Any]:
        """Analyze color distribution and consistency.
        
        AI-generated images often have:
        - Unnatural color distributions
        - Missing or excessive saturation
        - Incorrect color relationships
        """
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
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
        """Analyze compression artifacts and ELA patterns.
        
        AI-generated images often show:
        - Different compression patterns
        - Unusual ELA distributions
        - Inconsistent artifact localization
        """
        try:
            # Enhanced Error Level Analysis
            original = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # Save at different quality levels and analyze differences
            qualities = [95, 85, 75, 65]
            ela_scores = []
            
            for quality in qualities:
                buffer = io.BytesIO()
                original.save(buffer, 'JPEG', quality=quality)
                buffer.seek(0)
                recompressed = Image.open(buffer)
                
                # Calculate difference
                diff = ImageChops.difference(original, recompressed)
                diff_array = np.array(diff)
                
                # Calculate ELA score (average difference)
                ela_score = np.mean(diff_array)
                ela_scores.append(ela_score)
            
            ela_scores = np.array(ela_scores)
            
            # Analyze ELA patterns
            ela_variance = np.var(ela_scores)
            ela_mean = np.mean(ela_scores)
            ela_range = np.max(ela_scores) - np.min(ela_scores)
            
            # Calculate ELA gradient (how quickly differences increase with compression)
            ela_gradient = (ela_scores[-1] - ela_scores[0]) / (qualities[-1] - qualities[0])
            
            # Natural images typically have:
            # - Moderate ELA variance
            # - Consistent ELA gradient
            # - ELA mean in certain range
            
            variance_score = 100 - min(100, ela_variance * 20)
            mean_score = 100 - min(100, ela_mean * 5)
            gradient_score = min(100, ela_gradient * 50)
            range_score = min(100, ela_range * 30)
            
            compression_score = int(np.clip(
                (variance_score * 0.25 + mean_score * 0.25 + 
                 gradient_score * 0.25 + range_score * 0.25),
                0, 100
            ))
            
            return {
                'ela_scores': [float(x) for x in ela_scores],
                'ela_variance': float(ela_variance),
                'ela_mean': float(ela_mean),
                'ela_range': float(ela_range),
                'ela_gradient': float(ela_gradient),
                'compression_score': compression_score
            }
            
        except Exception as e:
            logger.warning(f"Compression artifact analysis failed: {e}")
            return {'compression_score': 50, 'error': str(e)}

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
            # Weight different analyses based on their discriminative power
            weights = {
                'noise_uniformity': 0.20,
                'noise_frequency': 0.10,
                'frequency_analysis': 0.18,
                'geometric_consistency': 0.10,
                'lighting_analysis': 0.15,
                'texture_analysis': 0.12,
                'color_analysis': 0.10,
                'compression_analysis': 0.05
            }
            
            total_score = 0
            total_weight = 0
            
            for analysis_type, weight in weights.items():
                if analysis_type in results:
                    result = results[analysis_type]
                    if isinstance(result, dict):
                        # Extract score from nested dictionary
                        score_key = f"{analysis_type.rstrip('_analysis').rstrip('_consistency').rstrip('_uniformity').rstrip('_frequency').rstrip('_patterns').rstrip('_distribution').rstrip('_artifacts')}_score"
                        
                        # Map analysis types to their score keys
                        score_key_map = {
                            'noise_uniformity': 'uniformity_score',
                            'noise_frequency': 'frequency_score',
                            'frequency_analysis': 'dct_score',
                            'geometric_consistency': 'geometric_score',
                            'lighting_analysis': 'lighting_score',
                            'texture_analysis': 'texture_score',
                            'color_analysis': 'color_score',
                            'compression_analysis': 'compression_score'
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
        """Calculate Local Binary Pattern for texture analysis."""
        h, w = gray_img.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = gray_img[i, j]
                binary_val = 0
                
                for n in range(neighbors):
                    angle = 2 * np.pi * n / neighbors
                    ni = i + int(radius * np.sin(angle))
                    nj = j + int(radius * np.cos(angle))
                    
                    if 0 <= ni < h and 0 <= nj < w:
                        neighbor = gray_img[ni, nj]
                    else:
                        neighbor = 0
                    
                    if neighbor >= center:
                        binary_val |= (1 << n)
                
                lbp[i, j] = binary_val
        
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