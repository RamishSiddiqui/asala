import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
import io
from .types import LayerResult

class PhysicsVerifier:
    """Enhanced physics-based verification layer for detecting AI-generated images."""

    def verify_image(self, image_bytes: bytes) -> LayerResult:
        """Verify image for physical consistency using multiple mathematical approaches."""
        try:
            # Convert bytes to numpy array for OpenCV
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return LayerResult(
                    name="Physics Verification (Image)",
                    passed=False,
                    score=0,
                    details={"error": "Invalid image data"}
                )

            # Enhanced Analysis Suite
            results = {}
            
            # 1. Advanced Noise Pattern Analysis
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

            # Calculate weighted score based on analysis
            final_score = self._calculate_composite_score(results)
            
            # Determine if likely AI-generated
            # More nuanced threshold based on multiple factors
            ai_indicators = 0
            total_indicators = 0
            
            # Check various indicators
            noise_score = results.get('noise_uniformity', {}).get('uniformity_score', 50)
            dct_score = results.get('frequency_analysis', {}).get('dct_score', 50)
            geo_score = results.get('geometric_consistency', {}).get('geometric_score', 50)
            lighting_score = results.get('lighting_analysis', {}).get('lighting_score', 50)
            texture_score = results.get('texture_analysis', {}).get('texture_score', 50)
            
            if noise_score > 85:  # Very uniform noise (AI-like)
                ai_indicators += 1
            total_indicators += 1
            
            if dct_score < 25:  # Very low DCT (AI-like)
                ai_indicators += 1
            total_indicators += 1
            
            if geo_score < 30:  # Very low geometric consistency (AI-like)
                ai_indicators += 1
            total_indicators += 1
            
            if lighting_score < 25:  # Very poor lighting (AI-like)
                ai_indicators += 1
            total_indicators += 1
            
            if texture_score < 30:  # Very low texture complexity (AI-like)
                ai_indicators += 1
            total_indicators += 1
            
            # Calculate AI probability
            ai_probability = ai_indicators / total_indicators if total_indicators > 0 else 0.5
            
            # Adjust final score based on AI indicators
            if ai_probability > 0.6:  # Strong AI indicators
                final_score = int(final_score * 0.7)  # Reduce score significantly
                passed = False
                results['warning'] = 'Strong physical inconsistencies detected - likely AI-generated'
            elif ai_probability > 0.4:  # Moderate AI indicators
                final_score = int(final_score * 0.85)  # Reduce score moderately
                passed = final_score >= 45
                results['warning'] = 'Some physical inconsistencies detected - possibly AI-generated'
            else:  # Low AI indicators
                passed = final_score >= 50  # Standard threshold
                if not passed:
                    results['warning'] = 'Physical inconsistencies detected - review recommended'
            
            return LayerResult(
                name="Physics Verification (Image)",
                passed=passed,
                score=final_score,
                details=results
            )

        except Exception as e:
            return LayerResult(
                name="Physics Verification (Image)",
                passed=False,
                score=0,
                details={"error": str(e)}
            )

    def _analyze_noise_uniformity(self, img: np.ndarray) -> dict:
        """Analyze noise uniformity - AI images often have unnaturally uniform noise."""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Divide image into regions
            h, w = gray.shape
            regions = [
                gray[0:h//2, 0:w//2],
                gray[0:h//2, w//2:],
                gray[h//2:, 0:w//2],
                gray[h//2:, w//2:]
            ]
            
            # Calculate noise variance in each region
            variances = []
            for region in regions:
                # Use Laplacian variance for noise estimation
                laplacian = cv2.Laplacian(region, cv2.CV_64F)
                variance = laplacian.var()
                variances.append(variance)
            
            # AI images often have very similar variance across regions
            variance_std = np.std(variances)
            variance_mean = np.mean(variances)
            
            # Lower std = more uniform = more likely AI
            uniformity_score = max(0, 100 - int(variance_std / variance_mean * 500))
            
            return {
                'variance_mean': float(variance_mean),
                'variance_std': float(variance_std),
                'uniformity_score': uniformity_score if uniformity_score > 0 else 50  # Ensure valid score
            }
        except:
            return {'uniformity_score': 50}

    def _analyze_noise_frequency(self, img: np.ndarray) -> dict:
        """Analyze noise frequency characteristics using FFT."""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply FFT to analyze frequency components
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            
            # Analyze high-frequency content
            h, w = magnitude.shape
            center_h, center_w = h // 2, w // 2
            
            # Define high-frequency region (corners)
            mask = np.zeros((h, w), np.uint8)
            mask_size = min(h, w) // 8
            mask[0:center_h-mask_size, 0:center_w-mask_size] = 1
            mask[0:center_h-mask_size, center_w+mask_size:] = 1
            mask[center_h+mask_size:, 0:center_w-mask_size] = 1
            mask[center_h+mask_size:, center_w+mask_size:] = 1
            
            high_freq_energy = np.sum(magnitude * mask)
            total_energy = np.sum(magnitude)
            high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
            
            # AI images often have different high-frequency characteristics
            freq_score = int(high_freq_ratio * 200)
            freq_score = min(100, max(0, freq_score))
            
            return {
                'high_frequency_ratio': float(high_freq_ratio),
                'frequency_score': freq_score if freq_score > 0 else 50  # Ensure valid score
            }
        except:
            return {'frequency_score': 50}

    def _analyze_frequency_domain(self, img: np.ndarray) -> dict:
        """Analyze DCT coefficients - AI images have characteristic DCT patterns."""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply DCT
            dct = cv2.dct(gray.astype(np.float32))
            
            # Analyze coefficient distribution
            dct_abs = np.abs(dct)
            
            # AI images often have:
            # 1. Fewer non-zero high-frequency coefficients
            # 2. Different coefficient distribution patterns
            
            # Count non-zero coefficients by frequency
            h, w = dct_abs.shape
            high_freq_mask = np.zeros((h, w), np.bool_)
            high_freq_mask[h//4:, w//4:] = True  # Upper-right = high frequencies
            
            high_freq_count = np.sum(dct_abs[high_freq_mask] > 10)
            total_count = np.sum(dct_abs > 10)
            high_freq_ratio = high_freq_count / total_count if total_count > 0 else 0
            
            # Analyze coefficient entropy
            dct_flat = dct_abs.flatten()
            dct_entropy = -np.sum((dct_flat + 1e-10) * np.log2(dct_flat + 1e-10))
            
            # Score based on high-frequency content and entropy
            dct_score = int((high_freq_ratio * 150) + (dct_entropy / 1000))
            dct_score = min(100, max(0, dct_score))
            
            return {
                'high_frequency_ratio': float(high_freq_ratio),
                'dct_entropy': float(dct_entropy),
                'dct_score': dct_score if dct_score > 0 else 50  # Ensure valid score
            }
        except:
            return {'dct_score': 50}

    def _analyze_geometric_consistency(self, img: np.ndarray) -> dict:
        """Analyze geometric consistency - lines, perspectives, shapes."""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
            
            # Analyze line consistency
            if lines is not None:
                # Calculate angle distribution
                angles = [np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi 
                         for x1, y1, x2, y2 in lines]
                
                # AI images often have less consistent geometric patterns
                angle_std = np.std(angles) if angles else 0
                
                # Count straight vs curved lines
                straight_lines = len([l for l in lines if abs(l[3] - l[1]) > abs(l[2] - l[0])])
                line_ratio = straight_lines / len(lines) if lines else 0.5
            else:
                angle_std = 45  # High variance = inconsistent
                line_ratio = 0.5
            
            # Analyze perspective consistency
            # Detect corners and analyze their distribution
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
            
            if corners is not None and len(corners) > 10:
                # Analyze corner distribution
                corner_positions = np.array(corners).reshape(-1, 2)
                corner_density = self._calculate_point_density(corner_positions, gray.shape)
            else:
                corner_density = 0.5
            
            # Calculate geometric consistency score
            geo_score = int(((100 - angle_std) * 0.3 + line_ratio * 40 + corner_density * 30))
            geo_score = min(100, max(0, geo_score))
            
            return {
                'line_count': len(lines) if lines is not None else 0,
                'angle_std': float(angle_std),
                'line_ratio': float(line_ratio),
                'corner_density': float(corner_density),
                'geometric_score': geo_score
            }
        except:
            return {'geometric_score': 50}

    def _analyze_lighting_consistency(self, img: np.ndarray) -> dict:
        """Enhanced lighting and shadow analysis."""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            # Analyze lighting consistency
            # Split into regions and check lighting variation
            h, w = img.shape[:2]
            regions = self._divide_image_into_grid(img, 3, 3)
            
            brightness_values = []
            contrast_values = []
            
            for region in regions:
                # Calculate brightness and contrast for each region
                gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray_region)
                contrast = np.std(gray_region)
                
                brightness_values.append(brightness)
                contrast_values.append(contrast)
            
            # AI images often have inconsistent lighting
            brightness_std = np.std(brightness_values)
            contrast_std = np.std(contrast_values)
            
            # Analyze shadow consistency
            # Look at L channel in LAB (lightness)
            l_channel = lab[:, :, 0]
            
            # Detect shadow regions (dark areas)
            shadow_mask = l_channel < np.percentile(l_channel, 25)
            shadow_coverage = np.sum(shadow_mask) / (h * w)
            
            # Analyze highlight consistency
            highlight_mask = l_channel > np.percentile(l_channel, 75)
            highlight_coverage = np.sum(highlight_mask) / (h * w)
            
            # Calculate lighting consistency score
            lighting_score = int(max(0, 100 - (brightness_std * 2) - (contrast_std * 2) - (shadow_coverage * 100)))
            lighting_score = min(100, max(0, lighting_score))
            
            return {
                'brightness_std': float(brightness_std),
                'contrast_std': float(contrast_std),
                'shadow_coverage': float(shadow_coverage),
                'highlight_coverage': float(highlight_coverage),
                'lighting_score': lighting_score
            }
        except:
            return {'lighting_score': 50}

    def _analyze_texture_patterns(self, img: np.ndarray) -> dict:
        """Analyze texture patterns using statistical methods."""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate Local Binary Patterns (LBP)
            lbp = self._calculate_lbp(gray)
            
            # Analyze texture statistics
            lbp_hist = np.histogram(lbp.flatten(), bins=256)[0]
            lbp_hist = lbp_hist / np.sum(lbp_hist)  # Normalize
            
            # Calculate texture entropy
            texture_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))
            
            # Calculate Gray-Level Co-occurrence Matrix (GLCM) features
            glcm = cv2.calcHist([gray], [0], None, [256], [0, 256])
            glcm = glcm / np.sum(glcm)
            
            # Calculate contrast and homogeneity
            contrast = 0
            homogeneity = 0
            for i in range(256):
                for j in range(256):
                    contrast += glcm[i, j] * (i - j) ** 2
                    if i != j:
                        homogeneity += glcm[i, j] / (1 + abs(i - j))
            
            # AI images often have different texture characteristics
            texture_score = int((texture_entropy * 10) + (homogeneity * 50) + (100 - min(100, contrast / 1000)))
            texture_score = min(100, max(0, texture_score))
            
            return {
                'texture_entropy': float(texture_entropy),
                'glcm_contrast': float(contrast),
                'glcm_homogeneity': float(homogeneity),
                'texture_score': texture_score
            }
        except:
            return {'texture_score': 50}

    def _analyze_color_distribution(self, img: np.ndarray) -> dict:
        """Analyze color distribution and consistency."""
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            # Analyze HSV distributions
            h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
            
            # Normalize histograms
            h_hist = h_hist / np.sum(h_hist)
            s_hist = s_hist / np.sum(s_hist)
            v_hist = v_hist / np.sum(v_hist)
            
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
                color_variations.append([mean_hue, mean_sat])
            
            color_variations = np.array(color_variations)
            hue_std = np.std(color_variations[:, 0])
            sat_std = np.std(color_variations[:, 1])
            
            # AI images often have different color characteristics
            color_score = int(((h_entropy + s_entropy + v_entropy) * 10) - (hue_std * 5) - (sat_std * 10))
            color_score = min(100, max(0, color_score))
            
            return {
                'hue_entropy': float(h_entropy),
                'saturation_entropy': float(s_entropy),
                'value_entropy': float(v_entropy),
                'hue_std': float(hue_std),
                'saturation_std': float(sat_std),
                'color_score': color_score
            }
        except:
            return {'color_score': 50}

    def _analyze_compression_artifacts(self, image_bytes: bytes) -> dict:
        """Analyze compression artifacts and ELA patterns."""
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
            
            # Analyze ELA patterns
            ela_variance = np.var(ela_scores)
            ela_mean = np.mean(ela_scores)
            
            # AI images often show different ELA patterns
            compression_score = int(max(0, 100 - (ela_variance * 5) - (ela_mean * 0.5)))
            compression_score = min(100, max(0, compression_score))
            
            return {
                'ela_scores': [float(x) for x in ela_scores],
                'ela_variance': float(ela_variance),
                'ela_mean': float(ela_mean),
                'compression_score': compression_score
            }
        except:
            return {'compression_score': 50}

    def _calculate_composite_score(self, results: dict) -> int:
        """Calculate final composite score from all analyses."""
        try:
            # Weight different analyses based on their discriminative power
            weights = {
                'noise_uniformity': 0.15,
                'noise_frequency': 0.10,
                'frequency_analysis': 0.15,
                'geometric_consistency': 0.15,
                'lighting_analysis': 0.15,
                'texture_analysis': 0.15,
                'color_analysis': 0.10,
                'compression_analysis': 0.05
            }
            
            total_score = 0
            total_weight = 0
            
            for analysis_type, weight in weights.items():
                if analysis_type in results:
                    if isinstance(results[analysis_type], dict):
                        # Extract score from nested dictionary
                        score_key = f"{analysis_type}_score"
                        if score_key in results[analysis_type]:
                            score = results[analysis_type][score_key]
                        else:
                            # Look for any score key
                            score_keys = [k for k in results[analysis_type].keys() if 'score' in k]
                            if score_keys:
                                score = results[analysis_type][score_keys[0]]
                            else:
                                continue
                    else:
                        score = results[analysis_type]
                    
                    total_score += score * weight
                    total_weight += weight
            
            if total_weight > 0:
                final_score = int(total_score / total_weight)
            else:
                final_score = 50
            
            return min(100, max(0, final_score))
        except:
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

    def _calculate_lbp(self, gray_img: np.ndarray, radius=1, neighbors=8) -> np.ndarray:
        """Calculate Local Binary Pattern."""
        h, w = gray_img.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = gray_img[i, j]
                binary_string = ""
                
                for n in range(neighbors):
                    # Calculate neighbor position
                    if n < neighbors - 1:
                        angle = 2 * np.pi * n / neighbors
                    else:
                        angle = 0
                    
                    ni = i + int(radius * np.sin(angle))
                    nj = j + int(radius * np.cos(angle))
                    
                    if 0 <= ni < h and 0 <= nj < w:
                        neighbor = gray_img[ni, nj]
                    else:
                        neighbor = 0
                    
                    binary_string += "1" if neighbor >= center else "0"
                
                lbp[i, j] = int(binary_string, 2)
        
        return lbp

    def _calculate_point_density(self, points: np.ndarray, shape: tuple) -> float:
        """Calculate point density in image."""
        if len(points) == 0:
            return 0
        
        h, w = shape
        # Divide image into grid and count points per cell
        grid_size = 10
        grid_h, grid_w = h // grid_size, w // grid_size
        
        point_counts = np.zeros((grid_h, grid_w))
        
        for point in points:
            grid_i = min(int(point[1] / grid_size), grid_h - 1)
            grid_j = min(int(point[0] / grid_size), grid_w - 1)
            point_counts[grid_i, grid_j] += 1
        
        # Calculate density variance
        density_variance = np.var(point_counts[point_counts > 0]) if np.any(point_counts > 0) else 0
        
        # Lower variance = more consistent density = more natural
        density_score = max(0, 100 - density_variance * 10)
        
        return density_score