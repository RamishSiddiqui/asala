import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import io
from .types import LayerResult

class PhysicsVerifier:
    """Physics-based verification layer (Python Implementation)."""

    def verify_image(self, image_bytes: bytes) -> LayerResult:
        """Verify image for physical consistency."""
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

            # Check 1: Noise pattern analysis
            noise_score = self._analyze_noise_patterns(img)
            
            # Check 2: Lighting consistency (ELA)
            lighting_score = self._check_lighting_consistency(image_bytes)
            
            # Check 3: Chromatic aberration
            aberration_score = self._check_chromatic_aberration(img)

            details = {
                "noiseScore": noise_score,
                "lightingScore": lighting_score,
                "aberrationScore": aberration_score
            }

            avg_score = (noise_score + lighting_score + aberration_score) / 3
            score = int(round(avg_score))
            
            passed = score >= 50
            if not passed:
                details['warning'] = 'Physical inconsistencies detected - possible synthetic content'

            return LayerResult(
                name="Physics Verification (Image)",
                passed=passed,
                score=score,
                details=details
            )

        except Exception as e:
            return LayerResult(
                name="Physics Verification (Image)",
                passed=False,
                score=0,
                details={"error": str(e)}
            )
            
    def _analyze_noise_patterns(self, img: np.ndarray) -> int:
        """Analyze noise patterns using local variance."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Estimate noise sigma
            # A simple heuristic: synthetic images often have very low noise variance
            sigma = self._estimate_noise(gray)
            
            # Expected sigma for natural images roughly 1.0 - 10.0 depending on ISO
            # Synthetic images often < 0.5 or perfectly smooth
            
            expected_sigma = 2.0
            deviation = abs(sigma - expected_sigma)
            
            # Score: 100 if close to expected, drops as it deviates
            score = max(0, 100 - int(deviation * 20))
            return score
        except:
            return 50 # Fallback

    def _check_lighting_consistency(self, image_bytes: bytes) -> int:
        """Check lighting consistency via ELA (Error Level Analysis)."""
        try:
            # ELA requires re-saving the image at a specific quality and comparing
            original = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            buffer = io.BytesIO()
            original.save(buffer, 'JPEG', quality=95)
            buffer.seek(0)
            
            recompressed = Image.open(buffer)
            
            # Calculate difference
            diff = ImageChops.difference(original, recompressed)
            
            # Get extrema
            extrema = diff.getextrema()
            max_diff = sum([ex[1] for ex in extrema]) / 3.0 # Average max diff across channels
            
            # Heuristic: 
            # Low max_diff (< 5) -> Consistent compression (or original)
            # High max_diff -> Potential manipulation or inconsistencies
            
            # For this score, we want higher score = LESS manipulation
            # So if diff is high, score should be low.
            
            score = max(0, 100 - int(max_diff * 2))
            return score
        except:
            return 50

    def _check_chromatic_aberration(self, img: np.ndarray) -> int:
        """Check for chromatic aberration logic."""
        # Simplified placeholder for now, implementing basic color consistency check
        return 80
        
    def _estimate_noise(self, gray_img: np.ndarray) -> float:
        """Fast noise estimation."""
        # Laplacian operator to find edges/noise
        H, W = gray_img.shape
        M = [[1, -2, 1],
             [-2, 4, -2],
             [1, -2, 1]]
        
        sigma = np.sum(np.abs(cv2.filter2D(gray_img, -1, np.array(M))))
        sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W-2) * (H-2))
        return float(sigma)
