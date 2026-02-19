"""
Benchmark test for the Physics Verification Layer (Layer 2).

Tests the PhysicsVerifier against:
1. Real camera images (from test-data/original/)
2. Programmatically generated synthetic images (simulating AI-generated content)
3. Manipulated images (real images with edits applied)

Goal: Verify that the physics layer can reliably distinguish
real images from synthetic/manipulated ones.
"""

import sys
import os
import io
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pathlib import Path

# Bypass asala.__init__ (which imports crypto -> cryptography Rust bindings
# that fail on Python 3.9.0 with DLL load errors). Load types and physics directly.
import importlib.util

_pkg_dir = Path(__file__).resolve().parents[1] / "asala"

def _load_module(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # register so relative imports within it resolve
    spec.loader.exec_module(mod)
    return mod

# Load types first (physics.py imports from .types)
_types_mod = _load_module("asala.types", str(_pkg_dir / "types.py"))
LayerResult = _types_mod.LayerResult

# Now load physics
_physics_mod = _load_module("asala.physics", str(_pkg_dir / "physics.py"))
PhysicsVerifier = _physics_mod.PhysicsVerifier


# ---------------------------------------------------------------------------
# Helpers: generate synthetic / manipulated test images
# ---------------------------------------------------------------------------

def _encode_jpg(img_array: np.ndarray, quality: int = 95) -> bytes:
    """Encode a numpy BGR array to JPEG bytes."""
    ok, buf = cv2.imencode('.jpg', img_array, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok, "JPEG encoding failed"
    return buf.tobytes()


def generate_smooth_gradient(width: int = 512, height: int = 512) -> bytes:
    """Pure smooth gradient – no noise at all. AI-like."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        val = int(255 * y / height)
        img[y, :] = [val, val, val]
    return _encode_jpg(img)


def generate_uniform_noise(width: int = 512, height: int = 512) -> bytes:
    """Uniform random noise – no structure. Very un-natural."""
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return _encode_jpg(img)


def generate_perlin_like_synthetic(width: int = 512, height: int = 512) -> bytes:
    """
    Simulates a smooth AI-generated image:
    blurred noise + smooth color blending. Lacks sensor noise,
    chromatic aberration, and natural frequency falloff.
    """
    # Start with smooth random blobs
    base = np.random.rand(height // 8, width // 8, 3).astype(np.float32)
    base = cv2.resize(base, (width, height), interpolation=cv2.INTER_CUBIC)
    base = (base * 255).astype(np.uint8)
    # Heavy gaussian blur to make it ultra smooth
    base = cv2.GaussianBlur(base, (31, 31), 10)
    return _encode_jpg(base)


def generate_gan_like_face(width: int = 512, height: int = 512) -> bytes:
    """
    Simulates a GAN-generated face: symmetric, smooth skin, perfect gradients,
    unnaturally uniform noise, strong edges but no lens artifacts.
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 200  # light skin base

    # Face oval
    center = (width // 2, height // 2)
    axes = (width // 3, height // 3)
    cv2.ellipse(img, center, axes, 0, 0, 360, (180, 160, 150), -1)

    # Smooth it heavily (GAN-like smoothness)
    img = cv2.GaussianBlur(img, (51, 51), 15)

    # Add perfectly uniform noise (unnatural – real sensors have spatially varying noise)
    noise = np.random.normal(0, 3, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Eyes (simple circles)
    eye_y = height // 2 - 20
    cv2.circle(img, (width // 2 - 50, eye_y), 15, (80, 60, 50), -1)
    cv2.circle(img, (width // 2 + 50, eye_y), 15, (80, 60, 50), -1)

    return _encode_jpg(img)


def generate_diffusion_like_landscape(width: int = 512, height: int = 512) -> bytes:
    """
    Simulates a diffusion-model landscape: smooth sky gradient,
    textured ground from upscaled noise, sharp horizon line.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky gradient (very smooth)
    for y in range(height // 2):
        t = y / (height // 2)
        r = int(100 + 120 * t)
        g = int(150 + 80 * t)
        b = int(220 - 20 * t)
        img[y, :] = [b, g, r]  # BGR

    # Ground: upscaled noise (lacks real texture detail)
    ground_h = height - height // 2
    ground = np.random.randint(30, 120, (ground_h // 8, width // 8, 3), dtype=np.uint8)
    ground = cv2.resize(ground, (width, ground_h), interpolation=cv2.INTER_CUBIC)
    ground[:, :, 1] = np.clip(ground[:, :, 1].astype(int) + 40, 0, 255)  # green tint
    img[height // 2:, :] = ground

    # Smooth everything
    img = cv2.GaussianBlur(img, (11, 11), 4)

    return _encode_jpg(img)


def manipulate_splice(real_img_bytes: bytes) -> bytes:
    """
    Splice manipulation: copy-paste a region from one part of the image
    to another, creating compression-level inconsistencies.
    """
    nparr = np.frombuffer(real_img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return real_img_bytes
    h, w = img.shape[:2]

    # Copy a rectangular patch from top-left to bottom-right
    patch_h, patch_w = h // 4, w // 4
    patch = img[0:patch_h, 0:patch_w].copy()

    # Re-compress the patch at low quality (simulates external source)
    _, buf = cv2.imencode('.jpg', patch, [cv2.IMWRITE_JPEG_QUALITY, 30])
    patch_low = cv2.imdecode(np.frombuffer(buf.tobytes(), np.uint8), cv2.IMREAD_COLOR)
    patch_low = cv2.resize(patch_low, (patch_w, patch_h))

    img[h - patch_h:h, w - patch_w:w] = patch_low
    return _encode_jpg(img)


def manipulate_brightness_region(real_img_bytes: bytes) -> bytes:
    """
    Brighten a region unnaturally – simulates local editing.
    """
    nparr = np.frombuffer(real_img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return real_img_bytes
    h, w = img.shape[:2]

    # Brighten center square
    cy, cx = h // 2, w // 2
    s = min(h, w) // 4
    roi = img[cy - s:cy + s, cx - s:cx + s].astype(np.float32)
    roi = np.clip(roi * 1.6 + 30, 0, 255).astype(np.uint8)
    img[cy - s:cy + s, cx - s:cx + s] = roi

    return _encode_jpg(img)


# ---------------------------------------------------------------------------
# The benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    verifier = PhysicsVerifier()
    results = {}

    print("=" * 70)
    print("ASALA PHYSICS LAYER (Layer 2) – BENCHMARK")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # 1. Real images from test-data/original/
    # -----------------------------------------------------------------------
    test_data_dir = Path(__file__).resolve().parents[2] / "test-data" / "original"
    real_images = {}
    if test_data_dir.exists():
        for fpath in sorted(test_data_dir.iterdir()):
            if fpath.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                real_images[fpath.name] = fpath.read_bytes()
    else:
        print(f"\n[WARN] test-data/original/ not found at {test_data_dir}")

    print(f"\nFound {len(real_images)} real test images")

    print("\n" + "-" * 70)
    print("CATEGORY 1: REAL CAMERA IMAGES  (expected: passed=True, score>=50)")
    print("-" * 70)
    real_results = []
    for name, img_bytes in real_images.items():
        result = verifier.verify_image(img_bytes)
        real_results.append(result)
        status = "PASS" if result.passed else "FAIL"
        warning = result.details.get('warning', '')
        print(f"  [{status}] {name:40s}  score={result.score:3d}  ai_prob={result.details.get('ai_probability', '?')}")
        if warning:
            print(f"         warning: {warning}")
        # Print sub-scores
        for key in ['noise_uniformity', 'noise_frequency', 'frequency_analysis',
                     'geometric_consistency', 'lighting_analysis', 'texture_analysis',
                     'color_analysis', 'compression_analysis']:
            sub = result.details.get(key, {})
            if isinstance(sub, dict):
                score_keys = [k for k in sub if k.endswith('_score')]
                scores_str = ", ".join(f"{k}={sub[k]}" for k in score_keys)
                if scores_str:
                    print(f"           {key}: {scores_str}")
    results['real'] = real_results

    # -----------------------------------------------------------------------
    # 2. Synthetic / AI-like images
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("CATEGORY 2: SYNTHETIC / AI-LIKE IMAGES  (expected: passed=False, score<50)")
    print("-" * 70)

    synthetic_tests = {
        "smooth_gradient":           generate_smooth_gradient(),
        "uniform_noise":             generate_uniform_noise(),
        "perlin_like_synthetic":     generate_perlin_like_synthetic(),
        "gan_like_face":             generate_gan_like_face(),
        "diffusion_like_landscape":  generate_diffusion_like_landscape(),
    }

    synthetic_results = []
    for name, img_bytes in synthetic_tests.items():
        result = verifier.verify_image(img_bytes)
        synthetic_results.append(result)
        status = "PASS" if not result.passed else "FAIL"  # We WANT passed=False here
        warning = result.details.get('warning', '')
        print(f"  [{status}] {name:40s}  score={result.score:3d}  ai_prob={result.details.get('ai_probability', '?')}")
        if warning:
            print(f"         warning: {warning}")
        for key in ['noise_uniformity', 'noise_frequency', 'frequency_analysis',
                     'geometric_consistency', 'lighting_analysis', 'texture_analysis',
                     'color_analysis', 'compression_analysis']:
            sub = result.details.get(key, {})
            if isinstance(sub, dict):
                score_keys = [k for k in sub if k.endswith('_score')]
                scores_str = ", ".join(f"{k}={sub[k]}" for k in score_keys)
                if scores_str:
                    print(f"           {key}: {scores_str}")
    results['synthetic'] = synthetic_results

    # -----------------------------------------------------------------------
    # 3. Manipulated images (real + edits)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("CATEGORY 3: MANIPULATED IMAGES  (expected: passed=False or low score)")
    print("-" * 70)

    manipulated_results = []
    if real_images:
        # Pick the first JPEG for manipulation tests
        first_name = next((n for n in real_images if n.lower().endswith('.jpg')), None)
        if first_name is None:
            # Fall back to any available image
            first_name = next(iter(real_images))
        first_bytes = real_images[first_name]

        manip_tests = {
            f"splice({first_name})":     manipulate_splice(first_bytes),
            f"brightness({first_name})": manipulate_brightness_region(first_bytes),
        }
        for name, img_bytes in manip_tests.items():
            result = verifier.verify_image(img_bytes)
            manipulated_results.append(result)
            status = "PASS" if not result.passed else "FAIL"
            warning = result.details.get('warning', '')
            print(f"  [{status}] {name:40s}  score={result.score:3d}  ai_prob={result.details.get('ai_probability', '?')}")
            if warning:
                print(f"         warning: {warning}")
            for key in ['noise_uniformity', 'noise_frequency', 'frequency_analysis',
                         'geometric_consistency', 'lighting_analysis', 'texture_analysis',
                         'color_analysis', 'compression_analysis']:
                sub = result.details.get(key, {})
                if isinstance(sub, dict):
                    score_keys = [k for k in sub if k.endswith('_score')]
                    scores_str = ", ".join(f"{k}={sub[k]}" for k in score_keys)
                    if scores_str:
                        print(f"           {key}: {scores_str}")
    results['manipulated'] = manipulated_results

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    real_correct = sum(1 for r in real_results if r.passed)
    real_total = len(real_results)
    synth_correct = sum(1 for r in synthetic_results if not r.passed)
    synth_total = len(synthetic_results)
    manip_correct = sum(1 for r in manipulated_results if not r.passed)
    manip_total = len(manipulated_results)

    total_correct = real_correct + synth_correct + manip_correct
    total_tests = real_total + synth_total + manip_total

    print(f"  Real images identified correctly:        {real_correct}/{real_total}")
    print(f"  Synthetic images identified correctly:   {synth_correct}/{synth_total}")
    print(f"  Manipulated images identified correctly: {manip_correct}/{manip_total}")
    print(f"  -----------------------------------------")
    print(f"  Overall accuracy:                        {total_correct}/{total_tests} ({100*total_correct/max(total_tests,1):.1f}%)")

    if real_results:
        real_scores = [r.score for r in real_results]
        print(f"\n  Real image scores:      min={min(real_scores)}  max={max(real_scores)}  avg={sum(real_scores)/len(real_scores):.1f}")
    if synthetic_results:
        synth_scores = [r.score for r in synthetic_results]
        print(f"  Synthetic image scores: min={min(synth_scores)}  max={max(synth_scores)}  avg={sum(synth_scores)/len(synth_scores):.1f}")
    if manipulated_results:
        manip_scores = [r.score for r in manipulated_results]
        print(f"  Manipulated img scores: min={min(manip_scores)}  max={max(manip_scores)}  avg={sum(manip_scores)/len(manip_scores):.1f}")

    # Score separation analysis
    if real_results and synthetic_results:
        real_min = min(r.score for r in real_results)
        synth_max = max(r.score for r in synthetic_results)
        gap = real_min - synth_max
        print(f"\n  Score gap (real_min - synthetic_max): {gap}")
        if gap > 0:
            print(f"  -> Clean separation: real images always score higher than synthetic")
        else:
            print(f"  -> OVERLAP: some synthetic images score as high as real ones!")
            print(f"     This means the physics layer cannot reliably distinguish them.")

    print("\n" + "=" * 70)
    return results


if __name__ == "__main__":
    run_benchmark()
