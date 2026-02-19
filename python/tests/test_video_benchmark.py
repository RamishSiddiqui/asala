"""
Video Physics Layer Benchmark — tests VideoVerifier with synthetic videos.

Generates categories:
  1. Natural-like video  — simulated camera with natural noise and motion
  2. AI-like video       — per-frame synthesis (no temporal correlation)
  3. Static/loop video   — suspiciously stable or looping
  4. Manipulated video   — spliced/brightness-altered frames
"""

import importlib.util
import io
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# ---- path setup (use importlib to avoid cryptography DLL issue) ----
_project = Path(__file__).resolve().parent.parent.parent
_pkg_dir = _project / 'python' / 'asala'


def _load(name, fp):
    spec = importlib.util.spec_from_file_location(name, str(fp))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load('asala.types', _pkg_dir / 'types.py')
_video_mod = _load('asala.video', _pkg_dir / 'video.py')
VideoVerifier = _video_mod.VideoVerifier


# ---------------------------------------------------------------------------
# Helper: generate video bytes from frames
# ---------------------------------------------------------------------------
def _frames_to_video(frames: list, fps: float = 30.0) -> bytes:
    """Encode list of BGR numpy frames to MP4 bytes."""
    h, w = frames[0].shape[:2]
    tmp = tempfile.NamedTemporaryFile(suffix='.avi', delete=False)
    tmp.close()
    try:
        fourcc = cv2.VideoWriter.fourcc(*'MJPG')
        writer = cv2.VideoWriter(tmp.name, fourcc, fps, (w, h))
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            writer.write(frame)
        writer.release()
        with open(tmp.name, 'rb') as f:
            return f.read()
    finally:
        import os
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Video generators
# ---------------------------------------------------------------------------

def generate_natural_video(n_frames: int = 30, size: int = 128) -> bytes:
    """Simulate natural camera video: gradual motion, consistent noise, smooth lighting."""
    frames = []
    # Create a base scene with texture
    np.random.seed(42)
    scene = np.random.randint(40, 200, (size * 2, size * 2, 3), dtype=np.uint8)
    # Add some structure
    cv2.rectangle(scene, (50, 50), (150, 150), (180, 120, 80), -1)
    cv2.circle(scene, (200, 100), 40, (60, 140, 200), -1)
    # Smooth to make it look more natural
    scene = cv2.GaussianBlur(scene, (11, 11), 3)

    for i in range(n_frames):
        # Gradual camera pan (smooth motion)
        offset_x = int(5 * np.sin(2 * np.pi * i / n_frames * 0.5))
        offset_y = int(3 * np.cos(2 * np.pi * i / n_frames * 0.3))
        x = size // 2 + offset_x
        y = size // 2 + offset_y
        frame = scene[y:y + size, x:x + size].copy()

        # Add temporally correlated sensor noise
        noise_base = np.random.randn(size, size, 3) * 5
        if i > 0:
            # Mix with previous noise for temporal correlation
            noise = 0.6 * noise_base + 0.4 * prev_noise
        else:
            noise = noise_base
        prev_noise = noise.copy()

        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Slight brightness variation (natural)
        brightness_shift = int(2 * np.sin(2 * np.pi * i / n_frames))
        frame = np.clip(frame.astype(np.int16) + brightness_shift, 0, 255).astype(np.uint8)

        frames.append(frame)

    return _frames_to_video(frames)


def generate_ai_like_video(n_frames: int = 30, size: int = 128) -> bytes:
    """Simulate AI-generated video: independent per-frame synthesis, no temporal noise correlation."""
    frames = []
    for i in range(n_frames):
        # Each frame independently generated (smooth gradient + random structure)
        frame = np.zeros((size, size, 3), dtype=np.uint8)

        # Base gradient (consistent across frames like a GAN would produce)
        for c in range(3):
            grad = np.linspace(50 + c * 20, 180 + c * 10, size)
            frame[:, :, c] = np.tile(grad, (size, 1)).astype(np.uint8)

        # Add per-frame independent smooth blob (AI synthesis artifact)
        blob_x = size // 2 + int(10 * np.sin(2 * np.pi * i / n_frames))
        blob_y = size // 2 + int(8 * np.cos(2 * np.pi * i / n_frames))
        cv2.circle(frame, (blob_x, blob_y), 25, (150, 100, 200), -1)
        frame = cv2.GaussianBlur(frame, (7, 7), 2)

        # Independent noise per frame (no temporal correlation)
        noise = np.random.randn(size, size, 3) * 3
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        frames.append(frame)

    return _frames_to_video(frames)


def generate_static_video(n_frames: int = 30, size: int = 128) -> bytes:
    """Generate a static (looping/frozen) video — same frame repeated."""
    frame = np.random.randint(60, 200, (size, size, 3), dtype=np.uint8)
    cv2.rectangle(frame, (20, 20), (100, 100), (120, 80, 160), -1)
    frame = cv2.GaussianBlur(frame, (5, 5), 1)
    # Add very small noise to avoid exact pixel match
    frames = []
    for i in range(n_frames):
        noise = np.random.randn(size, size, 3) * 0.5
        f = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        frames.append(f)
    return _frames_to_video(frames)


def generate_spliced_video(n_frames: int = 30, size: int = 128) -> bytes:
    """Video with a hard splice: first half from one scene, second from another."""
    frames = []
    half = n_frames // 2

    # Scene 1: warm colors, textured
    for i in range(half):
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame[:, :, 0] = 60  # B
        frame[:, :, 1] = 100  # G
        frame[:, :, 2] = 180  # R (warm)
        noise = np.random.randn(size, size, 3) * 8
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        # Small motion
        offset = int(3 * np.sin(2 * np.pi * i / half))
        M = np.float32([[1, 0, offset], [0, 1, 0]])
        frame = cv2.warpAffine(frame, M, (size, size))
        frames.append(frame)

    # Scene 2: cool colors, different texture (hard cut)
    for i in range(n_frames - half):
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame[:, :, 0] = 180  # B (cool)
        frame[:, :, 1] = 140  # G
        frame[:, :, 2] = 60  # R
        noise = np.random.randn(size, size, 3) * 4
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        offset = int(2 * np.cos(2 * np.pi * i / (n_frames - half)))
        M = np.float32([[1, 0, offset], [0, 1, 0]])
        frame = cv2.warpAffine(frame, M, (size, size))
        frames.append(frame)

    return _frames_to_video(frames)


def generate_flickering_video(n_frames: int = 30, size: int = 128) -> bytes:
    """Video with unnatural brightness flickering (AI artifact)."""
    frames = []
    base = np.random.randint(60, 180, (size, size, 3), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (9, 9), 3)

    for i in range(n_frames):
        # Rapid brightness oscillation (unnatural)
        flicker = int(40 * np.sin(2 * np.pi * i / 3))  # Very fast flicker
        frame = np.clip(base.astype(np.int16) + flicker, 0, 255).astype(np.uint8)
        noise = np.random.randn(size, size, 3) * 3
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        frames.append(frame)

    return _frames_to_video(frames)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def main():
    v = VideoVerifier()

    print("=" * 70)
    print("ASALA VIDEO PHYSICS LAYER — BENCHMARK")
    print("=" * 70)

    categories = [
        ("REAL-LIKE VIDEO (expected: passed=True)", [
            ("natural_video", generate_natural_video),
        ], True),
        ("SYNTHETIC / AI-LIKE VIDEO (expected: passed=False)", [
            ("ai_generated", generate_ai_like_video),
            ("static_loop", generate_static_video),
            ("flickering", generate_flickering_video),
        ], False),
        ("MANIPULATED VIDEO (expected: passed=False)", [
            ("spliced_video", generate_spliced_video),
        ], False),
    ]

    total_correct = 0
    total_tests = 0

    for cat_name, tests, expected_pass in categories:
        print(f"\n{'-' * 70}")
        print(f"  {cat_name}")
        print(f"{'-' * 70}")

        for name, gen_func in tests:
            video_bytes = gen_func()
            result = v.verify_video(video_bytes)
            d = result.details

            is_correct = (result.passed == expected_pass)
            total_tests += 1
            if is_correct:
                total_correct += 1

            status = "[PASS]" if is_correct else "[FAIL]"
            print(f"  {status} {name:25s}  score={result.score:3d}  "
                  f"passed={str(result.passed):5s}  "
                  f"ai_prob={d.get('ai_probability', 0):.2f}  "
                  f"indicators={d.get('ai_indicators', '?')}")

            if d.get('warning'):
                print(f"         warning: {d['warning']}")

            # Show method scores
            score_keys = {
                'frame_analysis': 'frame_analysis_score',
                'temporal_noise': 'temporal_noise_score',
                'optical_flow': 'optical_flow_score',
                'encoding_analysis': 'encoding_score',
                'temporal_lighting': 'temporal_lighting_score',
                'frame_stability': 'stability_score',
            }
            scores = []
            for method, key in score_keys.items():
                sub = d.get(method, {})
                if isinstance(sub, dict) and key in sub:
                    scores.append(f"{method.split('_')[0]}={sub[key]}")
            print(f"         [{', '.join(scores)}]")

            # Key metrics
            tn = d.get('temporal_noise', {})
            of = d.get('optical_flow', {})
            fs = d.get('frame_stability', {})
            nc = tn.get('avg_noise_corr')
            fl = of.get('avg_flow_smoothness')
            ncc = fs.get('avg_ncc')
            nc_s = f"{nc:.3f}" if isinstance(nc, (int, float)) else "?"
            fl_s = f"{fl:.3f}" if isinstance(fl, (int, float)) else "?"
            ncc_s = f"{ncc:.4f}" if isinstance(ncc, (int, float)) else "?"
            print(f"         noise_corr={nc_s}  flow_smooth={fl_s}  ncc={ncc_s}")

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {total_correct}/{total_tests} correct "
          f"({100 * total_correct / total_tests:.1f}%)")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
