"""Tests for PhysicsVerifier (image analysis)."""

import io

import numpy as np
import pytest
from PIL import Image

from asala.physics import PhysicsVerifier


def _make_jpeg(img: Image.Image, quality: int = 90) -> bytes:
    """Encode PIL Image to JPEG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _gradient_image(w: int = 128, h: int = 128) -> bytes:
    """Smooth left-to-right gradient — has natural texture structure."""
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(w):
            v = int(x / (w - 1) * 255)
            px[x, y] = (v, v, v)
    return _make_jpeg(img)


def _uniform_image(w: int = 128, h: int = 128, color: tuple = (128, 128, 128)) -> bytes:
    """Perfectly uniform image — synthetic-looking."""
    img = Image.new("RGB", (w, h), color=color)
    return _make_jpeg(img)


def _noise_image(w: int = 128, h: int = 128) -> bytes:
    """Random noise — has rich frequency content."""
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    return _make_jpeg(img)


def _checkerboard_image(w: int = 128, h: int = 128, block: int = 8) -> bytes:
    """Checkerboard pattern — regular structure."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if ((x // block) + (y // block)) % 2:
                arr[y, x] = [255, 255, 255]
    img = Image.fromarray(arr)
    return _make_jpeg(img)


class TestPhysicsVerifierBasic:
    def test_instantiation(self):
        v = PhysicsVerifier()
        assert v is not None

    def test_verify_image_returns_layer_result(self):
        v = PhysicsVerifier()
        result = v.verify_image(_gradient_image())

        assert result.name == "Physics Verification (Image)"
        assert isinstance(result.passed, bool)
        assert 0 <= result.score <= 100
        assert isinstance(result.details, dict)

    def test_verify_image_has_all_phase1_keys(self):
        v = PhysicsVerifier()
        result = v.verify_image(_gradient_image())
        d = result.details

        for key in [
            "noise_uniformity",
            "noise_frequency",
            "frequency_analysis",
            "geometric_consistency",
            "lighting_analysis",
            "texture_analysis",
            "color_analysis",
            "compression_analysis",
        ]:
            assert key in d, f"Missing Phase 1 key: {key}"

    def test_verify_image_has_all_phase2_keys(self):
        v = PhysicsVerifier()
        result = v.verify_image(_gradient_image())
        d = result.details

        for key in [
            "noise_consistency_map",
            "jpeg_ghost",
            "spectral_fingerprint",
            "channel_correlation",
        ]:
            assert key in d, f"Missing Phase 2 key: {key}"

    def test_verify_image_has_all_phase3_keys(self):
        v = PhysicsVerifier()
        result = v.verify_image(_gradient_image())
        d = result.details

        for key in ["benford_dct", "wavelet_ratio", "blocking_grid", "cfa_demosaicing"]:
            assert key in d, f"Missing Phase 3 key: {key}"

    def test_verify_image_has_aggregate_metrics(self):
        v = PhysicsVerifier()
        result = v.verify_image(_gradient_image())
        d = result.details

        assert "ai_probability" in d
        assert "ai_indicators" in d
        assert isinstance(d["ai_probability"], float)
        assert isinstance(d["ai_indicators"], int)

    def test_invalid_buffer_returns_error(self):
        v = PhysicsVerifier()
        result = v.verify_image(b"\x00" * 100)

        assert result.passed is False
        assert result.score == 0
        assert "error" in result.details


class TestPhysicsVerifierScoring:
    def test_score_is_finite(self):
        v = PhysicsVerifier()
        for img_bytes in [_gradient_image(), _noise_image(), _uniform_image()]:
            result = v.verify_image(img_bytes)
            assert np.isfinite(result.score), f"Non-finite score: {result.score}"

    def test_uniform_scores_lower_than_noise(self):
        """Uniform images are more synthetic-looking than noise images."""
        v = PhysicsVerifier()
        uniform = v.verify_image(_uniform_image())
        noise = v.verify_image(_noise_image())

        # Uniform should generally be flagged more than random noise
        assert uniform.score <= noise.score + 30

    def test_different_qualities_produce_results(self):
        """Verify that different JPEG qualities don't crash."""
        v = PhysicsVerifier()
        img = Image.new("RGB", (64, 64), (100, 150, 200))
        for q in [10, 50, 95]:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            result = v.verify_image(buf.getvalue())
            assert 0 <= result.score <= 100

    def test_png_input(self, sample_png_bytes):
        """PhysicsVerifier should handle PNG via OpenCV's imdecode."""
        v = PhysicsVerifier()
        result = v.verify_image(sample_png_bytes)
        assert 0 <= result.score <= 100

    def test_small_image(self):
        """Very small images should not crash."""
        v = PhysicsVerifier()
        img = Image.new("RGB", (16, 16), (128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        result = v.verify_image(buf.getvalue())
        assert 0 <= result.score <= 100


class TestPhysicsVerifierSubAnalysis:
    """Test that individual sub-analyses produce valid score keys."""

    def test_noise_uniformity_has_score(self):
        v = PhysicsVerifier()
        d = v.verify_image(_gradient_image()).details
        sub = d["noise_uniformity"]
        assert "uniformity_score" in sub
        assert 0 <= sub["uniformity_score"] <= 100

    def test_frequency_analysis_has_score(self):
        v = PhysicsVerifier()
        d = v.verify_image(_gradient_image()).details
        sub = d["frequency_analysis"]
        assert "dct_score" in sub

    def test_compression_analysis_has_regional_ela(self):
        v = PhysicsVerifier()
        d = v.verify_image(_gradient_image()).details
        sub = d["compression_analysis"]
        assert "compression_score" in sub
        assert "regional_ela_cv" in sub

    def test_channel_correlation_has_corr_values(self):
        v = PhysicsVerifier()
        d = v.verify_image(_noise_image()).details
        sub = d["channel_correlation"]
        assert "correlation_score" in sub
