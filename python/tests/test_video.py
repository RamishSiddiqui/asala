"""Tests for VideoVerifier."""

import io
import os
import tempfile

import cv2
import numpy as np
import pytest

from asala.video import VideoVerifier


def _make_mp4(frames: list, fps: float = 30.0) -> bytes:
    """Write a list of BGR uint8 frames to an MP4 byte buffer via temp file."""
    h, w = frames[0].shape[:2]
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp.name, fourcc, fps, (w, h))
        for frame in frames:
            writer.write(frame)
        writer.release()
        tmp.close()
        with open(tmp.name, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _static_frames(n: int = 30, w: int = 64, h: int = 64) -> list:
    """N identical frames — static video."""
    frame = np.full((h, w, 3), 128, dtype=np.uint8)
    # Add some gradient so it's not perfectly flat
    for y in range(h):
        frame[y, :, 0] = int(y / h * 200)
    return [frame.copy() for _ in range(n)]


def _natural_frames(n: int = 30, w: int = 64, h: int = 64) -> list:
    """Frames with smooth brightness change and noise — natural-looking."""
    frames = []
    for i in range(n):
        brightness = 80 + int(40 * np.sin(2 * np.pi * i / n))
        frame = np.full((h, w, 3), brightness, dtype=np.uint8)
        # Add texture
        noise = np.random.randint(-15, 16, (h, w, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


def _flickering_frames(n: int = 30, w: int = 64, h: int = 64) -> list:
    """Frames with abrupt brightness changes — AI artifact."""
    frames = []
    for i in range(n):
        brightness = 100 + (80 if i % 2 else 0)
        frame = np.full((h, w, 3), brightness, dtype=np.uint8)
        frames.append(frame)
    return frames


class TestVideoVerifierBasic:
    def test_instantiation(self):
        v = VideoVerifier()
        assert v is not None

    @pytest.mark.slow
    def test_verify_natural_video(self):
        """Natural video frames should produce a valid result."""
        v = VideoVerifier()
        video_bytes = _make_mp4(_natural_frames(30))
        result = v.verify_video(video_bytes)

        assert result.name == "physics"
        assert isinstance(result.passed, bool)
        assert 0 <= result.score <= 100

    @pytest.mark.slow
    def test_verify_has_all_analysis_keys(self):
        v = VideoVerifier()
        video_bytes = _make_mp4(_natural_frames(30))
        result = v.verify_video(video_bytes)
        d = result.details

        for key in [
            "frame_analysis",
            "temporal_noise",
            "optical_flow",
            "encoding_analysis",
            "temporal_lighting",
            "frame_stability",
        ]:
            assert key in d, f"Missing key: {key}"

        assert "ai_probability" in d
        assert "ai_indicators" in d

    @pytest.mark.slow
    def test_too_short_video(self):
        """Video with < 2 frames should return error."""
        v = VideoVerifier()
        one_frame = _static_frames(1)
        video_bytes = _make_mp4(one_frame)
        result = v.verify_video(video_bytes)
        # Might decode to 1 frame or fail
        assert result.score <= 50 or "error" in result.details

    def test_invalid_buffer(self):
        v = VideoVerifier()
        result = v.verify_video(b"\x00" * 100)
        assert result.passed is False
        assert result.score == 0
        assert "error" in result.details


class TestVideoVerifierScoring:
    @pytest.mark.slow
    def test_static_video_flagged(self):
        """Static (looped) video should be flagged as suspicious."""
        v = VideoVerifier()
        video_bytes = _make_mp4(_static_frames(30))
        result = v.verify_video(video_bytes)
        # Static video has very high NCC, should have low score
        assert result.score <= 70

    @pytest.mark.slow
    def test_flickering_flagged(self):
        """Flickering video should be flagged."""
        v = VideoVerifier()
        video_bytes = _make_mp4(_flickering_frames(30))
        result = v.verify_video(video_bytes)
        assert result.passed is False

    @pytest.mark.slow
    def test_scores_are_finite(self):
        v = VideoVerifier()
        video_bytes = _make_mp4(_natural_frames(20))
        result = v.verify_video(video_bytes)
        assert np.isfinite(result.score)


class TestVideoVerifierSubAnalysis:
    @pytest.fixture()
    def result(self):
        v = VideoVerifier()
        video_bytes = _make_mp4(_natural_frames(30))
        return v.verify_video(video_bytes)

    @pytest.mark.slow
    def test_frame_analysis_has_score(self, result):
        sub = result.details["frame_analysis"]
        assert "frame_analysis_score" in sub

    @pytest.mark.slow
    def test_temporal_noise_has_score(self, result):
        sub = result.details["temporal_noise"]
        assert "temporal_noise_score" in sub

    @pytest.mark.slow
    def test_optical_flow_has_score(self, result):
        sub = result.details["optical_flow"]
        assert "optical_flow_score" in sub

    @pytest.mark.slow
    def test_encoding_has_score(self, result):
        sub = result.details["encoding_analysis"]
        assert "encoding_score" in sub

    @pytest.mark.slow
    def test_temporal_lighting_has_score(self, result):
        sub = result.details["temporal_lighting"]
        assert "temporal_lighting_score" in sub

    @pytest.mark.slow
    def test_frame_stability_has_score(self, result):
        sub = result.details["frame_stability"]
        assert "stability_score" in sub
