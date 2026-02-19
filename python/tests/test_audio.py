"""Tests for AudioVerifier."""

import io
import wave

import numpy as np
import pytest

from asala.audio import AudioVerifier


def _make_wav(samples: np.ndarray, sr: int = 16000) -> bytes:
    """Encode float32 mono samples to 16-bit PCM WAV."""
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _sine_wav(freq: float = 440.0, duration: float = 1.0, sr: int = 16000) -> bytes:
    """Pure sine wave — clearly synthetic."""
    t = np.arange(int(sr * duration)) / sr
    return _make_wav(0.5 * np.sin(2 * np.pi * freq * t), sr)


def _tts_like_wav(duration: float = 2.0, sr: int = 16000) -> bytes:
    """TTS-like: rock-steady pitch, no shimmer, minimal noise."""
    t = np.arange(int(sr * duration)) / sr
    f0 = 150.0
    phase = 2 * np.pi * f0 * t
    signal = np.zeros_like(t)
    for h in range(1, 10):
        signal += (1.0 / h) * np.sin(h * phase)
    signal += np.random.randn(len(t)) * 0.002
    signal = signal / (np.max(np.abs(signal)) + 1e-10) * 0.8
    return _make_wav(signal, sr)


def _noise_wav(duration: float = 1.0, sr: int = 16000) -> bytes:
    """Pure noise — no speech structure."""
    return _make_wav(np.random.randn(int(sr * duration)) * 0.3, sr)


class TestAudioVerifierBasic:
    def test_instantiation(self):
        v = AudioVerifier()
        assert v is not None

    def test_verify_returns_layer_result(self, sample_wav_bytes):
        v = AudioVerifier()
        result = v.verify_audio(sample_wav_bytes)

        assert result.name == "physics"
        assert isinstance(result.passed, bool)
        assert 0 <= result.score <= 100
        assert isinstance(result.details, dict)

    def test_invalid_buffer_returns_error(self):
        v = AudioVerifier()
        result = v.verify_audio(b"\x00" * 100)

        assert result.passed is False
        assert result.score == 0
        assert "error" in result.details

    def test_too_short_audio(self):
        """Audio shorter than 0.1s should return error."""
        sr = 16000
        samples = np.zeros(int(sr * 0.05))  # 0.05s
        wav = _make_wav(samples, sr)
        v = AudioVerifier()
        result = v.verify_audio(wav)
        assert result.score == 0


class TestAudioVerifierSubAnalysis:
    """Test all 10 sub-analysis keys are present."""

    @pytest.fixture()
    def result(self, natural_wav_bytes):
        v = AudioVerifier()
        return v.verify_audio(natural_wav_bytes)

    def test_phase_coherence_present(self, result):
        assert "phase_coherence" in result.details
        assert "phase_score" in result.details["phase_coherence"]

    def test_voice_quality_present(self, result):
        assert "voice_quality" in result.details
        sub = result.details["voice_quality"]
        assert "voice_quality_score" in sub

    def test_enf_analysis_present(self, result):
        assert "enf_analysis" in result.details
        assert "enf_score" in result.details["enf_analysis"]

    def test_spectral_tilt_present(self, result):
        assert "spectral_tilt" in result.details
        assert "spectral_tilt_score" in result.details["spectral_tilt"]

    def test_noise_consistency_present(self, result):
        assert "noise_consistency" in result.details
        assert "noise_consistency_score" in result.details["noise_consistency"]

    def test_mel_regularity_present(self, result):
        assert "mel_regularity" in result.details
        assert "mel_regularity_score" in result.details["mel_regularity"]

    def test_formant_bandwidth_present(self, result):
        assert "formant_bandwidth" in result.details
        assert "formant_score" in result.details["formant_bandwidth"]

    def test_double_compression_present(self, result):
        assert "double_compression" in result.details
        assert "double_comp_score" in result.details["double_compression"]

    def test_spectral_discontinuity_present(self, result):
        assert "spectral_discontinuity" in result.details
        assert "splice_score" in result.details["spectral_discontinuity"]

    def test_subband_energy_present(self, result):
        assert "subband_energy" in result.details
        assert "subband_score" in result.details["subband_energy"]

    def test_aggregate_metrics(self, result):
        assert "ai_probability" in result.details
        assert "ai_indicators" in result.details


class TestAudioVerifierScoring:
    def test_scores_are_finite(self, sample_wav_bytes):
        v = AudioVerifier()
        result = v.verify_audio(sample_wav_bytes)
        assert np.isfinite(result.score)

    def test_pure_sine_detected(self):
        """Pure sine wave should be flagged as synthetic."""
        v = AudioVerifier()
        result = v.verify_audio(_sine_wav(440, 2.0))
        assert result.passed is False

    def test_tts_like_detected(self):
        """TTS-like audio should be flagged."""
        v = AudioVerifier()
        result = v.verify_audio(_tts_like_wav())
        assert result.passed is False

    def test_noise_produces_result(self):
        """Pure noise should not crash and produce a result."""
        v = AudioVerifier()
        result = v.verify_audio(_noise_wav(2.0))
        assert 0 <= result.score <= 100

    def test_different_sample_rates(self):
        """Test that 8 kHz and 44.1 kHz WAVs work."""
        v = AudioVerifier()
        for sr in [8000, 44100]:
            t = np.arange(int(sr * 1.0)) / sr
            samples = 0.5 * np.sin(2 * np.pi * 300 * t)
            wav = _make_wav(samples, sr)
            result = v.verify_audio(wav)
            assert 0 <= result.score <= 100
