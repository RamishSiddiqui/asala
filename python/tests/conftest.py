"""Shared pytest fixtures for Asala tests."""

import io
import struct
import wave

import numpy as np
import pytest

from asala import Asala, CryptoUtils, ManifestBuilder
from asala.types import ContentManifest, ContentType


# ---------------------------------------------------------------------------
# Crypto fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def key_pair():
    """Generate a reusable RSA key pair (session-scoped for speed)."""
    public_key, private_key = CryptoUtils.generate_key_pair()
    return public_key, private_key


@pytest.fixture()
def asala_instance():
    """Fresh Asala instance."""
    return Asala()


# ---------------------------------------------------------------------------
# Sample content fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_text_content():
    """Plain text content bytes."""
    return b"The quick brown fox jumps over the lazy dog."


@pytest.fixture()
def sample_jpeg_bytes():
    """Minimal synthetic JPEG buffer (gradient image)."""
    from PIL import Image

    img = Image.new("RGB", (64, 64))
    pixels = img.load()
    for y in range(64):
        for x in range(64):
            pixels[x, y] = (x * 4, y * 4, 128)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture()
def sample_png_bytes():
    """Minimal synthetic PNG buffer."""
    from PIL import Image

    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def sample_wav_bytes():
    """Generate a 0.5 s mono 16-bit PCM WAV at 16 kHz (440 Hz sine)."""
    sr = 16000
    duration = 0.5
    n_samples = int(sr * duration)
    t = np.arange(n_samples) / sr
    samples = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


@pytest.fixture()
def natural_wav_bytes():
    """Generate a longer WAV with harmonic content, jitter, and noise."""
    sr = 16000
    duration = 2.0
    n_samples = int(sr * duration)
    t = np.arange(n_samples) / sr

    # F0 with jitter
    f0 = 120.0
    f0_jitter = f0 * (1 + 0.01 * np.cumsum(np.random.randn(n_samples) * 0.005))
    phase = 2 * np.pi * np.cumsum(f0_jitter / sr)

    signal = np.zeros(n_samples)
    for h in range(1, 6):
        amp = (1.0 / h) * (1 + 0.03 * np.random.randn(n_samples))
        signal += amp * np.sin(h * phase)

    signal += np.random.randn(n_samples) * 0.02
    signal = signal / (np.max(np.abs(signal)) + 1e-10) * 0.7
    pcm = (np.clip(signal, -1, 1) * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


@pytest.fixture()
def signed_manifest(sample_text_content, key_pair):
    """A signed ContentManifest for sample_text_content."""
    _, private_key = key_pair
    asala = Asala()
    return asala.sign_content(sample_text_content, private_key, "Test Author")
