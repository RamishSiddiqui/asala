"""
Audio Physics Layer Benchmark — tests AudioVerifier with synthetic audio.

Generates several categories of audio:
  1. Real-like speech  — simulated natural voice with noise, jitter, ENF
  2. TTS-like audio    — unnaturally smooth, low jitter, high HNR
  3. Spliced audio     — discontinuous noise floor / spectral features
  4. Vocoder artifacts — phase randomisation, HF cutoff
"""

import io
import struct
import sys
import wave
from pathlib import Path

import numpy as np

# ---- path setup (use importlib to avoid cryptography DLL issue) ----
import importlib.util
_project = Path(__file__).resolve().parent.parent.parent
_pkg_dir = _project / 'python' / 'asala'


def _load(name, fp):
    spec = importlib.util.spec_from_file_location(name, str(fp))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load('asala.types', _pkg_dir / 'types.py')
_audio_mod = _load('asala.audio', _pkg_dir / 'audio.py')
AudioVerifier = _audio_mod.AudioVerifier


# ---------------------------------------------------------------------------
# Helper: write WAV bytes from float32 mono samples
# ---------------------------------------------------------------------------
def _to_wav(samples: np.ndarray, sr: int = 16000) -> bytes:
    """Convert float32 mono samples to 16-bit PCM WAV bytes."""
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Audio generators
# ---------------------------------------------------------------------------

def generate_natural_speech(duration: float = 2.0, sr: int = 16000) -> bytes:
    """Simulate natural-sounding speech with jitter, shimmer, noise, ENF."""
    t = np.arange(int(sr * duration)) / sr

    # Fundamental frequency with natural jitter (~1%)
    f0_base = 120.0  # Hz
    f0_jitter = f0_base * (1 + 0.01 * np.cumsum(np.random.randn(len(t)) * 0.005))
    phase = 2 * np.pi * np.cumsum(f0_jitter / sr)

    # Harmonics with shimmer (~4%)
    signal_out = np.zeros_like(t)
    for h in range(1, 8):
        amp = (1.0 / h) * (1 + 0.04 * np.random.randn(len(t)))
        signal_out += amp * np.sin(h * phase)

    # Add formant-like envelope (simple bandpass shaping)
    from scipy.signal import butter, sosfilt
    sos = butter(2, [200, 3500], btype='band', fs=sr, output='sos')
    signal_out = sosfilt(sos, signal_out)

    # Add natural background noise
    noise = np.random.randn(len(t)) * 0.02
    signal_out += noise

    # Add subtle ENF (50 Hz hum)
    enf = 0.003 * np.sin(2 * np.pi * 50 * t)
    signal_out += enf

    # Normalize
    signal_out = signal_out / (np.max(np.abs(signal_out)) + 1e-10) * 0.8

    # Add amplitude envelope (natural speech has varying loudness)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t)  # ~2 Hz modulation
    signal_out *= envelope

    return _to_wav(signal_out, sr)


def generate_tts_like(duration: float = 2.0, sr: int = 16000) -> bytes:
    """Simulate TTS output: very low jitter, no shimmer, high HNR, no ENF."""
    t = np.arange(int(sr * duration)) / sr

    # Rock-steady F0 (near-zero jitter)
    f0 = 150.0
    phase = 2 * np.pi * f0 * t

    # Perfect harmonics (no shimmer)
    signal_out = np.zeros_like(t)
    for h in range(1, 10):
        amp = 1.0 / h
        signal_out += amp * np.sin(h * phase)

    # Mel-smoothed output (very regular)
    from scipy.signal import butter, sosfilt
    # Sharp HF cutoff at 7.5 kHz (vocoder limitation)
    if sr > 15000:
        sos = butter(6, 7500, btype='low', fs=sr, output='sos')
        signal_out = sosfilt(sos, signal_out)

    # Minimal noise (high HNR)
    noise = np.random.randn(len(t)) * 0.002
    signal_out += noise

    # No ENF

    signal_out = signal_out / (np.max(np.abs(signal_out)) + 1e-10) * 0.8
    return _to_wav(signal_out, sr)


def generate_spliced_audio(duration: float = 2.0, sr: int = 16000) -> bytes:
    """Create audio with an obvious splice in the middle."""
    half = int(sr * duration / 2)
    t1 = np.arange(half) / sr
    t2 = np.arange(half) / sr

    # First half: low-pitched voice with one noise floor
    f0_1 = 100.0
    phase1 = 2 * np.pi * f0_1 * t1
    seg1 = np.sin(phase1) + 0.5 * np.sin(2 * phase1) + np.random.randn(half) * 0.03

    # Second half: higher-pitched voice with different noise floor
    f0_2 = 200.0
    phase2 = 2 * np.pi * f0_2 * t2
    seg2 = np.sin(phase2) + 0.3 * np.sin(3 * phase2) + np.random.randn(half) * 0.08

    # Hard splice (no crossfade)
    combined = np.concatenate([seg1, seg2])
    combined = combined / (np.max(np.abs(combined)) + 1e-10) * 0.8
    return _to_wav(combined, sr)


def generate_vocoder_output(duration: float = 2.0, sr: int = 16000) -> bytes:
    """Simulate vocoder output: magnitude from speech, random phase."""
    t = np.arange(int(sr * duration)) / sr

    # Generate speech-like magnitude spectrum
    f0 = 130.0
    phase_natural = 2 * np.pi * f0 * t
    speech = np.zeros_like(t)
    for h in range(1, 8):
        speech += (1.0 / h) * np.sin(h * phase_natural)

    # STFT, randomize phase, ISTFT
    from scipy.signal import stft, istft
    f, t_stft, Zxx = stft(speech, fs=sr, nperseg=512, noverlap=384)
    magnitude = np.abs(Zxx)
    random_phase = np.random.uniform(-np.pi, np.pi, Zxx.shape)
    Zxx_vocoder = magnitude * np.exp(1j * random_phase)
    _, reconstructed = istft(Zxx_vocoder, fs=sr, nperseg=512, noverlap=384)

    # Trim to original length
    reconstructed = reconstructed[:len(t)]
    if len(reconstructed) < len(t):
        reconstructed = np.pad(reconstructed, (0, len(t) - len(reconstructed)))

    # Add minimal noise
    reconstructed += np.random.randn(len(reconstructed)) * 0.005
    reconstructed = reconstructed / (np.max(np.abs(reconstructed)) + 1e-10) * 0.7

    return _to_wav(reconstructed, sr)


def generate_pure_tone(duration: float = 2.0, sr: int = 16000) -> bytes:
    """Simple sine wave — clearly synthetic."""
    t = np.arange(int(sr * duration)) / sr
    signal_out = 0.5 * np.sin(2 * np.pi * 440 * t)
    return _to_wav(signal_out, sr)


def generate_noise_only(duration: float = 2.0, sr: int = 16000) -> bytes:
    """Pure noise — no speech structure."""
    samples = np.random.randn(int(sr * duration)) * 0.3
    return _to_wav(samples, sr)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def main():
    v = AudioVerifier()

    print("=" * 70)
    print("ASALA AUDIO PHYSICS LAYER — BENCHMARK")
    print("=" * 70)

    categories = [
        ("REAL-LIKE AUDIO (expected: passed=True, score>=50)", [
            ("natural_speech", generate_natural_speech),
        ], True),
        ("SYNTHETIC / AI-LIKE AUDIO (expected: passed=False, score<50)", [
            ("tts_like", generate_tts_like),
            ("vocoder_output", generate_vocoder_output),
            ("pure_tone", generate_pure_tone),
        ], False),
        ("MANIPULATED AUDIO (expected: passed=False or low score)", [
            ("spliced_audio", generate_spliced_audio),
        ], False),
        ("EDGE CASES", [
            ("noise_only", generate_noise_only),
        ], None),  # No expected result
    ]

    total_correct = 0
    total_tests = 0

    for cat_name, tests, expected_pass in categories:
        print(f"\n{'-' * 70}")
        print(f"  {cat_name}")
        print(f"{'-' * 70}")

        for name, gen_func in tests:
            audio_bytes = gen_func()
            result = v.verify_audio(audio_bytes)
            d = result.details

            is_correct = None
            if expected_pass is not None:
                is_correct = (result.passed == expected_pass)
                total_tests += 1
                if is_correct:
                    total_correct += 1

            status = "[PASS]" if (is_correct or is_correct is None) else "[FAIL]"
            print(f"  {status} {name:30s}  score={result.score:3d}  "
                  f"passed={str(result.passed):5s}  "
                  f"ai_prob={d.get('ai_probability', '?'):.2f}  "
                  f"indicators={d.get('ai_indicators', '?')}")

            if d.get('warning'):
                print(f"         warning: {d['warning']}")

            # Show individual method scores
            score_keys = {
                'phase_coherence': 'phase_score',
                'voice_quality': 'voice_quality_score',
                'enf_analysis': 'enf_score',
                'spectral_tilt': 'spectral_tilt_score',
                'noise_consistency': 'noise_consistency_score',
                'mel_regularity': 'mel_regularity_score',
                'formant_bandwidth': 'formant_score',
                'double_compression': 'double_comp_score',
                'spectral_discontinuity': 'splice_score',
                'subband_energy': 'subband_score',
            }
            scores = []
            for method, key in score_keys.items():
                sub = d.get(method, {})
                if isinstance(sub, dict) and key in sub:
                    scores.append(f"{method.split('_')[0]}={sub[key]}")
            print(f"         [{', '.join(scores)}]")

            # Show key metrics for voice quality
            vq = d.get('voice_quality', {})
            if 'jitter' in vq:
                print(f"         jitter={vq['jitter']:.4f}  "
                      f"shimmer={vq['shimmer']:.4f}  "
                      f"hnr={vq.get('hnr', 0):.1f}dB  "
                      f"voiced={vq.get('voiced_frames', 0)}")

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {total_correct}/{total_tests} correct "
          f"({100 * total_correct / total_tests:.1f}%)" if total_tests else "No scored tests")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
