"""
Audio Physics-based Verification for AI-generated Audio Detection.

Implements 10 mathematical analysis techniques to detect AI-generated or
manipulated audio based on physical and acoustic inconsistencies.  All methods
use scipy/numpy signal processing — no neural networks or training data.

Methods:
  1. Phase Coherence Analysis      — vocoder detection via group delay deviation
  2. Voice Quality Metrics         — jitter, shimmer, HNR
  3. ENF Analysis                  — 50/60 Hz electrical network frequency
  4. Spectral Tilt / LTAS          — long-term spectral envelope shape
  5. Background Noise Consistency  — noise floor stability across segments
  6. Mel-Spectrogram Regularity    — frame-to-frame variability in mel bands
  7. Formant Bandwidth Analysis    — LPC-based formant valley depth
  8. Double Compression Detection  — re-encoding artifact patterns
  9. Spectral Discontinuity        — splice point detection via spectral flux
 10. Sub-band Energy Analysis      — frequency-band energy distribution
"""

import concurrent.futures
import io
import logging
import struct
import wave
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.fft import fft, rfft, rfftfreq

from .types import LayerResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WAV decoder
# ---------------------------------------------------------------------------

def _decode_wav(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Decode WAV bytes to mono float32 samples + sample rate.

    Supports 8-bit, 16-bit, 24-bit, and 32-bit PCM WAV files.
    Returns (samples_float32_mono, sample_rate).
    """
    buf = io.BytesIO(audio_bytes)
    with wave.open(buf, 'rb') as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    elif sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 3:
        # 24-bit: vectorized unpack of 3-byte little-endian samples
        n_samples = len(raw) // 3
        raw_arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        samples = (raw_arr[:, 0].astype(np.int32)
                   | (raw_arr[:, 1].astype(np.int32) << 8)
                   | (raw_arr[:, 2].astype(np.int32) << 16))
        # Sign extension for negative values
        samples = np.where(samples >= 0x800000, samples - 0x1000000, samples)
        samples = samples.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    # Convert to mono by averaging channels
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples, sr


# ---------------------------------------------------------------------------
# Helper: mel filterbank
# ---------------------------------------------------------------------------

def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 40) -> np.ndarray:
    """Build a mel-scale triangular filterbank matrix (n_mels x n_fft//2+1)."""
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    n_bins = n_fft // 2 + 1
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sr / 2)
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = np.array([mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_mels, n_bins))
    for m in range(1, n_mels + 1):
        f_left = bin_points[m - 1]
        f_center = bin_points[m]
        f_right = bin_points[m + 1]
        for k in range(f_left, f_center):
            if f_center > f_left:
                fb[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right > f_center:
                fb[m - 1, k] = (f_right - k) / (f_right - f_center)
    return fb


# ---------------------------------------------------------------------------
# AudioVerifier
# ---------------------------------------------------------------------------

class AudioVerifier:
    """Physics-based audio verification layer.

    Detects AI-generated or manipulated audio using 10 signal-processing
    methods that require no training data or neural networks.

    Scoring: each method returns a score 0-100:
        - Higher = more consistent with real recording
        - Lower  = more consistent with AI / manipulation
    """

    def __init__(self, max_workers: int = 1):
        """Initialize AudioVerifier.

        Args:
            max_workers: Number of threads for parallel analysis.
                1 (default) runs sequentially.  Values > 1 use a
                ThreadPoolExecutor to run the 10 analysis methods
                concurrently.
        """
        self._max_workers = max(1, max_workers)

    # Weights for composite scoring — discriminative methods get more weight
    WEIGHTS = {
        'phase_coherence': 0.08,
        'voice_quality': 0.22,       # Best discriminator: jitter/shimmer/HNR
        'enf_analysis': 0.05,        # Neutral-default, weak discriminator
        'spectral_tilt': 0.10,
        'noise_consistency': 0.12,
        'mel_regularity': 0.10,
        'formant_bandwidth': 0.10,
        'double_compression': 0.06,
        'spectral_discontinuity': 0.09,
        'subband_energy': 0.08,
    }

    # Indicator thresholds
    THRESHOLDS = {
        'phase_gdd_std_low': 0.5,      # GDD std below this → vocoder
        'jitter_low': 0.002,            # Jitter below 0.2% → TTS
        'shimmer_low': 0.015,           # Shimmer below 1.5% → TTS
        'hnr_high': 30.0,              # HNR above 30 dB → TTS
        'enf_snr_high': 5.0,           # ENF SNR above this → real
        'mel_cv_low': 0.15,            # Mel CV below this → TTS
        'spectral_tilt_flat': -2.0,    # Tilt flatter than -2 dB/oct → TTS
        'noise_floor_cv_high': 0.8,    # Noise floor CV above this → splice
        'formant_valley_low': 8.0,     # Valley depth below 8 dB → vocoder
        'splice_flux_high': 3.0,       # Spectral flux z-score > 3 → splice
    }

    def verify_audio(self, audio_bytes: bytes) -> LayerResult:
        """Verify audio for physical consistency.

        Args:
            audio_bytes: Raw audio file bytes (WAV format).

        Returns:
            LayerResult with passed status, score, and detailed analysis.
        """
        try:
            samples, sr = _decode_wav(audio_bytes)
        except Exception as e:
            logger.error(f"Failed to decode audio: {e}")
            return LayerResult(
                name='physics',
                passed=False,
                score=0,
                details={'error': f'Audio decode failed: {e}'}
            )

        if len(samples) < sr * 0.1:  # Need at least 0.1s
            return LayerResult(
                name='physics', passed=False, score=0,
                details={'error': 'Audio too short for analysis (< 0.1s)'}
            )

        results: Dict[str, Any] = {}

        # Pre-compute shared STFTs once (avoid redundant computation)
        # Methods 1+6 share n_fft=1024, hop=256
        stft_1024 = signal.stft(samples, fs=sr, nperseg=1024, noverlap=1024 - 256)
        # Methods 4+10 share n_fft=2048, hop=512
        stft_2048 = signal.stft(samples, fs=sr, nperseg=2048, noverlap=2048 - 512)

        # Define all analysis tasks: (key, callable, args)
        analysis_tasks = [
            ('phase_coherence', self._analyze_phase_coherence, (samples, sr, stft_1024)),
            ('voice_quality', self._analyze_voice_quality, (samples, sr)),
            ('enf_analysis', self._analyze_enf, (samples, sr)),
            ('spectral_tilt', self._analyze_spectral_tilt, (samples, sr, stft_2048)),
            ('noise_consistency', self._analyze_noise_consistency, (samples, sr)),
            ('mel_regularity', self._analyze_mel_regularity, (samples, sr, stft_1024)),
            ('formant_bandwidth', self._analyze_formant_bandwidth, (samples, sr)),
            ('double_compression', self._analyze_double_compression, (samples, sr)),
            ('spectral_discontinuity', self._analyze_spectral_discontinuity, (samples, sr)),
            ('subband_energy', self._analyze_subband_energy, (samples, sr, stft_2048)),
        ]

        # Run all 10 analyses (parallel or sequential)
        if self._max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = {
                    key: executor.submit(fn, *args)
                    for key, fn, args in analysis_tasks
                }
                for key, future in futures.items():
                    results[key] = future.result()
        else:
            for key, fn, args in analysis_tasks:
                results[key] = fn(*args)

        # Composite score
        final_score = self._calculate_composite_score(results)

        # Count AI indicators
        ai_indicators = self._count_ai_indicators(results)
        total_indicators = 18  # max possible (14 base + 2 vocoder + 2 perfect-pitch)
        ai_probability = ai_indicators / total_indicators

        # Determine result
        passed, final_score, warning = self._determine_result(
            final_score, ai_probability, results
        )

        if warning:
            results['warning'] = warning
        results['ai_probability'] = ai_probability
        results['ai_indicators'] = ai_indicators

        return LayerResult(
            name='physics',
            passed=passed,
            score=final_score,
            details=results
        )

    # ------------------------------------------------------------------
    # 1. Phase Coherence Analysis
    # ------------------------------------------------------------------
    def _analyze_phase_coherence(self, samples: np.ndarray, sr: int,
                                   stft_result: tuple = None) -> Dict[str, Any]:
        """Detect vocoders via inter-frame phase coherence in speech bands.

        Natural speech has smoothly varying phase between frames in the
        1-4 kHz band.  Neural vocoders reconstruct magnitude but produce
        incoherent phase.  Pure tones have unnaturally HIGH coherence.
        """
        try:
            if stft_result is not None:
                f, t, Zxx = stft_result
            else:
                n_fft = 1024
                hop = 256
                f, t, Zxx = signal.stft(samples, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)

            phase = np.angle(Zxx)

            # Focus on speech-critical band (300-4000 Hz)
            band_mask = (f >= 300) & (f <= min(4000, sr / 2))
            if np.sum(band_mask) < 5:
                return {'phase_score': 50, 'note': 'Insufficient frequency range'}

            phase_band = phase[band_mask, :]

            # Inter-frame phase coherence: how consistent is phase evolution?
            # For each freq bin, compute mean resultant length of phase diffs
            if phase_band.shape[1] < 3:
                return {'phase_score': 50, 'note': 'Too few frames'}

            phase_diff = np.diff(phase_band, axis=1)
            # Mean resultant length per freq bin (high = coherent phase evolution)
            mrl_per_bin = np.abs(np.mean(np.exp(1j * phase_diff), axis=1))
            avg_coherence = float(np.mean(mrl_per_bin))

            # Coherence variability across frequency bins
            coherence_std = float(np.std(mrl_per_bin))

            # GDD for additional info
            d_phase_freq = np.diff(phase, axis=0)
            d_phase_freq = np.angle(np.exp(1j * d_phase_freq))
            gdd_std = float(np.mean(np.std(d_phase_freq, axis=0)))

            # Score based on coherence:
            # Natural speech: moderate coherence (0.2-0.6) with variation
            # Vocoders/random phase: low coherence (<0.15)
            # Pure tones: very high coherence (>0.8) — also suspicious
            if avg_coherence < 0.1:
                score = 20  # Random phase — vocoder
            elif avg_coherence < 0.25:
                score = int(20 + (avg_coherence - 0.1) / 0.15 * 30)
            elif avg_coherence < 0.65:
                score = int(50 + (avg_coherence - 0.25) / 0.4 * 40)
            elif avg_coherence < 0.85:
                score = int(90 - (avg_coherence - 0.65) / 0.2 * 20)
            else:
                score = int(np.clip(70 - (avg_coherence - 0.85) * 200, 20, 70))  # Too perfect

            return {
                'gdd_std': gdd_std,
                'phase_coherence_val': avg_coherence,
                'coherence_std': coherence_std,
                'phase_score': score,
            }
        except Exception as e:
            logger.warning(f"Phase coherence analysis failed: {e}")
            return {'phase_score': 50, 'error': str(e)}

    # ------------------------------------------------------------------
    # 2. Voice Quality Metrics (Jitter / Shimmer / HNR)
    # ------------------------------------------------------------------
    def _analyze_voice_quality(self, samples: np.ndarray, sr: int) -> Dict[str, Any]:
        """Measure jitter, shimmer, and HNR via autocorrelation pitch tracking.

        Human voices have natural micro-perturbations:
          - Jitter (F0 variation): ~0.5-1.0% in normal speech
          - Shimmer (amplitude variation): ~3-5%
          - HNR: 15-25 dB
        TTS: jitter < 0.2%, shimmer < 1%, HNR > 30 dB
        """
        try:
            # Work on a voiced segment (middle 80% to avoid silence)
            start = len(samples) // 10
            end = len(samples) - start
            seg = samples[start:end]

            # Frame-based pitch detection via autocorrelation
            frame_len = int(0.03 * sr)  # 30ms frames
            hop = int(0.01 * sr)  # 10ms hop
            min_lag = int(sr / 500)  # max F0 = 500 Hz
            max_lag = int(sr / 50)   # min F0 = 50 Hz

            periods = []
            amplitudes = []
            hnr_estimates = []

            for i in range(0, len(seg) - frame_len, hop):
                frame = seg[i:i + frame_len]
                if np.max(np.abs(frame)) < 0.01:
                    continue  # Skip silence

                # Autocorrelation (computed once per frame)
                corr = np.correlate(frame, frame, mode='full')
                corr = corr[len(corr) // 2:]  # positive lags only

                if max_lag >= len(corr):
                    continue

                # Normalize
                corr = corr / (corr[0] + 1e-10)

                # Find peak in valid lag range
                search_region = corr[min_lag:max_lag]
                if len(search_region) == 0:
                    continue

                peak_idx = np.argmax(search_region) + min_lag
                peak_val = corr[peak_idx]

                # HNR from autocorrelation peak (reuse same corr)
                r_max = np.max(search_region)
                if 0 < r_max < 1:
                    hnr_estimates.append(10 * np.log10(r_max / (1 - r_max + 1e-10)))

                if peak_val > 0.3:  # Voiced frame
                    periods.append(peak_idx / sr)
                    amplitudes.append(float(np.sqrt(np.mean(frame ** 2))))

            if len(periods) < 5:
                return {
                    'jitter': 0.0, 'shimmer': 0.0, 'hnr': 0.0,
                    'voice_quality_score': 50,
                    'voiced_frames': len(periods),
                    'note': 'Insufficient voiced frames'
                }

            periods = np.array(periods)
            amplitudes = np.array(amplitudes)

            # Jitter: average absolute difference between consecutive periods
            # divided by mean period
            period_diffs = np.abs(np.diff(periods))
            jitter = float(np.mean(period_diffs) / (np.mean(periods) + 1e-10))

            # Shimmer: average absolute difference in amplitude / mean amplitude
            amp_diffs = np.abs(np.diff(amplitudes))
            shimmer = float(np.mean(amp_diffs) / (np.mean(amplitudes) + 1e-10))

            # HNR: from autocorrelation peak (already computed above)
            hnr = float(np.median(hnr_estimates)) if hnr_estimates else 15.0

            # Score: natural ranges get high scores
            # Jitter: 0.003-0.015 is natural (0.3%-1.5%)
            if jitter < 0.001:
                jitter_score = int(jitter / 0.001 * 30)  # Too low = TTS
            elif jitter < 0.02:
                jitter_score = int(np.clip(30 + (jitter - 0.001) / 0.019 * 70, 30, 100))
            else:
                jitter_score = int(np.clip(100 - (jitter - 0.02) * 500, 20, 100))

            # Shimmer: 0.02-0.08 is natural (2%-8%)
            if shimmer < 0.01:
                shimmer_score = int(shimmer / 0.01 * 30)
            elif shimmer < 0.10:
                shimmer_score = int(np.clip(30 + (shimmer - 0.01) / 0.09 * 70, 30, 100))
            else:
                shimmer_score = int(np.clip(100 - (shimmer - 0.10) * 300, 20, 100))

            # HNR: 12-28 dB is natural
            if hnr > 35:
                hnr_score = int(np.clip(100 - (hnr - 35) * 10, 10, 60))  # Too clean = TTS
            elif hnr > 10:
                hnr_score = int(np.clip((hnr - 10) / 25 * 80 + 20, 20, 100))
            else:
                hnr_score = int(np.clip(hnr / 10 * 30, 0, 30))

            voice_quality_score = int((jitter_score + shimmer_score + hnr_score) / 3)

            # Pathological patterns get extra penalty:
            # High jitter + very low HNR = random phase reconstruction (vocoder)
            if jitter > 0.05 and hnr < 5:
                voice_quality_score = max(voice_quality_score - 30, 0)

            return {
                'jitter': jitter,
                'shimmer': shimmer,
                'hnr': hnr,
                'voiced_frames': len(periods),
                'jitter_score': jitter_score,
                'shimmer_score': shimmer_score,
                'hnr_score': hnr_score,
                'voice_quality_score': voice_quality_score,
            }
        except Exception as e:
            logger.warning(f"Voice quality analysis failed: {e}")
            return {'voice_quality_score': 50, 'error': str(e)}

    # ------------------------------------------------------------------
    # 3. ENF Analysis (Electrical Network Frequency)
    # ------------------------------------------------------------------
    def _analyze_enf(self, samples: np.ndarray, sr: int) -> Dict[str, Any]:
        """Detect 50/60 Hz electrical network frequency in the recording.

        Real recordings made near AC power embed a subtle ENF trace via
        electromagnetic induction or lighting flicker.  Only checks 50 Hz
        and 60 Hz fundamentals (not harmonics) to avoid confusion with
        speech F0 harmonics near 100/120 Hz.
        """
        try:
            if sr < 200:
                return {'enf_score': 50, 'note': 'Sample rate too low for ENF'}

            # Use long FFT for high frequency resolution
            n_fft = min(len(samples), sr * 4)  # up to 4 seconds
            spec = np.abs(rfft(samples[:n_fft]))
            freqs = rfftfreq(n_fft, 1.0 / sr)
            freq_res = sr / n_fft

            best_snr = 0.0
            best_freq = 0.0

            # Only check 50 and 60 Hz fundamentals (not harmonics, which
            # overlap with speech F0 harmonics)
            for target_freq in [50.0, 60.0]:
                if target_freq >= sr / 2:
                    continue

                # Check that this isn't a speech harmonic:
                # if there's energy at integer multiples of target_freq,
                # it's likely speech, not ENF
                is_speech_harmonic = False
                for mult in [2, 3, 4]:
                    harm_freq = target_freq * mult
                    if harm_freq < sr / 2:
                        harm_bin = np.argmin(np.abs(freqs - harm_freq))
                        target_bin = np.argmin(np.abs(freqs - target_freq))
                        # If harmonic is stronger than fundamental, it's speech
                        if spec[harm_bin] > spec[target_bin] * 0.5:
                            is_speech_harmonic = True
                            break

                if is_speech_harmonic:
                    continue

                # Narrow band peak detection
                target_bin = np.argmin(np.abs(freqs - target_freq))
                window = max(1, int(0.5 / freq_res))  # 0.5 Hz window
                lo = max(0, target_bin - window)
                hi = min(len(spec), target_bin + window + 1)
                peak_val = np.max(spec[lo:hi])

                # Background: median of wider region excluding the peak
                bg_lo = max(0, target_bin - 100)
                bg_hi = min(len(spec), target_bin + 100)
                bg_mask = np.ones(bg_hi - bg_lo, dtype=bool)
                bg_mask[lo - bg_lo:hi - bg_lo] = False
                bg_region = spec[bg_lo:bg_hi][bg_mask]
                bg_val = np.median(bg_region) if len(bg_region) > 0 else 1.0

                peak_snr = peak_val / (bg_val + 1e-10)

                if peak_snr > best_snr:
                    best_snr = peak_snr
                    best_freq = target_freq

            enf_detected = best_snr > 5.0  # Stricter threshold

            # Score: ENF detected = bonus (real recording). Neutral otherwise.
            if enf_detected:
                enf_score = int(np.clip(55 + min(best_snr, 20) * 2, 55, 90))
            else:
                enf_score = 45  # Neutral — absence doesn't prove synthetic

            return {
                'enf_detected': enf_detected,
                'enf_best_freq': best_freq,
                'enf_snr': float(best_snr),
                'enf_score': enf_score,
            }
        except Exception as e:
            logger.warning(f"ENF analysis failed: {e}")
            return {'enf_score': 50, 'error': str(e)}

    # ------------------------------------------------------------------
    # 4. Spectral Tilt / LTAS
    # ------------------------------------------------------------------
    def _analyze_spectral_tilt(self, samples: np.ndarray, sr: int,
                                stft_result: tuple = None) -> Dict[str, Any]:
        """Measure long-term average spectrum and spectral tilt.

        Natural speech has ~-6 dB/octave tilt above F0.  TTS vocoders
        produce different tilt (too flat above 4 kHz, or sharp cutoff
        at 8-12 kHz from vocoder bandwidth limitations).
        """
        try:
            if stft_result is not None:
                f, t, Sxx = stft_result
            else:
                n_fft = 2048
                hop = 512
                f, t, Sxx = signal.stft(samples, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
            magnitude = np.mean(np.abs(Sxx), axis=1)  # Average across time = LTAS
            magnitude_db = 20 * np.log10(magnitude + 1e-10)

            # Spectral tilt: linear regression of dB spectrum in log-frequency
            # Only consider 100 Hz to Nyquist
            valid = (f > 100) & (f < sr / 2)
            if np.sum(valid) < 10:
                return {'spectral_tilt_score': 50, 'note': 'Insufficient frequency range'}

            log_f = np.log2(f[valid] + 1e-10)
            db_vals = magnitude_db[valid]

            # Linear regression
            A = np.vstack([log_f, np.ones(len(log_f))]).T
            slope, intercept = np.linalg.lstsq(A, db_vals, rcond=None)[0]

            # Slope is dB per octave (since we used log2)
            tilt_db_per_octave = float(slope)

            # Check for sharp high-frequency cutoff (vocoder artifact)
            # Energy ratio: 4-8 kHz vs 1-4 kHz
            if sr > 8000:
                band_low = (f >= 1000) & (f < 4000)
                band_high = (f >= 4000) & (f < min(8000, sr / 2))
                if np.sum(band_low) > 0 and np.sum(band_high) > 0:
                    ratio = np.mean(magnitude[band_high]) / (np.mean(magnitude[band_low]) + 1e-10)
                    hf_ratio = float(ratio)
                else:
                    hf_ratio = 0.5
            else:
                hf_ratio = 0.5

            # LTAS flatness (spectral entropy)
            mag_norm = magnitude[valid] / (np.sum(magnitude[valid]) + 1e-10)
            spectral_entropy = -float(np.sum(mag_norm * np.log2(mag_norm + 1e-20)))
            max_entropy = np.log2(len(mag_norm))
            flatness = spectral_entropy / (max_entropy + 1e-10)

            # Score: natural tilt is -4 to -8 dB/octave
            # Too flat (> -2) or too steep (< -12) is suspicious
            if -8 <= tilt_db_per_octave <= -3:
                tilt_score = 80
            elif -12 <= tilt_db_per_octave < -3 or -8 < tilt_db_per_octave <= -1:
                tilt_score = 60
            else:
                tilt_score = 35

            # Penalize sharp HF cutoff (vocoder bandwidth limit)
            if hf_ratio < 0.05:
                tilt_score = min(tilt_score, 30)

            return {
                'tilt_db_per_octave': tilt_db_per_octave,
                'hf_ratio': hf_ratio,
                'spectral_flatness': flatness,
                'spectral_tilt_score': tilt_score,
            }
        except Exception as e:
            logger.warning(f"Spectral tilt analysis failed: {e}")
            return {'spectral_tilt_score': 50, 'error': str(e)}

    # ------------------------------------------------------------------
    # 5. Background Noise Consistency
    # ------------------------------------------------------------------
    def _analyze_noise_consistency(self, samples: np.ndarray, sr: int) -> Dict[str, Any]:
        """Check noise floor consistency across segments.

        Real recordings have consistent ambient noise.  Spliced audio
        shows discontinuities in noise floor level and spectral shape.
        Also detects unnaturally LOW noise (TTS with almost no background).
        """
        try:
            frame_len = int(0.05 * sr)  # 50ms frames
            hop = frame_len // 2  # 50% overlap for better coverage

            # Compute RMS for ALL frames first
            all_rms = []
            for i in range(0, len(samples) - frame_len, hop):
                frame = samples[i:i + frame_len]
                all_rms.append(float(np.sqrt(np.mean(frame ** 2))))

            if len(all_rms) < 4:
                return {'noise_consistency_score': 50, 'note': 'Audio too short'}

            all_rms = np.array(all_rms)

            # Noise floor estimation: extract residual after median filtering
            # This captures the noise in ALL frames, not just quiet ones
            residual_levels = []
            residual_spectra = []
            half_frames = len(all_rms) // 2

            for segment_start, segment_end in [(0, half_frames), (half_frames, len(all_rms))]:
                seg_rms = all_rms[segment_start:segment_end]
                seg_frames = []
                for i in range(segment_start, min(segment_end, len(all_rms))):
                    start = i * hop
                    if start + frame_len > len(samples):
                        break
                    frame = samples[start:start + frame_len]
                    # High-pass filter to isolate noise from tonal content
                    if sr > 400:
                        sos = signal.butter(2, 200, btype='high', fs=sr, output='sos')
                        hp_frame = signal.sosfilt(sos, frame)
                    else:
                        hp_frame = frame
                    noise_rms = float(np.sqrt(np.mean(hp_frame ** 2)))
                    residual_levels.append(noise_rms)
                    spec = np.abs(rfft(hp_frame * np.hanning(frame_len)))
                    residual_spectra.append(spec / (np.max(spec) + 1e-10))

            residual_levels = np.array(residual_levels)
            noise_level_cv = float(np.std(residual_levels) / (np.mean(residual_levels) + 1e-10))

            # Check for step change in noise level (splice indicator)
            # Split into first half and second half
            mid = len(residual_levels) // 2
            if mid > 2:
                first_half_mean = np.mean(residual_levels[:mid])
                second_half_mean = np.mean(residual_levels[mid:])
                overall_mean = np.mean(residual_levels) + 1e-10
                step_change = abs(first_half_mean - second_half_mean) / overall_mean
            else:
                step_change = 0.0

            # Spectral shape consistency between halves
            if len(residual_spectra) >= 4:
                mid_s = len(residual_spectra) // 2
                first_avg = np.mean(residual_spectra[:mid_s], axis=0)
                second_avg = np.mean(residual_spectra[mid_s:], axis=0)
                min_len = min(len(first_avg), len(second_avg))
                if np.std(first_avg[:min_len]) > 1e-10 and np.std(second_avg[:min_len]) > 1e-10:
                    corr_matrix = np.corrcoef(
                        first_avg[:min_len], second_avg[:min_len]
                    )
                    spectral_consistency = float(corr_matrix[0, 1])
                    if np.isnan(spectral_consistency):
                        spectral_consistency = 1.0
                    spectral_consistency = max(0, spectral_consistency)
                else:
                    spectral_consistency = 1.0
            else:
                spectral_consistency = 0.5

            # Overall noise level (detect unnaturally quiet TTS)
            mean_noise = float(np.mean(residual_levels))
            too_quiet = mean_noise < 0.001  # Extremely low noise floor

            # Score: high consistency (low CV, high spectral corr) = natural
            # Low step change = no splice
            level_score = int(np.clip(100 - noise_level_cv * 60, 20, 100))
            step_penalty = min(30, int(step_change * 60))
            spectral_score = int(np.clip(spectral_consistency * 80 + 20, 0, 100))
            quiet_penalty = 20 if too_quiet else 0

            noise_consistency_score = int(np.clip(
                0.4 * level_score + 0.3 * spectral_score - step_penalty - quiet_penalty,
                0, 100
            ))

            return {
                'noise_level_cv': noise_level_cv,
                'spectral_consistency': spectral_consistency,
                'step_change': step_change,
                'mean_noise_level': mean_noise,
                'too_quiet': too_quiet,
                'noise_consistency_score': noise_consistency_score,
            }
        except Exception as e:
            logger.warning(f"Noise consistency analysis failed: {e}")
            return {'noise_consistency_score': 50, 'error': str(e)}

    # ------------------------------------------------------------------
    # 6. Mel-Spectrogram Regularity
    # ------------------------------------------------------------------
    def _analyze_mel_regularity(self, samples: np.ndarray, sr: int,
                                 stft_result: tuple = None) -> Dict[str, Any]:
        """Check frame-to-frame variability in mel spectrogram.

        TTS systems (Tacotron, VITS) produce unnaturally smooth
        mel-spectrograms.  Natural speech has higher frame-to-frame
        variability.  Measured by CV of delta-mel across bands.
        """
        try:
            n_fft = 1024
            n_mels = 40

            # Compute STFT
            if stft_result is not None:
                f, t, Zxx = stft_result
            else:
                hop = 256
                f, t, Zxx = signal.stft(samples, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
            power = np.abs(Zxx) ** 2

            # Apply mel filterbank
            fb = _mel_filterbank(sr, n_fft, n_mels)
            # Trim filterbank to match STFT frequency bins
            n_freq = power.shape[0]
            if fb.shape[1] > n_freq:
                fb = fb[:, :n_freq]
            elif fb.shape[1] < n_freq:
                pad = np.zeros((fb.shape[0], n_freq - fb.shape[1]))
                fb = np.hstack([fb, pad])

            mel_spec = np.dot(fb, power)
            mel_db = 10 * np.log10(mel_spec + 1e-10)

            if mel_db.shape[1] < 3:
                return {'mel_regularity_score': 50, 'note': 'Too few frames'}

            # Delta mel: frame-to-frame difference
            delta_mel = np.diff(mel_db, axis=1)

            # CV of delta across time for each mel band
            band_cvs = []
            for b in range(n_mels):
                band_delta = delta_mel[b, :]
                std = np.std(band_delta)
                mean_abs = np.mean(np.abs(band_delta)) + 1e-10
                band_cvs.append(std / mean_abs)

            avg_mel_cv = float(np.mean(band_cvs))

            # Also measure temporal regularity: autocorrelation of delta
            # High autocorrelation = rhythmic/mechanical = TTS
            temporal_autocorrs = []
            for b in range(n_mels):
                bd = delta_mel[b, :]
                if len(bd) > 10:
                    ac = np.correlate(bd - np.mean(bd), bd - np.mean(bd), mode='full')
                    ac = ac[len(ac) // 2:]
                    ac = ac / (ac[0] + 1e-10)
                    # Mean of first few lags (excluding 0)
                    temporal_autocorrs.append(float(np.mean(np.abs(ac[1:min(6, len(ac))]))))

            avg_temporal_autocorr = float(np.mean(temporal_autocorrs)) if temporal_autocorrs else 0.3

            # Primary metric: temporal autocorrelation (TTS is MORE regular)
            # Low autocorrelation = natural variability = high score
            # High autocorrelation = mechanical regularity = low score (TTS)
            if avg_temporal_autocorr < 0.15:
                autocorr_score = 80  # Natural — low temporal regularity
            elif avg_temporal_autocorr < 0.3:
                autocorr_score = int(80 - (avg_temporal_autocorr - 0.15) / 0.15 * 30)
            elif avg_temporal_autocorr < 0.5:
                autocorr_score = int(50 - (avg_temporal_autocorr - 0.3) / 0.2 * 20)
            else:
                autocorr_score = int(np.clip(30 - (avg_temporal_autocorr - 0.5) * 40, 10, 30))

            # Secondary: mel CV (less discriminative at high values, useful at extremes)
            if avg_mel_cv < 0.2:
                cv_bonus = -15  # Unnaturally smooth — TTS
            elif avg_mel_cv > 2.0:
                cv_bonus = 0  # Normal or high variability
            else:
                cv_bonus = 0

            mel_regularity_score = int(np.clip(autocorr_score + cv_bonus, 0, 100))

            return {
                'mel_cv': avg_mel_cv,
                'temporal_autocorr': avg_temporal_autocorr,
                'mel_regularity_score': mel_regularity_score,
            }
        except Exception as e:
            logger.warning(f"Mel regularity analysis failed: {e}")
            return {'mel_regularity_score': 50, 'error': str(e)}

    # ------------------------------------------------------------------
    # 7. Formant Bandwidth Analysis
    # ------------------------------------------------------------------
    def _analyze_formant_bandwidth(self, samples: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze formant structure via LPC spectral envelope.

        Natural vocal tract produces formants (F1-F4) with characteristic
        bandwidths and inter-formant valley depths (>15 dB natural,
        <10 dB for vocoders producing "buzzy" or flat formants).
        """
        try:
            # Downsample to ~16 kHz if needed for formant analysis
            target_sr = min(sr, 16000)
            if sr > target_sr:
                ratio = sr // target_sr
                ds_samples = signal.decimate(samples, ratio)
                ds_sr = sr // ratio
            else:
                ds_samples = samples
                ds_sr = sr

            # LPC analysis on voiced frames
            frame_len = int(0.025 * ds_sr)  # 25ms
            hop = int(0.01 * ds_sr)
            lpc_order = 2 + ds_sr // 1000  # Rule of thumb

            valley_depths = []
            formant_bws = []

            for i in range(0, len(ds_samples) - frame_len, hop):
                frame = ds_samples[i:i + frame_len]
                if np.max(np.abs(frame)) < 0.01:
                    continue

                # Pre-emphasis
                frame_pe = np.append(frame[0], frame[1:] - 0.97 * frame[:-1])
                # Window
                windowed = frame_pe * np.hamming(len(frame_pe))

                # Autocorrelation method for LPC
                autocorr = np.correlate(windowed, windowed, mode='full')
                autocorr = autocorr[len(autocorr) // 2:]

                # Levinson-Durbin
                try:
                    a_coeffs = self._levinson_durbin(autocorr[:lpc_order + 1], lpc_order)
                except Exception:
                    continue

                # LPC spectral envelope
                n_points = 512
                w, h = signal.freqz([1.0], a_coeffs, worN=n_points, fs=ds_sr)
                lpc_spectrum = 20 * np.log10(np.abs(h) + 1e-10)

                # Find peaks (formants) and valleys
                peaks, peak_props = signal.find_peaks(lpc_spectrum, distance=20, prominence=3)
                if len(peaks) >= 2:
                    # Valley between first two formants
                    p1, p2 = peaks[0], peaks[1]
                    valley = np.min(lpc_spectrum[p1:p2 + 1])
                    peak_avg = (lpc_spectrum[p1] + lpc_spectrum[p2]) / 2
                    depth = peak_avg - valley
                    valley_depths.append(float(depth))

                    # Formant bandwidths from LPC roots
                    roots = np.roots(a_coeffs)
                    roots = roots[np.imag(roots) > 0]
                    if len(roots) > 0:
                        angles = np.angle(roots)
                        formant_freqs = angles * ds_sr / (2 * np.pi)
                        bws = -ds_sr / (2 * np.pi) * np.log(np.abs(roots) + 1e-10)
                        # Filter to speech range
                        valid_mask = (formant_freqs > 200) & (formant_freqs < 5000)
                        if np.any(valid_mask):
                            formant_bws.extend(bws[valid_mask].tolist())

            if not valley_depths:
                return {
                    'formant_score': 50,
                    'note': 'No formant structure detected'
                }

            avg_valley_depth = float(np.mean(valley_depths))
            avg_bw = float(np.mean(formant_bws)) if formant_bws else 100.0

            # Score: deeper valleys (>15 dB) = natural, shallow (<8 dB) = vocoder
            if avg_valley_depth > 15:
                valley_score = 85
            elif avg_valley_depth > 8:
                valley_score = int(40 + (avg_valley_depth - 8) / 7 * 45)
            else:
                valley_score = int(np.clip(avg_valley_depth / 8 * 40, 10, 40))

            # Bandwidth: natural 50-200 Hz, vocoder too narrow (<30) or too wide (>300)
            if 40 <= avg_bw <= 250:
                bw_score = 80
            elif 20 <= avg_bw < 40 or 250 < avg_bw <= 400:
                bw_score = 50
            else:
                bw_score = 30

            formant_score = int(0.6 * valley_score + 0.4 * bw_score)

            return {
                'avg_valley_depth': avg_valley_depth,
                'avg_formant_bw': avg_bw,
                'n_analyzed_frames': len(valley_depths),
                'formant_score': formant_score,
            }
        except Exception as e:
            logger.warning(f"Formant bandwidth analysis failed: {e}")
            return {'formant_score': 50, 'error': str(e)}

    @staticmethod
    def _levinson_durbin(r: np.ndarray, order: int) -> np.ndarray:
        """Levinson-Durbin recursion for LPC coefficients."""
        a = np.zeros(order + 1)
        a[0] = 1.0
        e = r[0]

        for i in range(1, order + 1):
            acc = sum(a[j] * r[i - j] for j in range(1, i))
            k = -(r[i] + acc) / (e + 1e-10)
            a_new = a.copy()
            for j in range(1, i):
                a_new[j] = a[j] + k * a[i - j]
            a_new[i] = k
            a = a_new
            e = e * (1 - k * k)
            if e <= 0:
                break
        return a

    # ------------------------------------------------------------------
    # 8. Double Compression Detection
    # ------------------------------------------------------------------
    def _analyze_double_compression(self, samples: np.ndarray, sr: int) -> Dict[str, Any]:
        """Detect double compression artifacts in MDCT-like domain.

        Re-encoding audio (e.g. MP3 → WAV → MP3) creates periodic
        patterns in transform coefficient histograms.  Single-encoded
        audio has smooth coefficient distributions.
        """
        try:
            # Analyze in MDCT-like blocks (simulate with windowed DCT)
            block_size = 1024
            hop = 512
            n_blocks = (len(samples) - block_size) // hop

            if n_blocks < 5:
                return {'double_comp_score': 50, 'note': 'Audio too short'}

            from scipy.fft import dct

            coeff_histograms = []
            for i in range(n_blocks):
                start = i * hop
                block = samples[start:start + block_size]
                windowed = block * np.hanning(block_size)
                coeffs = dct(windowed, type=2)
                coeff_histograms.append(coeffs)

            coeff_matrix = np.array(coeff_histograms)

            # Check for periodic patterns in coefficient magnitudes
            # Double compression creates periodic dips at quantization boundaries
            avg_coeff_magnitude = np.mean(np.abs(coeff_matrix), axis=0)

            # Compute autocorrelation of coefficient magnitude pattern
            acm = avg_coeff_magnitude - np.mean(avg_coeff_magnitude)
            autocorr = np.correlate(acm, acm, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)

            # Look for periodic peaks (double compression signature)
            peaks, _ = signal.find_peaks(autocorr[1:], height=0.1, distance=5)
            periodic_peak_count = len(peaks)

            # Histogram uniformity: double compression creates non-uniform
            # quantization patterns
            hist_scores = []
            for band_start in range(0, min(block_size, 512), 64):
                band = coeff_matrix[:, band_start:band_start + 64].flatten()
                hist, _ = np.histogram(band, bins=50)
                hist_norm = hist / (np.sum(hist) + 1e-10)
                # Entropy of histogram (lower = more quantized = double compression)
                entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-20))
                max_ent = np.log2(50)
                hist_scores.append(entropy / max_ent)

            avg_hist_uniformity = float(np.mean(hist_scores))

            # Score: high uniformity + few periodic peaks = single compression
            uniformity_score = int(np.clip(avg_hist_uniformity * 100, 0, 100))
            periodicity_penalty = min(30, periodic_peak_count * 5)
            double_comp_score = int(np.clip(uniformity_score - periodicity_penalty, 0, 100))

            return {
                'hist_uniformity': avg_hist_uniformity,
                'periodic_peaks': periodic_peak_count,
                'double_comp_score': double_comp_score,
            }
        except Exception as e:
            logger.warning(f"Double compression analysis failed: {e}")
            return {'double_comp_score': 50, 'error': str(e)}

    # ------------------------------------------------------------------
    # 9. Spectral Discontinuity Detection (Splice Points)
    # ------------------------------------------------------------------
    def _analyze_spectral_discontinuity(self, samples: np.ndarray, sr: int) -> Dict[str, Any]:
        """Detect splice points via abrupt changes in spectral features.

        Audio splices create discontinuities in spectral centroid,
        spectral flux, and zero-crossing rate at edit boundaries.
        """
        try:
            frame_len = int(0.025 * sr)  # 25ms
            hop = int(0.010 * sr)  # 10ms
            n_frames = (len(samples) - frame_len) // hop

            if n_frames < 10:
                return {'splice_score': 50, 'note': 'Audio too short'}

            centroids = []
            fluxes = []
            zcrs = []
            prev_spec = None

            for i in range(n_frames):
                start = i * hop
                frame = samples[start:start + frame_len]
                windowed = frame * np.hanning(frame_len)

                spec = np.abs(rfft(windowed))
                freqs = rfftfreq(frame_len, 1.0 / sr)

                # Spectral centroid
                centroid = np.sum(freqs * spec) / (np.sum(spec) + 1e-10)
                centroids.append(centroid)

                # Spectral flux
                if prev_spec is not None:
                    min_len = min(len(spec), len(prev_spec))
                    flux = np.sum((spec[:min_len] - prev_spec[:min_len]) ** 2)
                    fluxes.append(flux)
                prev_spec = spec

                # Zero crossing rate
                zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
                zcrs.append(zcr)

            centroids = np.array(centroids)
            fluxes = np.array(fluxes) if fluxes else np.array([0.0])
            zcrs = np.array(zcrs)

            # Detect discontinuities via z-score of spectral flux
            if len(fluxes) > 3:
                flux_mean = np.mean(fluxes)
                flux_std = np.std(fluxes) + 1e-10
                flux_zscore = (fluxes - flux_mean) / flux_std
                max_flux_z = float(np.max(np.abs(flux_zscore)))
                n_spikes = int(np.sum(np.abs(flux_zscore) > 3.0))
            else:
                max_flux_z = 0.0
                n_spikes = 0

            # Centroid stability
            centroid_cv = float(np.std(centroids) / (np.mean(centroids) + 1e-10))

            # ZCR stability
            zcr_cv = float(np.std(zcrs) / (np.mean(zcrs) + 1e-10))

            # Score: low discontinuity = natural, high = splice
            spike_penalty = min(50, n_spikes * 20)
            cv_penalty = min(25, max(0, (centroid_cv - 0.15) * 80))
            splice_score = int(np.clip(100 - spike_penalty - cv_penalty, 0, 100))

            return {
                'max_flux_zscore': max_flux_z,
                'n_flux_spikes': n_spikes,
                'centroid_cv': centroid_cv,
                'zcr_cv': zcr_cv,
                'splice_score': splice_score,
            }
        except Exception as e:
            logger.warning(f"Spectral discontinuity analysis failed: {e}")
            return {'splice_score': 50, 'error': str(e)}

    # ------------------------------------------------------------------
    # 10. Sub-band Energy Analysis
    # ------------------------------------------------------------------
    def _analyze_subband_energy(self, samples: np.ndarray, sr: int,
                                 stft_result: tuple = None) -> Dict[str, Any]:
        """Analyze energy distribution across frequency sub-bands.

        Voice cloning systems show artifacts in specific bands:
          0-500 Hz:   Unnaturally consistent harmonics
          500-4000:   Correct formant freqs but wrong bandwidths
          4000-8000:  Incorrect sibilant shape
          8000+:      Missing energy or band replication
        """
        try:
            if stft_result is not None:
                f, t, Sxx = stft_result
            else:
                n_fft = 2048
                hop = 512
                f, t, Sxx = signal.stft(samples, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
            power = np.abs(Sxx) ** 2

            nyquist = sr / 2

            # Define sub-bands (adjusted for sample rate)
            bands = []
            band_defs = [
                ('low', 0, 500),
                ('mid_low', 500, 2000),
                ('mid', 2000, 4000),
                ('mid_high', 4000, 8000),
                ('high', 8000, nyquist),
            ]

            band_energies = {}
            band_variabilities = {}

            for name, lo, hi in band_defs:
                if lo >= nyquist:
                    continue
                hi = min(hi, nyquist)
                mask = (f >= lo) & (f < hi)
                if not np.any(mask):
                    continue

                band_power = power[mask, :]
                # Energy per frame
                frame_energies = np.sum(band_power, axis=0)
                band_energies[name] = float(np.mean(frame_energies))

                # Temporal variability within band
                if len(frame_energies) > 1:
                    cv = float(np.std(frame_energies) / (np.mean(frame_energies) + 1e-10))
                    band_variabilities[name] = cv
                else:
                    band_variabilities[name] = 0.0

            if not band_energies:
                return {'subband_score': 50, 'note': 'No bands analyzed'}

            total_energy = sum(band_energies.values()) + 1e-10
            band_ratios = {k: v / total_energy for k, v in band_energies.items()}

            # Check for vocoder cutoff: sharp drop in high bands
            has_hf_cutoff = False
            if 'mid_high' in band_ratios and 'mid' in band_ratios:
                if band_ratios.get('mid_high', 0) < band_ratios['mid'] * 0.05:
                    has_hf_cutoff = True
            if 'high' in band_ratios and 'mid_high' in band_ratios:
                if band_ratios.get('high', 0) < band_ratios.get('mid_high', 1) * 0.01:
                    has_hf_cutoff = True

            # Check for unnaturally consistent low band (TTS harmonic regularity)
            low_cv = band_variabilities.get('low', 0.3)
            mid_cv = band_variabilities.get('mid', 0.3)

            # Natural speech: low band CV 0.3-1.0+, TTS: < 0.15
            low_band_too_stable = low_cv < 0.15

            # Score components
            cutoff_penalty = 25 if has_hf_cutoff else 0
            stability_penalty = 20 if low_band_too_stable else 0

            # Energy distribution: natural speech has most energy in mid_low
            mid_low_ratio = band_ratios.get('mid_low', 0)
            distribution_score = 70 if 0.2 < mid_low_ratio < 0.6 else 50

            subband_score = int(np.clip(
                distribution_score - cutoff_penalty - stability_penalty, 0, 100
            ))

            return {
                'band_ratios': band_ratios,
                'band_variabilities': band_variabilities,
                'hf_cutoff_detected': has_hf_cutoff,
                'low_band_cv': low_cv,
                'subband_score': subband_score,
            }
        except Exception as e:
            logger.warning(f"Sub-band energy analysis failed: {e}")
            return {'subband_score': 50, 'error': str(e)}

    # ------------------------------------------------------------------
    # Composite scoring
    # ------------------------------------------------------------------
    def _calculate_composite_score(self, results: Dict[str, Any]) -> int:
        """Weighted average of all analysis scores."""
        score_key_map = {
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

        total_score = 0.0
        total_weight = 0.0

        for method, weight in self.WEIGHTS.items():
            sub = results.get(method, {})
            if isinstance(sub, dict):
                score_key = score_key_map.get(method)
                if score_key and score_key in sub:
                    score = sub[score_key]
                    if isinstance(score, (int, float)):
                        total_score += score * weight
                        total_weight += weight

        if total_weight > 0:
            return int(np.clip(total_score / total_weight, 0, 100))
        return 50

    # ------------------------------------------------------------------
    # AI indicator counting
    # ------------------------------------------------------------------
    # Declarative indicator table for simple threshold checks.
    # Each entry: (result_key, metric_key, op, threshold_or_key, default, weight)
    # String threshold → looked up from self.THRESHOLDS at runtime.
    AUDIO_INDICATOR_TABLE = [
        ('phase_coherence', 'gdd_std', '<', 'phase_gdd_std_low', 1.0, 1),
        ('voice_quality', 'jitter', '<', 'jitter_low', 0.005, 1),
        ('voice_quality', 'shimmer', '<', 'shimmer_low', 0.03, 1),
        ('voice_quality', 'hnr', '>', 'hnr_high', 18.0, 1),
        ('mel_regularity', 'mel_cv', '<', 'mel_cv_low', 0.3, 1),
        ('spectral_tilt', 'tilt_db_per_octave', '>', 'spectral_tilt_flat', -5.0, 1),
        ('noise_consistency', 'noise_level_cv', '>', 'noise_floor_cv_high', 0.3, 1),
        ('formant_bandwidth', 'avg_valley_depth', '<', 'formant_valley_low', 12.0, 1),
        ('spectral_discontinuity', 'n_flux_spikes', '>=', 2, 0, 1),
        ('subband_energy', 'low_band_cv', '<', 0.12, 0.3, 1),
        ('double_compression', 'periodic_peaks', '>', 5, 0, 1),
        ('noise_consistency', 'step_change', '>', 0.5, 0.0, 1),
        ('voice_quality', 'voice_quality_score', '<', 15, 50, 1),
    ]

    def _count_ai_indicators(self, results: Dict[str, Any]) -> int:
        """Count indicators of AI-generated audio."""
        indicators = 0

        # Simple threshold checks from declarative table
        for result_key, metric_key, op, threshold, default, weight in self.AUDIO_INDICATOR_TABLE:
            val = results.get(result_key, {}).get(metric_key, default)
            thresh = self.THRESHOLDS[threshold] if isinstance(threshold, str) else threshold
            if op == '<' and val < thresh:
                indicators += weight
            elif op == '>' and val > thresh:
                indicators += weight
            elif op == '>=' and val >= thresh:
                indicators += weight

        # Compound: ENF absent
        enf_detected = results.get('enf_analysis', {}).get('enf_detected', True)
        enf_snr = results.get('enf_analysis', {}).get('enf_snr', 5.0)
        if not enf_detected and enf_snr < 1.5:
            indicators += 1

        # Boolean: HF cutoff
        hf_cutoff = results.get('subband_energy', {}).get('hf_cutoff_detected', False)
        if hf_cutoff:
            indicators += 1

        # Compound: low jitter + low shimmer + high HNR = strong TTS
        jitter = results.get('voice_quality', {}).get('jitter', 0.005)
        shimmer = results.get('voice_quality', {}).get('shimmer', 0.03)
        hnr = results.get('voice_quality', {}).get('hnr', 18.0)
        if (jitter < self.THRESHOLDS['jitter_low'] and
                shimmer < self.THRESHOLDS['shimmer_low'] and
                hnr > self.THRESHOLDS['hnr_high']):
            indicators += 1

        # Vocoder pattern: high jitter + very low HNR
        if jitter > 0.05 and hnr < 3:
            indicators += 2

        # Perfect-pitch non-speech
        voiced = results.get('voice_quality', {}).get('voiced_frames', 0)
        if jitter < 0.0005 and shimmer < 0.005 and voiced > 50:
            indicators += 2

        return indicators

    # ------------------------------------------------------------------
    # Result determination
    # ------------------------------------------------------------------
    def _determine_result(
        self,
        score: int,
        ai_probability: float,
        results: Dict[str, Any]
    ) -> Tuple[bool, int, Optional[str]]:
        """Determine verification result based on score and probability."""

        # --- Targeted detection rules ---

        # Vocoder pattern: high jitter + very low HNR = random phase reconstruction
        jitter = results.get('voice_quality', {}).get('jitter', 0.005)
        hnr = results.get('voice_quality', {}).get('hnr', 18.0)
        if jitter > 0.05 and hnr < 3:
            warning = 'Phase reconstruction artifacts detected - likely vocoder output'
            adjusted = int(score * 0.5)
            return False, adjusted, warning

        # Splice detection: spectral flux spikes + noise inconsistency
        n_spikes = results.get('spectral_discontinuity', {}).get('n_flux_spikes', 0)
        noise_step = results.get('noise_consistency', {}).get('step_change', 0.0)
        noise_cv = results.get('noise_consistency', {}).get('noise_level_cv', 0.3)
        if (n_spikes >= 2 and noise_step > 0.3) or (n_spikes >= 3 and noise_cv > 0.4):
            warning = 'Audio splice detected - likely manipulated'
            adjusted = int(score * 0.6)
            return adjusted >= 50, adjusted, warning

        # --- General probability-based brackets ---
        if ai_probability >= 0.4:
            warning = 'Strong indicators of AI-generated audio'
            adjusted = int(score * 0.5)
            return False, adjusted, warning

        elif ai_probability >= 0.3:
            warning = 'Some indicators of AI-generated audio'
            adjusted = int(score * 0.7)
            return adjusted >= 50, adjusted, warning

        elif ai_probability >= 0.15:
            warning = 'Minor audio inconsistencies detected'
            passed = score >= 50
            return passed, score, warning if not passed else None

        else:
            passed = score >= 40
            return passed, score, None
