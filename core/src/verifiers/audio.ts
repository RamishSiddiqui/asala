/**
 * Audio Physics-based Verification for AI-generated Audio Detection.
 *
 * Implements 10 mathematical analysis techniques to detect AI-generated or
 * manipulated audio based on physical and acoustic inconsistencies.
 * All methods use pure JS signal processing — no neural networks or training data.
 *
 * Methods:
 *   1. Phase Coherence Analysis      — vocoder detection via inter-frame coherence
 *   2. Voice Quality Metrics         — jitter, shimmer, HNR
 *   3. ENF Analysis                  — 50/60 Hz electrical network frequency
 *   4. Spectral Tilt / LTAS          — long-term spectral envelope shape
 *   5. Background Noise Consistency  — noise floor stability across segments
 *   6. Mel-Spectrogram Regularity    — frame-to-frame variability in mel bands
 *   7. Formant Bandwidth Analysis    — LPC-based formant valley depth
 *   8. Double Compression Detection  — re-encoding artifact patterns
 *   9. Spectral Discontinuity        — splice point detection via spectral flux
 *  10. Sub-band Energy Analysis      — frequency-band energy distribution
 */

import { LayerResult } from '../types';
import { fft1d } from '../imaging/processing';
import { mean, std, variance, clamp } from '../imaging/stats';

// ---------------------------------------------------------------------------
// WAV decoder (pure JS — supports 8/16/24/32-bit PCM)
// ---------------------------------------------------------------------------

interface DecodedAudio {
  samples: Float64Array; // mono float64 in [-1, 1]
  sampleRate: number;
}

function decodeWav(buffer: Buffer): DecodedAudio {
  // Parse RIFF header
  if (buffer.toString('ascii', 0, 4) !== 'RIFF') {
    throw new Error('Not a valid WAV file (missing RIFF header)');
  }
  if (buffer.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error('Not a valid WAV file (missing WAVE header)');
  }

  let offset = 12;
  let fmtFound = false;
  let sampleRate = 0;
  let numChannels = 1;
  let bitsPerSample = 16;
  let dataBuffer: Buffer | null = null;

  while (offset < buffer.length - 8) {
    const chunkId = buffer.toString('ascii', offset, offset + 4);
    const chunkSize = buffer.readUInt32LE(offset + 4);
    offset += 8;

    if (chunkId === 'fmt ') {
      numChannels = buffer.readUInt16LE(offset + 2);
      sampleRate = buffer.readUInt32LE(offset + 4);
      bitsPerSample = buffer.readUInt16LE(offset + 14);
      fmtFound = true;
    } else if (chunkId === 'data') {
      dataBuffer = buffer.subarray(offset, offset + chunkSize);
    }
    offset += chunkSize;
    // Align to 2-byte boundary
    if (chunkSize % 2 !== 0) offset++;
  }

  if (!fmtFound || !dataBuffer) {
    throw new Error('Invalid WAV: missing fmt or data chunk');
  }

  const bytesPerSample = bitsPerSample / 8;
  const totalSamples = Math.floor(dataBuffer.length / bytesPerSample);

  // Decode raw samples to float
  const rawSamples = new Float64Array(totalSamples);
  for (let i = 0; i < totalSamples; i++) {
    const pos = i * bytesPerSample;
    if (bitsPerSample === 8) {
      rawSamples[i] = dataBuffer[pos] / 128.0 - 1.0;
    } else if (bitsPerSample === 16) {
      rawSamples[i] = dataBuffer.readInt16LE(pos) / 32768.0;
    } else if (bitsPerSample === 24) {
      const low = dataBuffer[pos];
      const mid = dataBuffer[pos + 1];
      const high = dataBuffer[pos + 2];
      let val = (high << 16) | (mid << 8) | low;
      if (val & 0x800000) val |= ~0xFFFFFF; // Sign extend
      rawSamples[i] = val / 8388608.0;
    } else if (bitsPerSample === 32) {
      rawSamples[i] = dataBuffer.readInt32LE(pos) / 2147483648.0;
    }
  }

  // Convert to mono by averaging channels
  if (numChannels > 1) {
    const monoLen = Math.floor(totalSamples / numChannels);
    const mono = new Float64Array(monoLen);
    for (let i = 0; i < monoLen; i++) {
      let sum = 0;
      for (let ch = 0; ch < numChannels; ch++) {
        sum += rawSamples[i * numChannels + ch];
      }
      mono[i] = sum / numChannels;
    }
    return { samples: mono, sampleRate };
  }

  return { samples: rawSamples, sampleRate };
}

// ---------------------------------------------------------------------------
// Helper: mel filterbank
// ---------------------------------------------------------------------------

function melFilterbank(sr: number, nFft: number, nMels = 40): Float64Array[] {
  const hzToMel = (hz: number) => 2595.0 * Math.log10(1.0 + hz / 700.0);
  const melToHz = (mel: number) => 700.0 * (Math.pow(10, mel / 2595.0) - 1.0);

  const nBins = Math.floor(nFft / 2) + 1;
  const lowMel = hzToMel(0);
  const highMel = hzToMel(sr / 2);

  const melPoints = new Float64Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) {
    melPoints[i] = lowMel + (i * (highMel - lowMel)) / (nMels + 1);
  }
  const hzPoints = new Float64Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) {
    hzPoints[i] = melToHz(melPoints[i]);
  }
  const binPoints = new Int32Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) {
    binPoints[i] = Math.floor((nFft + 1) * hzPoints[i] / sr);
  }

  const fb: Float64Array[] = [];
  for (let m = 1; m <= nMels; m++) {
    const row = new Float64Array(nBins);
    const fLeft = binPoints[m - 1];
    const fCenter = binPoints[m];
    const fRight = binPoints[m + 1];
    for (let k = fLeft; k < fCenter; k++) {
      if (k < nBins && fCenter > fLeft) {
        row[k] = (k - fLeft) / (fCenter - fLeft);
      }
    }
    for (let k = fCenter; k < fRight; k++) {
      if (k < nBins && fRight > fCenter) {
        row[k] = (fRight - k) / (fRight - fCenter);
      }
    }
    fb.push(row);
  }
  return fb;
}

// ---------------------------------------------------------------------------
// Helper: STFT using our FFT
// ---------------------------------------------------------------------------

interface StftResult {
  magnitudes: Float64Array[]; // one row per frequency bin, one column per time frame
  phases: Float64Array[];
  freqs: Float64Array;
  nFrames: number;
  nFreqs: number;
}

function stft(
  samples: Float64Array,
  sr: number,
  nFft: number,
  hop: number
): StftResult {
  const nFrames = Math.floor((samples.length - nFft) / hop) + 1;
  const nFreqs = Math.floor(nFft / 2) + 1;

  // Hanning window
  const win = new Float64Array(nFft);
  for (let i = 0; i < nFft; i++) {
    win[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (nFft - 1)));
  }

  // Pad nFft to next power of 2 for FFT
  let fftSize = 1;
  while (fftSize < nFft) fftSize <<= 1;

  const magnitudes: Float64Array[] = [];
  const phases: Float64Array[] = [];
  for (let f = 0; f < nFreqs; f++) {
    magnitudes.push(new Float64Array(nFrames));
    phases.push(new Float64Array(nFrames));
  }

  // Pre-allocate FFT buffers (reused across all frames)
  const re = new Float64Array(fftSize);
  const im = new Float64Array(fftSize);

  for (let t = 0; t < nFrames; t++) {
    const start = t * hop;
    // Clear and fill buffers
    re.fill(0);
    im.fill(0);
    for (let i = 0; i < nFft; i++) {
      re[i] = samples[start + i] * win[i];
    }
    fft1d(re, im);

    for (let f = 0; f < nFreqs; f++) {
      magnitudes[f][t] = Math.sqrt(re[f] * re[f] + im[f] * im[f]);
      phases[f][t] = Math.atan2(im[f], re[f]);
    }
  }

  // Frequency axis
  const freqs = new Float64Array(nFreqs);
  for (let f = 0; f < nFreqs; f++) {
    freqs[f] = (f * sr) / fftSize;
  }

  return { magnitudes, phases, freqs, nFrames, nFreqs };
}

// ---------------------------------------------------------------------------
// Helper: rfft (real FFT — returns magnitude spectrum)
// ---------------------------------------------------------------------------

function rfft(samples: Float64Array, nFft: number): Float64Array {
  let fftSize = 1;
  while (fftSize < nFft) fftSize <<= 1;
  const re = new Float64Array(fftSize);
  const im = new Float64Array(fftSize);
  for (let i = 0; i < nFft; i++) re[i] = samples[i];
  fft1d(re, im);
  const nFreqs = Math.floor(fftSize / 2) + 1;
  const mag = new Float64Array(nFreqs);
  for (let i = 0; i < nFreqs; i++) {
    mag[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
  }
  return mag;
}

// ---------------------------------------------------------------------------
// Helper: DCT via FFT
// ---------------------------------------------------------------------------

function dctType2(input: Float64Array): Float64Array {
  const N = input.length;
  let fftSize = 1;
  while (fftSize < 2 * N) fftSize <<= 1;
  const re = new Float64Array(fftSize);
  const im = new Float64Array(fftSize);
  // Mirror: [x0, x1, ..., xN-1, xN-1, ..., x1, x0]
  for (let i = 0; i < N; i++) {
    re[2 * i] = input[i];
    re[2 * i + 1] = input[N - 1 - i];
  }
  fft1d(re, im);
  const result = new Float64Array(N);
  for (let k = 0; k < N; k++) {
    const angle = (-Math.PI * k) / (2 * N);
    result[k] = re[k] * Math.cos(angle) - im[k] * Math.sin(angle);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Helper: find peaks in an array
// ---------------------------------------------------------------------------

function findPeaks(arr: Float64Array, options: { height?: number; distance?: number } = {}): number[] {
  const { height = -Infinity, distance = 1 } = options;
  const peaks: number[] = [];
  for (let i = 1; i < arr.length - 1; i++) {
    if (arr[i] > arr[i - 1] && arr[i] > arr[i + 1] && arr[i] >= height) {
      if (peaks.length === 0 || i - peaks[peaks.length - 1] >= distance) {
        peaks.push(i);
      }
    }
  }
  return peaks;
}

// ---------------------------------------------------------------------------
// Helper: Levinson-Durbin recursion for LPC
// ---------------------------------------------------------------------------

function levinsonDurbin(r: Float64Array, order: number): Float64Array {
  const a = new Float64Array(order + 1);
  a[0] = 1.0;
  let e = r[0];

  for (let i = 1; i <= order; i++) {
    let acc = 0;
    for (let j = 1; j < i; j++) acc += a[j] * r[i - j];
    const k = -(r[i] + acc) / (e + 1e-10);
    const aNew = new Float64Array(order + 1);
    aNew[0] = 1.0;
    for (let j = 1; j < i; j++) aNew[j] = a[j] + k * a[i - j];
    aNew[i] = k;
    for (let j = 0; j <= order; j++) a[j] = aNew[j];
    e = e * (1 - k * k);
    if (e <= 0) break;
  }
  return a;
}

// ---------------------------------------------------------------------------
// Weight and threshold configuration
// ---------------------------------------------------------------------------

const WEIGHTS: Record<string, number> = {
  phase_coherence: 0.08,
  voice_quality: 0.22,
  enf_analysis: 0.05,
  spectral_tilt: 0.10,
  noise_consistency: 0.12,
  mel_regularity: 0.10,
  formant_bandwidth: 0.10,
  double_compression: 0.06,
  spectral_discontinuity: 0.09,
  subband_energy: 0.08,
};

const SCORE_KEY_MAP: Record<string, string> = {
  phase_coherence: 'phase_score',
  voice_quality: 'voice_quality_score',
  enf_analysis: 'enf_score',
  spectral_tilt: 'spectral_tilt_score',
  noise_consistency: 'noise_consistency_score',
  mel_regularity: 'mel_regularity_score',
  formant_bandwidth: 'formant_score',
  double_compression: 'double_comp_score',
  spectral_discontinuity: 'splice_score',
  subband_energy: 'subband_score',
};

// ---------------------------------------------------------------------------
// AudioVerifier
// ---------------------------------------------------------------------------

export class AudioVerifier {
  verifyAudio(audioBuffer: Buffer): LayerResult {
    try {
      const { samples, sampleRate: sr } = decodeWav(audioBuffer);

      if (samples.length < sr * 0.1) {
        return {
          name: 'Physics Verification (Audio)',
          passed: false,
          score: 0,
          details: { error: 'Audio too short for analysis (< 0.1s)' },
        };
      }

      const results: Record<string, Record<string, unknown>> = {};

      results['phase_coherence'] = this.analyzePhaseCoherence(samples, sr);
      results['voice_quality'] = this.analyzeVoiceQuality(samples, sr);
      results['enf_analysis'] = this.analyzeEnf(samples, sr);
      results['spectral_tilt'] = this.analyzeSpectralTilt(samples, sr);
      results['noise_consistency'] = this.analyzeNoiseConsistency(samples, sr);
      results['mel_regularity'] = this.analyzeMelRegularity(samples, sr);
      results['formant_bandwidth'] = this.analyzeFormantBandwidth(samples, sr);
      results['double_compression'] = this.analyzeDoubleCompression(samples, sr);
      results['spectral_discontinuity'] = this.analyzeSpectralDiscontinuity(samples, sr);
      results['subband_energy'] = this.analyzeSubbandEnergy(samples, sr);

      const compositeScore = this.calculateCompositeScore(results);
      const aiIndicators = this.countAiIndicators(results);
      const totalIndicators = 18;
      const aiProbability = aiIndicators / totalIndicators;

      const { passed, adjustedScore, warning } = this.determineResult(
        compositeScore, aiProbability, results
      );

      const details: Record<string, unknown> = { ...results };
      if (warning) details['warning'] = warning;
      details['ai_probability'] = aiProbability;
      details['ai_indicators'] = aiIndicators;

      return {
        name: 'Physics Verification (Audio)',
        passed,
        score: adjustedScore,
        details,
      };
    } catch (err) {
      return {
        name: 'Physics Verification (Audio)',
        passed: false,
        score: 0,
        details: { error: String(err) },
      };
    }
  }

  // =========================================================================
  // 1. Phase Coherence Analysis
  // =========================================================================

  private analyzePhaseCoherence(
    samples: Float64Array, sr: number
  ): Record<string, number> {
    try {
      const nFft = 1024;
      const hop = 256;
      const s = stft(samples, sr, nFft, hop);

      // Focus on speech-critical band (300-4000 Hz)
      const bandIndices: number[] = [];
      for (let f = 0; f < s.nFreqs; f++) {
        if (s.freqs[f] >= 300 && s.freqs[f] <= Math.min(4000, sr / 2)) {
          bandIndices.push(f);
        }
      }
      if (bandIndices.length < 5 || s.nFrames < 3) {
        return { phase_score: 50 };
      }

      // Inter-frame phase coherence per frequency bin
      const mrlPerBin: number[] = [];
      for (const fi of bandIndices) {
        let sumCos = 0, sumSin = 0;
        for (let t = 1; t < s.nFrames; t++) {
          const diff = s.phases[fi][t] - s.phases[fi][t - 1];
          sumCos += Math.cos(diff);
          sumSin += Math.sin(diff);
        }
        const n = s.nFrames - 1;
        const mrl = Math.sqrt(sumCos * sumCos + sumSin * sumSin) / n;
        mrlPerBin.push(mrl);
      }

      const avgCoherence = mean(mrlPerBin);
      const coherenceStd = std(mrlPerBin);

      // GDD for additional info
      let gddStdSum = 0;
      for (let t = 0; t < s.nFrames; t++) {
        const diffs: number[] = [];
        for (let f = 1; f < s.nFreqs; f++) {
          let d = s.phases[f][t] - s.phases[f - 1][t];
          d = Math.atan2(Math.sin(d), Math.cos(d)); // wrap to [-π, π]
          diffs.push(d);
        }
        gddStdSum += std(diffs);
      }
      const gddStd = gddStdSum / s.nFrames;

      // Score based on coherence
      let score: number;
      if (avgCoherence < 0.1) {
        score = 20;
      } else if (avgCoherence < 0.25) {
        score = Math.round(20 + (avgCoherence - 0.1) / 0.15 * 30);
      } else if (avgCoherence < 0.65) {
        score = Math.round(50 + (avgCoherence - 0.25) / 0.4 * 40);
      } else if (avgCoherence < 0.85) {
        score = Math.round(90 - (avgCoherence - 0.65) / 0.2 * 20);
      } else {
        score = Math.round(clamp(70 - (avgCoherence - 0.85) * 200, 20, 70));
      }

      return {
        gdd_std: gddStd,
        phase_coherence_val: avgCoherence,
        coherence_std: coherenceStd,
        phase_score: score,
      };
    } catch {
      return { phase_score: 50 };
    }
  }

  // =========================================================================
  // 2. Voice Quality Metrics (Jitter / Shimmer / HNR)
  // =========================================================================

  private analyzeVoiceQuality(
    samples: Float64Array, sr: number
  ): Record<string, number> {
    try {
      const start = Math.floor(samples.length / 10);
      const end = samples.length - start;
      const seg = samples.subarray(start, end);

      const frameLen = Math.floor(0.03 * sr);
      const hop = Math.floor(0.01 * sr);
      const minLag = Math.floor(sr / 500);
      const maxLag = Math.floor(sr / 50);

      const periods: number[] = [];
      const amplitudes: number[] = [];

      for (let i = 0; i <= seg.length - frameLen; i += hop) {
        const frame = seg.subarray(i, i + frameLen);
        let maxAbs = 0;
        for (let j = 0; j < frame.length; j++) {
          if (Math.abs(frame[j]) > maxAbs) maxAbs = Math.abs(frame[j]);
        }
        if (maxAbs < 0.01) continue;

        // Autocorrelation (positive lags only)
        const corr = new Float64Array(maxLag + 1);
        for (let lag = 0; lag <= maxLag && lag < frameLen; lag++) {
          let sum = 0;
          for (let j = 0; j < frameLen - lag; j++) {
            sum += frame[j] * frame[j + lag];
          }
          corr[lag] = sum;
        }
        if (corr[0] < 1e-10) continue;
        for (let lag = 0; lag <= maxLag; lag++) corr[lag] /= corr[0];

        // Find peak in valid lag range
        if (minLag >= maxLag) continue;
        let peakIdx = minLag;
        for (let lag = minLag + 1; lag <= maxLag; lag++) {
          if (corr[lag] > corr[peakIdx]) peakIdx = lag;
        }

        if (corr[peakIdx] > 0.3) {
          periods.push(peakIdx / sr);
          let rmsSum = 0;
          for (let j = 0; j < frameLen; j++) rmsSum += frame[j] * frame[j];
          amplitudes.push(Math.sqrt(rmsSum / frameLen));
        }
      }

      if (periods.length < 5) {
        return {
          jitter: 0, shimmer: 0, hnr: 0,
          voice_quality_score: 50, voiced_frames: periods.length,
        };
      }

      // Jitter
      let jitterSum = 0;
      for (let i = 1; i < periods.length; i++) {
        jitterSum += Math.abs(periods[i] - periods[i - 1]);
      }
      const meanPeriod = periods.reduce((a, b) => a + b, 0) / periods.length;
      const jitter = (jitterSum / (periods.length - 1)) / (meanPeriod + 1e-10);

      // Shimmer
      let shimmerSum = 0;
      for (let i = 1; i < amplitudes.length; i++) {
        shimmerSum += Math.abs(amplitudes[i] - amplitudes[i - 1]);
      }
      const meanAmp = amplitudes.reduce((a, b) => a + b, 0) / amplitudes.length;
      const shimmer = (shimmerSum / (amplitudes.length - 1)) / (meanAmp + 1e-10);

      // HNR estimate from autocorrelation
      // Re-compute median peak correlation
      const hnrEstimates: number[] = [];
      for (let i = 0; i <= seg.length - frameLen; i += hop) {
        const frame = seg.subarray(i, i + frameLen);
        let maxAbs = 0;
        for (let j = 0; j < frame.length; j++) {
          if (Math.abs(frame[j]) > maxAbs) maxAbs = Math.abs(frame[j]);
        }
        if (maxAbs < 0.01) continue;
        const corr = new Float64Array(maxLag + 1);
        for (let lag = 0; lag <= maxLag && lag < frameLen; lag++) {
          let sum = 0;
          for (let j = 0; j < frameLen - lag; j++) sum += frame[j] * frame[j + lag];
          corr[lag] = sum;
        }
        if (corr[0] < 1e-10) continue;
        for (let lag = 0; lag <= maxLag; lag++) corr[lag] /= corr[0];
        let rMax = 0;
        for (let lag = minLag; lag <= maxLag; lag++) {
          if (corr[lag] > rMax) rMax = corr[lag];
        }
        if (rMax > 0 && rMax < 1) {
          hnrEstimates.push(10 * Math.log10(rMax / (1 - rMax + 1e-10)));
        }
      }
      hnrEstimates.sort((a, b) => a - b);
      const hnr = hnrEstimates.length > 0
        ? hnrEstimates[hnrEstimates.length >> 1]
        : 15.0;

      // Jitter score
      let jitterScore: number;
      if (jitter < 0.001) {
        jitterScore = Math.round(jitter / 0.001 * 30);
      } else if (jitter < 0.02) {
        jitterScore = Math.round(clamp(30 + (jitter - 0.001) / 0.019 * 70, 30, 100));
      } else {
        jitterScore = Math.round(clamp(100 - (jitter - 0.02) * 500, 20, 100));
      }

      // Shimmer score
      let shimmerScore: number;
      if (shimmer < 0.01) {
        shimmerScore = Math.round(shimmer / 0.01 * 30);
      } else if (shimmer < 0.10) {
        shimmerScore = Math.round(clamp(30 + (shimmer - 0.01) / 0.09 * 70, 30, 100));
      } else {
        shimmerScore = Math.round(clamp(100 - (shimmer - 0.10) * 300, 20, 100));
      }

      // HNR score
      let hnrScore: number;
      if (hnr > 35) {
        hnrScore = Math.round(clamp(100 - (hnr - 35) * 10, 10, 60));
      } else if (hnr > 10) {
        hnrScore = Math.round(clamp((hnr - 10) / 25 * 80 + 20, 20, 100));
      } else {
        hnrScore = Math.round(clamp(hnr / 10 * 30, 0, 30));
      }

      let vqScore = Math.round((jitterScore + shimmerScore + hnrScore) / 3);

      // Vocoder penalty: high jitter + very low HNR
      if (jitter > 0.05 && hnr < 5) {
        vqScore = Math.max(vqScore - 30, 0);
      }

      return {
        jitter, shimmer, hnr,
        voiced_frames: periods.length,
        jitter_score: jitterScore,
        shimmer_score: shimmerScore,
        hnr_score: hnrScore,
        voice_quality_score: vqScore,
      };
    } catch {
      return { voice_quality_score: 50 };
    }
  }

  // =========================================================================
  // 3. ENF Analysis
  // =========================================================================

  private analyzeEnf(
    samples: Float64Array, sr: number
  ): Record<string, unknown> {
    try {
      if (sr < 200) return { enf_score: 50 };

      const nFft = Math.min(samples.length, sr * 4);
      const mag = rfft(samples.subarray(0, nFft), nFft);
      let fftSize = 1;
      while (fftSize < nFft) fftSize <<= 1;
      const freqRes = sr / fftSize;

      let bestSnr = 0;
      let bestFreq = 0;

      for (const targetFreq of [50.0, 60.0]) {
        if (targetFreq >= sr / 2) continue;

        // Check if this is a speech harmonic
        let isSpeechHarmonic = false;
        for (const mult of [2, 3, 4]) {
          const harmFreq = targetFreq * mult;
          if (harmFreq < sr / 2) {
            const harmBin = Math.round(harmFreq / freqRes);
            const targetBin = Math.round(targetFreq / freqRes);
            if (harmBin < mag.length && targetBin < mag.length) {
              if (mag[harmBin] > mag[targetBin] * 0.5) {
                isSpeechHarmonic = true;
                break;
              }
            }
          }
        }
        if (isSpeechHarmonic) continue;

        const targetBin = Math.round(targetFreq / freqRes);
        const window = Math.max(1, Math.round(0.5 / freqRes));
        const lo = Math.max(0, targetBin - window);
        const hi = Math.min(mag.length, targetBin + window + 1);
        let peakVal = 0;
        for (let i = lo; i < hi; i++) {
          if (mag[i] > peakVal) peakVal = mag[i];
        }

        // Background: median of wider region excluding peak
        const bgLo = Math.max(0, targetBin - 100);
        const bgHi = Math.min(mag.length, targetBin + 100);
        const bgValues: number[] = [];
        for (let i = bgLo; i < bgHi; i++) {
          if (i < lo || i >= hi) bgValues.push(mag[i]);
        }
        bgValues.sort((a, b) => a - b);
        const bgVal = bgValues.length > 0 ? bgValues[bgValues.length >> 1] : 1.0;

        const snr = peakVal / (bgVal + 1e-10);
        if (snr > bestSnr) {
          bestSnr = snr;
          bestFreq = targetFreq;
        }
      }

      const enfDetected = bestSnr > 5.0;
      let enfScore: number;
      if (enfDetected) {
        enfScore = Math.round(clamp(55 + Math.min(bestSnr, 20) * 2, 55, 90));
      } else {
        enfScore = 45;
      }

      return {
        enf_detected: enfDetected,
        enf_best_freq: bestFreq,
        enf_snr: bestSnr,
        enf_score: enfScore,
      };
    } catch {
      return { enf_score: 50 };
    }
  }

  // =========================================================================
  // 4. Spectral Tilt / LTAS
  // =========================================================================

  private analyzeSpectralTilt(
    samples: Float64Array, sr: number
  ): Record<string, number> {
    try {
      const nFft = 2048;
      const hop = 512;
      const s = stft(samples, sr, nFft, hop);

      // LTAS: average magnitude across time for each freq bin
      const ltas = new Float64Array(s.nFreqs);
      for (let f = 0; f < s.nFreqs; f++) {
        let sum = 0;
        for (let t = 0; t < s.nFrames; t++) sum += s.magnitudes[f][t];
        ltas[f] = sum / s.nFrames;
      }

      // Log magnitude
      const ltasDb = new Float64Array(s.nFreqs);
      for (let f = 0; f < s.nFreqs; f++) {
        ltasDb[f] = 20 * Math.log10(ltas[f] + 1e-10);
      }

      // Spectral tilt: linear regression in log-frequency (100 Hz to Nyquist)
      const validIndices: number[] = [];
      for (let f = 0; f < s.nFreqs; f++) {
        if (s.freqs[f] > 100 && s.freqs[f] < sr / 2) validIndices.push(f);
      }

      if (validIndices.length < 10) return { spectral_tilt_score: 50 };

      // Linear regression: slope in dB per log2(freq) = dB per octave
      let sumX = 0, sumY = 0, sumXX = 0, sumXY = 0;
      const n = validIndices.length;
      for (const fi of validIndices) {
        const x = Math.log2(s.freqs[fi] + 1e-10);
        const y = ltasDb[fi];
        sumX += x;
        sumY += y;
        sumXX += x * x;
        sumXY += x * y;
      }
      const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX + 1e-10);
      const tiltDbPerOctave = slope;

      // HF ratio check
      let hfRatio = 0.5;
      if (sr > 8000) {
        let lowSum = 0, lowCount = 0, highSum = 0, highCount = 0;
        for (let f = 0; f < s.nFreqs; f++) {
          if (s.freqs[f] >= 1000 && s.freqs[f] < 4000) { lowSum += ltas[f]; lowCount++; }
          if (s.freqs[f] >= 4000 && s.freqs[f] < Math.min(8000, sr / 2)) { highSum += ltas[f]; highCount++; }
        }
        if (lowCount > 0 && highCount > 0) {
          hfRatio = (highSum / highCount) / (lowSum / lowCount + 1e-10);
        }
      }

      // Score
      let tiltScore: number;
      if (tiltDbPerOctave >= -8 && tiltDbPerOctave <= -3) tiltScore = 80;
      else if ((tiltDbPerOctave >= -12 && tiltDbPerOctave < -3) ||
               (tiltDbPerOctave > -8 && tiltDbPerOctave <= -1)) tiltScore = 60;
      else tiltScore = 35;

      if (hfRatio < 0.05) tiltScore = Math.min(tiltScore, 30);

      return {
        tilt_db_per_octave: tiltDbPerOctave,
        hf_ratio: hfRatio,
        spectral_tilt_score: tiltScore,
      };
    } catch {
      return { spectral_tilt_score: 50 };
    }
  }

  // =========================================================================
  // 5. Background Noise Consistency
  // =========================================================================

  private analyzeNoiseConsistency(
    samples: Float64Array, sr: number
  ): Record<string, number> {
    try {
      const frameLen = Math.floor(0.05 * sr);
      const hop = Math.floor(frameLen / 2);
      const nFrames = Math.floor((samples.length - frameLen) / hop);
      if (nFrames < 4) return { noise_consistency_score: 50 };

      // High-pass filter approximation: subtract local mean
      const residualLevels: number[] = [];
      for (let i = 0; i < nFrames; i++) {
        const startPos = i * hop;
        const frame = samples.subarray(startPos, startPos + frameLen);
        // Simple high-pass: subtract mean
        let frameMean = 0;
        for (let j = 0; j < frameLen; j++) frameMean += frame[j];
        frameMean /= frameLen;
        let rmsSum = 0;
        for (let j = 0; j < frameLen; j++) {
          const v = frame[j] - frameMean;
          rmsSum += v * v;
        }
        residualLevels.push(Math.sqrt(rmsSum / frameLen));
      }

      const noiseLevelCv = std(residualLevels) / (mean(residualLevels) + 1e-10);

      // Step change detection
      const mid = Math.floor(residualLevels.length / 2);
      let stepChange = 0;
      if (mid > 2) {
        const firstHalfMean = mean(residualLevels.slice(0, mid));
        const secondHalfMean = mean(residualLevels.slice(mid));
        const overallMean = mean(residualLevels) + 1e-10;
        stepChange = Math.abs(firstHalfMean - secondHalfMean) / overallMean;
      }

      const meanNoise = mean(residualLevels);
      const tooQuiet = meanNoise < 0.001;

      const levelScore = Math.round(clamp(100 - noiseLevelCv * 60, 20, 100));
      const stepPenalty = Math.min(30, Math.round(stepChange * 60));
      const quietPenalty = tooQuiet ? 20 : 0;

      const noiseConsistencyScore = Math.round(
        clamp(levelScore - stepPenalty - quietPenalty, 0, 100)
      );

      return {
        noise_level_cv: noiseLevelCv,
        step_change: stepChange,
        mean_noise_level: meanNoise,
        noise_consistency_score: noiseConsistencyScore,
      };
    } catch {
      return { noise_consistency_score: 50 };
    }
  }

  // =========================================================================
  // 6. Mel-Spectrogram Regularity
  // =========================================================================

  private analyzeMelRegularity(
    samples: Float64Array, sr: number
  ): Record<string, number> {
    try {
      const nFft = 1024;
      const hop = 256;
      const nMels = 40;
      const s = stft(samples, sr, nFft, hop);

      // Build mel filterbank
      const fb = melFilterbank(sr, nFft, nMels);

      // Compute mel spectrogram
      const melSpec: Float64Array[] = [];
      for (let m = 0; m < nMels; m++) {
        const row = new Float64Array(s.nFrames);
        for (let t = 0; t < s.nFrames; t++) {
          let sum = 0;
          for (let f = 0; f < Math.min(fb[m].length, s.nFreqs); f++) {
            sum += fb[m][f] * s.magnitudes[f][t] * s.magnitudes[f][t];
          }
          row[t] = 10 * Math.log10(sum + 1e-10);
        }
        melSpec.push(row);
      }

      if (s.nFrames < 3) return { mel_regularity_score: 50 };

      // Delta mel and temporal autocorrelation
      const temporalAutocorrs: number[] = [];
      const bandCvs: number[] = [];

      for (let b = 0; b < nMels; b++) {
        // Delta
        const delta = new Float64Array(s.nFrames - 1);
        for (let t = 0; t < s.nFrames - 1; t++) {
          delta[t] = melSpec[b][t + 1] - melSpec[b][t];
        }

        const deltaStd = std(delta);
        let meanAbsDelta = 0;
        for (let t = 0; t < delta.length; t++) meanAbsDelta += Math.abs(delta[t]);
        meanAbsDelta /= delta.length + 1e-10;
        bandCvs.push(deltaStd / (meanAbsDelta + 1e-10));

        // Temporal autocorrelation
        if (delta.length > 10) {
          const deltaMean = mean(delta);
          const centered = new Float64Array(delta.length);
          for (let i = 0; i < delta.length; i++) centered[i] = delta[i] - deltaMean;
          let acZero = 0;
          for (let i = 0; i < centered.length; i++) acZero += centered[i] * centered[i];
          if (acZero > 1e-10) {
            let acSum = 0;
            const maxLag = Math.min(6, centered.length);
            for (let lag = 1; lag < maxLag; lag++) {
              let ac = 0;
              for (let i = 0; i < centered.length - lag; i++) {
                ac += centered[i] * centered[i + lag];
              }
              acSum += Math.abs(ac / acZero);
            }
            temporalAutocorrs.push(acSum / (maxLag - 1));
          }
        }
      }

      const avgMelCv = mean(bandCvs);
      const avgTemporalAutocorr = temporalAutocorrs.length > 0
        ? mean(temporalAutocorrs) : 0.3;

      // Score
      let autocorrScore: number;
      if (avgTemporalAutocorr < 0.15) autocorrScore = 80;
      else if (avgTemporalAutocorr < 0.3) autocorrScore = Math.round(80 - (avgTemporalAutocorr - 0.15) / 0.15 * 30);
      else if (avgTemporalAutocorr < 0.5) autocorrScore = Math.round(50 - (avgTemporalAutocorr - 0.3) / 0.2 * 20);
      else autocorrScore = Math.round(clamp(30 - (avgTemporalAutocorr - 0.5) * 40, 10, 30));

      const cvBonus = avgMelCv < 0.2 ? -15 : 0;
      const melRegularityScore = Math.round(clamp(autocorrScore + cvBonus, 0, 100));

      return {
        mel_cv: avgMelCv,
        temporal_autocorr: avgTemporalAutocorr,
        mel_regularity_score: melRegularityScore,
      };
    } catch {
      return { mel_regularity_score: 50 };
    }
  }

  // =========================================================================
  // 7. Formant Bandwidth Analysis
  // =========================================================================

  private analyzeFormantBandwidth(
    samples: Float64Array, sr: number
  ): Record<string, number> {
    try {
      const dsSr = Math.min(sr, 16000);
      let dsSamples = samples;
      if (sr > dsSr) {
        // Simple decimation (skip samples)
        const ratio = Math.floor(sr / dsSr);
        const len = Math.floor(samples.length / ratio);
        dsSamples = new Float64Array(len);
        for (let i = 0; i < len; i++) dsSamples[i] = samples[i * ratio];
      }

      const frameLen = Math.floor(0.025 * dsSr);
      const hop = Math.floor(0.01 * dsSr);
      const lpcOrder = 2 + Math.floor(dsSr / 1000);

      const valleyDepths: number[] = [];

      for (let i = 0; i <= dsSamples.length - frameLen; i += hop) {
        const frame = dsSamples.subarray(i, i + frameLen);
        let maxAbs = 0;
        for (let j = 0; j < frameLen; j++) {
          if (Math.abs(frame[j]) > maxAbs) maxAbs = Math.abs(frame[j]);
        }
        if (maxAbs < 0.01) continue;

        // Pre-emphasis + window
        const windowed = new Float64Array(frameLen);
        windowed[0] = frame[0] * 0.54; // hamming(0)
        for (let j = 1; j < frameLen; j++) {
          const hamming = 0.54 - 0.46 * Math.cos(2 * Math.PI * j / (frameLen - 1));
          windowed[j] = (frame[j] - 0.97 * frame[j - 1]) * hamming;
        }

        // Autocorrelation
        const autocorr = new Float64Array(lpcOrder + 1);
        for (let lag = 0; lag <= lpcOrder; lag++) {
          let sum = 0;
          for (let j = 0; j < frameLen - lag; j++) sum += windowed[j] * windowed[j + lag];
          autocorr[lag] = sum;
        }

        if (autocorr[0] < 1e-10) continue;

        // Levinson-Durbin
        let aCoeffs: Float64Array;
        try {
          aCoeffs = levinsonDurbin(autocorr, lpcOrder);
        } catch { continue; }

        // LPC spectral envelope via DFT of denominator
        const nPoints = 512;
        const lpcSpectrum = new Float64Array(nPoints);
        for (let k = 0; k < nPoints; k++) {
          const w = (Math.PI * k) / nPoints;
          let realPart = 0, imagPart = 0;
          for (let j = 0; j <= lpcOrder; j++) {
            realPart += aCoeffs[j] * Math.cos(j * w);
            imagPart -= aCoeffs[j] * Math.sin(j * w);
          }
          const mag = Math.sqrt(realPart * realPart + imagPart * imagPart);
          lpcSpectrum[k] = 20 * Math.log10(1.0 / (mag + 1e-10));
        }

        // Find peaks (formants)
        const peaks = findPeaks(lpcSpectrum, { distance: 20 });
        if (peaks.length >= 2) {
          const p1 = peaks[0];
          const p2 = peaks[1];
          let valley = Infinity;
          for (let k = p1; k <= p2; k++) {
            if (lpcSpectrum[k] < valley) valley = lpcSpectrum[k];
          }
          const peakAvg = (lpcSpectrum[p1] + lpcSpectrum[p2]) / 2;
          valleyDepths.push(peakAvg - valley);
        }
      }

      if (valleyDepths.length === 0) return { formant_score: 50 };

      const avgValleyDepth = mean(valleyDepths);

      let valleyScore: number;
      if (avgValleyDepth > 15) valleyScore = 85;
      else if (avgValleyDepth > 8) valleyScore = Math.round(40 + (avgValleyDepth - 8) / 7 * 45);
      else valleyScore = Math.round(clamp(avgValleyDepth / 8 * 40, 10, 40));

      return {
        avg_valley_depth: avgValleyDepth,
        n_analyzed_frames: valleyDepths.length,
        formant_score: Math.round(valleyScore),
      };
    } catch {
      return { formant_score: 50 };
    }
  }

  // =========================================================================
  // 8. Double Compression Detection
  // =========================================================================

  private analyzeDoubleCompression(
    samples: Float64Array, sr: number
  ): Record<string, number> {
    try {
      const blockSize = 1024;
      const hop = 512;
      const nBlocks = Math.floor((samples.length - blockSize) / hop);
      if (nBlocks < 5) return { double_comp_score: 50 };

      // Compute DCT coefficients per block
      const avgCoeffMag = new Float64Array(blockSize);
      for (let i = 0; i < nBlocks; i++) {
        const startPos = i * hop;
        const block = new Float64Array(blockSize);
        const win = new Float64Array(blockSize);
        for (let j = 0; j < blockSize; j++) {
          win[j] = 0.5 * (1 - Math.cos(2 * Math.PI * j / (blockSize - 1)));
          block[j] = samples[startPos + j] * win[j];
        }
        const coeffs = dctType2(block);
        for (let j = 0; j < blockSize; j++) {
          avgCoeffMag[j] += Math.abs(coeffs[j]);
        }
      }
      for (let j = 0; j < blockSize; j++) avgCoeffMag[j] /= nBlocks;

      // Autocorrelation of coefficient magnitude pattern
      const acm = new Float64Array(blockSize);
      const acmMean = mean(avgCoeffMag);
      for (let j = 0; j < blockSize; j++) acm[j] = avgCoeffMag[j] - acmMean;

      let acZero = 0;
      for (let j = 0; j < blockSize; j++) acZero += acm[j] * acm[j];
      const autocorr = new Float64Array(blockSize);
      for (let lag = 0; lag < blockSize; lag++) {
        let sum = 0;
        for (let j = 0; j < blockSize - lag; j++) sum += acm[j] * acm[j + lag];
        autocorr[lag] = sum / (acZero + 1e-10);
      }

      const peaks = findPeaks(autocorr.subarray(1), { height: 0.1, distance: 5 });
      const periodicPeakCount = peaks.length;

      // Histogram uniformity
      const histScores: number[] = [];
      for (let bandStart = 0; bandStart < Math.min(blockSize, 512); bandStart += 64) {
        const values: number[] = [];
        for (let i = 0; i < nBlocks; i++) {
          const startPos = i * hop;
          const block = new Float64Array(blockSize);
          for (let j = 0; j < blockSize; j++) {
            const win = 0.5 * (1 - Math.cos(2 * Math.PI * j / (blockSize - 1)));
            block[j] = samples[startPos + j] * win;
          }
          const coeffs = dctType2(block);
          for (let j = bandStart; j < Math.min(bandStart + 64, blockSize); j++) {
            values.push(coeffs[j]);
          }
        }

        if (values.length < 10) continue;
        // Histogram entropy
        const nBins = 50;
        let minVal = Infinity, maxVal = -Infinity;
        for (const v of values) {
          if (v < minVal) minVal = v;
          if (v > maxVal) maxVal = v;
        }
        const range = maxVal - minVal + 1e-10;
        const bins = new Float64Array(nBins);
        for (const v of values) {
          const bin = Math.min(nBins - 1, Math.floor((v - minVal) / range * nBins));
          bins[bin]++;
        }
        let entSum = 0;
        for (let b = 0; b < nBins; b++) {
          const p = bins[b] / values.length;
          if (p > 0) entSum -= p * Math.log2(p + 1e-20);
        }
        histScores.push(entSum / Math.log2(nBins));
      }

      const avgHistUniformity = histScores.length > 0 ? mean(histScores) : 0.5;
      const uniformityScore = Math.round(clamp(avgHistUniformity * 100, 0, 100));
      const periodicityPenalty = Math.min(30, periodicPeakCount * 5);
      const doubleCompScore = Math.round(clamp(uniformityScore - periodicityPenalty, 0, 100));

      return {
        hist_uniformity: avgHistUniformity,
        periodic_peaks: periodicPeakCount,
        double_comp_score: doubleCompScore,
      };
    } catch {
      return { double_comp_score: 50 };
    }
  }

  // =========================================================================
  // 9. Spectral Discontinuity (Splice Detection)
  // =========================================================================

  private analyzeSpectralDiscontinuity(
    samples: Float64Array, sr: number
  ): Record<string, number> {
    try {
      const frameLen = Math.floor(0.025 * sr);
      const hop = Math.floor(0.010 * sr);
      const nFrames = Math.floor((samples.length - frameLen) / hop);
      if (nFrames < 10) return { splice_score: 50 };

      const centroids: number[] = [];
      const fluxes: number[] = [];
      const zcrs: number[] = [];
      let prevSpec: Float64Array | null = null;

      for (let i = 0; i < nFrames; i++) {
        const startPos = i * hop;
        const frame = new Float64Array(frameLen);
        for (let j = 0; j < frameLen; j++) {
          const win = 0.5 * (1 - Math.cos(2 * Math.PI * j / (frameLen - 1)));
          frame[j] = samples[startPos + j] * win;
        }

        const spec = rfft(frame, frameLen);

        // Spectral centroid
        let sumFreqSpec = 0, sumSpec = 0;
        for (let f = 0; f < spec.length; f++) {
          const freq = (f * sr) / (frameLen);
          sumFreqSpec += freq * spec[f];
          sumSpec += spec[f];
        }
        centroids.push(sumFreqSpec / (sumSpec + 1e-10));

        // Spectral flux
        if (prevSpec) {
          const minLen = Math.min(spec.length, prevSpec.length);
          let flux = 0;
          for (let f = 0; f < minLen; f++) {
            const d = spec[f] - prevSpec[f];
            flux += d * d;
          }
          fluxes.push(flux);
        }
        prevSpec = spec;

        // Zero crossing rate
        let zcCount = 0;
        for (let j = 1; j < frameLen; j++) {
          if ((samples[startPos + j] >= 0) !== (samples[startPos + j - 1] >= 0)) zcCount++;
        }
        zcrs.push(zcCount / (2 * frameLen));
      }

      // Detect discontinuities via z-score of spectral flux
      let maxFluxZ = 0;
      let nSpikes = 0;
      if (fluxes.length > 3) {
        const fluxMean = mean(fluxes);
        const fluxStd = std(fluxes) + 1e-10;
        for (const f of fluxes) {
          const z = Math.abs((f - fluxMean) / fluxStd);
          if (z > maxFluxZ) maxFluxZ = z;
          if (z > 3.0) nSpikes++;
        }
      }

      const centroidCv = std(centroids) / (mean(centroids) + 1e-10);
      const spikePenalty = Math.min(50, nSpikes * 20);
      const cvPenalty = Math.min(25, Math.max(0, (centroidCv - 0.15) * 80));
      const spliceScore = Math.round(clamp(100 - spikePenalty - cvPenalty, 0, 100));

      return {
        max_flux_zscore: maxFluxZ,
        n_flux_spikes: nSpikes,
        centroid_cv: centroidCv,
        splice_score: spliceScore,
      };
    } catch {
      return { splice_score: 50 };
    }
  }

  // =========================================================================
  // 10. Sub-band Energy Analysis
  // =========================================================================

  private analyzeSubbandEnergy(
    samples: Float64Array, sr: number
  ): Record<string, number> {
    try {
      const nFft = 2048;
      const hop = 512;
      const s = stft(samples, sr, nFft, hop);
      const nyquist = sr / 2;

      const bandDefs: Array<[string, number, number]> = [
        ['low', 0, 500],
        ['mid_low', 500, 2000],
        ['mid', 2000, 4000],
        ['mid_high', 4000, 8000],
        ['high', 8000, nyquist],
      ];

      const bandEnergies: Record<string, number> = {};
      const bandVariabilities: Record<string, number> = {};

      for (const [name, lo, hiRaw] of bandDefs) {
        if (lo >= nyquist) continue;
        const hi = Math.min(hiRaw, nyquist);

        const bandIndices: number[] = [];
        for (let f = 0; f < s.nFreqs; f++) {
          if (s.freqs[f] >= lo && s.freqs[f] < hi) bandIndices.push(f);
        }
        if (bandIndices.length === 0) continue;

        // Energy per frame
        const frameEnergies = new Float64Array(s.nFrames);
        for (let t = 0; t < s.nFrames; t++) {
          let sum = 0;
          for (const fi of bandIndices) {
            sum += s.magnitudes[fi][t] * s.magnitudes[fi][t];
          }
          frameEnergies[t] = sum;
        }

        bandEnergies[name] = mean(frameEnergies);
        if (s.nFrames > 1) {
          bandVariabilities[name] = std(frameEnergies) / (mean(frameEnergies) + 1e-10);
        } else {
          bandVariabilities[name] = 0;
        }
      }

      const totalEnergy = Object.values(bandEnergies).reduce((a, b) => a + b, 0) + 1e-10;
      const bandRatios: Record<string, number> = {};
      for (const [k, v] of Object.entries(bandEnergies)) {
        bandRatios[k] = v / totalEnergy;
      }

      // HF cutoff detection
      let hasHfCutoff = false;
      if (bandRatios['mid_high'] !== undefined && bandRatios['mid'] !== undefined) {
        if (bandRatios['mid_high'] < bandRatios['mid'] * 0.05) hasHfCutoff = true;
      }
      if (bandRatios['high'] !== undefined && bandRatios['mid_high'] !== undefined) {
        if (bandRatios['high'] < bandRatios['mid_high'] * 0.01) hasHfCutoff = true;
      }

      const lowCv = bandVariabilities['low'] ?? 0.3;
      const lowBandTooStable = lowCv < 0.15;

      const cutoffPenalty = hasHfCutoff ? 25 : 0;
      const stabilityPenalty = lowBandTooStable ? 20 : 0;
      const midLowRatio = bandRatios['mid_low'] ?? 0;
      const distributionScore = (midLowRatio > 0.2 && midLowRatio < 0.6) ? 70 : 50;

      const subbandScore = Math.round(
        clamp(distributionScore - cutoffPenalty - stabilityPenalty, 0, 100)
      );

      return {
        hf_cutoff_detected: hasHfCutoff ? 1 : 0,
        low_band_cv: lowCv,
        subband_score: subbandScore,
      };
    } catch {
      return { subband_score: 50 };
    }
  }

  // =========================================================================
  // Composite scoring
  // =========================================================================

  private calculateCompositeScore(
    results: Record<string, Record<string, unknown>>
  ): number {
    let totalScore = 0;
    let totalWeight = 0;

    for (const [method, weight] of Object.entries(WEIGHTS)) {
      const sub = results[method];
      if (sub) {
        const scoreKey = SCORE_KEY_MAP[method];
        const score = sub[scoreKey];
        if (typeof score === 'number') {
          totalScore += score * weight;
          totalWeight += weight;
        }
      }
    }

    return totalWeight > 0
      ? Math.round(clamp(totalScore / totalWeight, 0, 100))
      : 50;
  }

  // =========================================================================
  // AI indicator counting
  // =========================================================================

  // Declarative indicator table for simple threshold checks.
  // [resultKey, metricKey, op, threshold, default, weight]
  private static readonly AUDIO_INDICATOR_TABLE: Array<
    [string, string, '<' | '>' | '>=', number, number, number]
  > = [
    ['phase_coherence', 'gdd_std', '<', 0.5, 1.0, 1],
    ['voice_quality', 'jitter', '<', 0.002, 0.005, 1],
    ['voice_quality', 'shimmer', '<', 0.015, 0.03, 1],
    ['voice_quality', 'hnr', '>', 30.0, 18.0, 1],
    ['mel_regularity', 'mel_cv', '<', 0.15, 0.3, 1],
    ['spectral_tilt', 'tilt_db_per_octave', '>', -2.0, -5.0, 1],
    ['noise_consistency', 'noise_level_cv', '>', 0.8, 0.3, 1],
    ['formant_bandwidth', 'avg_valley_depth', '<', 8.0, 12.0, 1],
    ['spectral_discontinuity', 'n_flux_spikes', '>=', 2, 0, 1],
    ['subband_energy', 'low_band_cv', '<', 0.12, 0.3, 1],
    ['double_compression', 'periodic_peaks', '>', 5, 0, 1],
    ['noise_consistency', 'step_change', '>', 0.5, 0, 1],
    ['voice_quality', 'voice_quality_score', '<', 15, 50, 1],
  ];

  private countAiIndicators(
    results: Record<string, Record<string, unknown>>
  ): number {
    let indicators = 0;

    // Simple threshold checks from declarative table
    for (const [resultKey, metricKey, op, threshold, def, weight] of AudioVerifier.AUDIO_INDICATOR_TABLE) {
      const val = ((results[resultKey] || {})[metricKey] as number) ?? def;
      if (op === '<' && val < threshold) indicators += weight;
      else if (op === '>' && val > threshold) indicators += weight;
      else if (op === '>=' && val >= threshold) indicators += weight;
    }

    // Compound: ENF absent
    const enfDetected = results['enf_analysis']?.['enf_detected'] as boolean;
    const enfSnr = (results['enf_analysis']?.['enf_snr'] as number) ?? 5.0;
    if (!enfDetected && enfSnr < 1.5) indicators++;

    // Boolean: HF cutoff
    const hfCutoff = results['subband_energy']?.['hf_cutoff_detected'] as number;
    if (hfCutoff) indicators++;

    // Compound: low jitter + low shimmer + high HNR = strong TTS
    const jitter = (results['voice_quality']?.['jitter'] as number) ?? 0.005;
    const shimmer = (results['voice_quality']?.['shimmer'] as number) ?? 0.03;
    const hnr = (results['voice_quality']?.['hnr'] as number) ?? 18.0;
    if (jitter < 0.002 && shimmer < 0.015 && hnr > 30.0) indicators++;

    // Vocoder pattern: high jitter + very low HNR
    if (jitter > 0.05 && hnr < 3) indicators += 2;

    // Perfect-pitch non-speech
    const voiced = (results['voice_quality']?.['voiced_frames'] as number) ?? 0;
    if (jitter < 0.0005 && shimmer < 0.005 && voiced > 50) indicators += 2;

    return indicators;
  }

  // =========================================================================
  // Result determination
  // =========================================================================

  private determineResult(
    score: number,
    aiProbability: number,
    results: Record<string, Record<string, unknown>>
  ): { passed: boolean; adjustedScore: number; warning: string | null } {
    // Vocoder pattern: high jitter + very low HNR
    const jitter = (results['voice_quality']?.['jitter'] as number) ?? 0.005;
    const hnr = (results['voice_quality']?.['hnr'] as number) ?? 18.0;
    if (jitter > 0.05 && hnr < 3) {
      return {
        passed: false,
        adjustedScore: Math.round(score * 0.5),
        warning: 'Phase reconstruction artifacts detected - likely vocoder output',
      };
    }

    // Splice detection
    const nSpikes = (results['spectral_discontinuity']?.['n_flux_spikes'] as number) ?? 0;
    const noiseStep = (results['noise_consistency']?.['step_change'] as number) ?? 0;
    const noiseCv = (results['noise_consistency']?.['noise_level_cv'] as number) ?? 0.3;
    if ((nSpikes >= 2 && noiseStep > 0.3) || (nSpikes >= 3 && noiseCv > 0.4)) {
      const adjusted = Math.round(score * 0.6);
      return {
        passed: adjusted >= 50,
        adjustedScore: adjusted,
        warning: 'Audio splice detected - likely manipulated',
      };
    }

    // General probability brackets
    if (aiProbability >= 0.4) {
      return {
        passed: false,
        adjustedScore: Math.round(score * 0.5),
        warning: 'Strong indicators of AI-generated audio',
      };
    }
    if (aiProbability >= 0.3) {
      const adjusted = Math.round(score * 0.7);
      return {
        passed: adjusted >= 50,
        adjustedScore: adjusted,
        warning: 'Some indicators of AI-generated audio',
      };
    }
    if (aiProbability >= 0.15) {
      const passed = score >= 50;
      return {
        passed,
        adjustedScore: score,
        warning: passed ? null : 'Minor audio inconsistencies detected',
      };
    }
    return {
      passed: score >= 40,
      adjustedScore: score,
      warning: null,
    };
  }
}
