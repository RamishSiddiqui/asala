/**
 * Video Physics-based Verification for AI-generated Video Detection.
 *
 * Implements temporal and per-frame analysis techniques to detect AI-generated
 * or manipulated video.  Composes with PhysicsVerifier for per-frame image
 * analysis and adds video-specific temporal methods.
 *
 * Methods:
 *   1. Per-Frame Image Forensics Aggregation   — existing image methods on sampled frames
 *   2. Temporal Noise Consistency              — cross-correlation of noise residuals across frames
 *   3. Optical Flow Anomaly Detection          — block-matching flow smoothness & consistency
 *   4. GOP / Double Encoding Analysis          — compression artifact periodicity
 *   5. Temporal Lighting Consistency           — lighting direction stability across frames
 *   6. Frame-to-Frame Stability               — global motion and jitter analysis
 */

import { LayerResult } from '../types';
import {
  toGrayscale,
  laplacian,
  sobelXY,
  gradientMagnitude,
  medianFilter,
} from '../imaging/processing';
import { mean, std, clamp } from '../imaging/stats';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface VideoFrame {
  width: number;
  height: number;
  data: Uint8Array; // RGBA interleaved
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Resize an RGBA frame to a target size using nearest-neighbor. */
function resizeFrame(frame: VideoFrame, tw: number, th: number): VideoFrame {
  if (frame.width === tw && frame.height === th) return frame;
  const out = new Uint8Array(tw * th * 4);
  for (let y = 0; y < th; y++) {
    const sy = Math.min(Math.floor(y * frame.height / th), frame.height - 1);
    for (let x = 0; x < tw; x++) {
      const sx = Math.min(Math.floor(x * frame.width / tw), frame.width - 1);
      const si = (sy * frame.width + sx) * 4;
      const di = (y * tw + x) * 4;
      out[di] = frame.data[si];
      out[di + 1] = frame.data[si + 1];
      out[di + 2] = frame.data[si + 2];
      out[di + 3] = frame.data[si + 3];
    }
  }
  return { width: tw, height: th, data: out };
}

/** Convert RGBA Uint8Array to Float64Array grayscale [0, 1]. */
function toGrayFloat(rgba: Uint8Array, w: number, h: number): Float64Array {
  const gray = toGrayscale(rgba, w, h); // returns 0-255 range
  const out = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) {
    out[i] = gray[i] / 255.0;
  }
  return out;
}

/** Pearson correlation coefficient between two arrays. */
function pearsonCorr(a: Float64Array, b: Float64Array): number {
  const n = Math.min(a.length, b.length);
  if (n === 0) return 0;
  let sumA = 0, sumB = 0;
  for (let i = 0; i < n; i++) { sumA += a[i]; sumB += b[i]; }
  const meanA = sumA / n;
  const meanB = sumB / n;
  let cov = 0, varA = 0, varB = 0;
  for (let i = 0; i < n; i++) {
    const da = a[i] - meanA;
    const db = b[i] - meanB;
    cov += da * db;
    varA += da * da;
    varB += db * db;
  }
  const denom = Math.sqrt(varA * varB);
  if (denom < 1e-15) return 0;
  return cov / denom;
}

/** Sample evenly-spaced indices from [0, total-1]. */
function sampleIndices(total: number, maxSamples: number): number[] {
  if (total <= maxSamples) return Array.from({ length: total }, (_, i) => i);
  const indices: number[] = [];
  for (let i = 0; i < maxSamples; i++) {
    indices.push(Math.round(i * (total - 1) / (maxSamples - 1)));
  }
  return [...new Set(indices)];
}

/**
 * Simplified block-matching optical flow.
 *
 * Divides the image into blocks and finds the best match in the next frame
 * within a search window. Returns per-block flow vectors.
 */
function blockMatchingFlow(
  gray1: Float64Array, gray2: Float64Array,
  w: number, h: number,
  blockSize: number = 16, searchRadius: number = 8
): { dx: Float64Array; dy: Float64Array; bw: number; bh: number } {
  const bw = Math.floor(w / blockSize);
  const bh = Math.floor(h / blockSize);
  const dx = new Float64Array(bw * bh);
  const dy = new Float64Array(bw * bh);

  for (let by = 0; by < bh; by++) {
    for (let bx = 0; bx < bw; bx++) {
      const refX = bx * blockSize;
      const refY = by * blockSize;
      let bestDx = 0, bestDy = 0;
      let bestSAD = Infinity;

      for (let sy = -searchRadius; sy <= searchRadius; sy++) {
        for (let sx = -searchRadius; sx <= searchRadius; sx++) {
          const candX = refX + sx;
          const candY = refY + sy;
          if (candX < 0 || candY < 0 ||
              candX + blockSize > w || candY + blockSize > h) continue;

          let sad = 0;
          for (let py = 0; py < blockSize; py++) {
            for (let px = 0; px < blockSize; px++) {
              const i1 = (refY + py) * w + (refX + px);
              const i2 = (candY + py) * w + (candX + px);
              sad += Math.abs(gray1[i1] - gray2[i2]);
            }
          }
          if (sad < bestSAD) {
            bestSAD = sad;
            bestDx = sx;
            bestDy = sy;
          }
        }
      }

      dx[by * bw + bx] = bestDx;
      dy[by * bw + bx] = bestDy;
    }
  }

  return { dx, dy, bw, bh };
}

// ---------------------------------------------------------------------------
// VideoVerifier
// ---------------------------------------------------------------------------

export class VideoVerifier {
  private static readonly WEIGHTS: Record<string, number> = {
    frame_analysis: 0.25,
    temporal_noise: 0.18,
    optical_flow: 0.18,
    encoding_analysis: 0.10,
    temporal_lighting: 0.14,
    frame_stability: 0.15,
  };

  private static readonly SCORE_KEY_MAP: Record<string, string> = {
    frame_analysis: 'frame_analysis_score',
    temporal_noise: 'temporal_noise_score',
    optical_flow: 'optical_flow_score',
    encoding_analysis: 'encoding_score',
    temporal_lighting: 'temporal_lighting_score',
    frame_stability: 'stability_score',
  };

  private imageVerifier: { verifyImage(buf: Buffer): LayerResult } | null;

  constructor(imageVerifier?: { verifyImage(buf: Buffer): LayerResult } | null) {
    this.imageVerifier = imageVerifier ?? null;
  }

  /**
   * Verify video from pre-decoded frames.
   *
   * Accepts an array of RGBA frames (VideoFrame) and fps.
   * For raw video bytes, first decode using ffmpeg-wasm or similar,
   * then pass the frames here.
   */
  verifyVideoFrames(frames: VideoFrame[], fps: number = 30): LayerResult {
    if (frames.length < 2) {
      return {
        name: 'Physics Verification (Video)',
        passed: false,
        score: 0,
        details: { error: 'Video too short or decode failed (< 2 frames)' },
      };
    }

    const results: Record<string, unknown> = {
      frame_count: frames.length,
      fps,
    };

    // Resize frames to 256x256 for analysis
    const tw = 256, th = 256;
    const resized = frames.map(f => resizeFrame(f, tw, th));

    // Convert to grayscale float [0, 1]
    const grays = resized.map(f => toGrayFloat(f.data, tw, th));

    // Run all analyses
    results['frame_analysis'] = this.analyzeFrames(resized, tw, th);
    results['temporal_noise'] = this.analyzeTemporalNoise(grays, tw, th);
    results['optical_flow'] = this.analyzeOpticalFlow(grays, tw, th);
    results['encoding_analysis'] = this.analyzeEncoding(grays, tw, th);
    results['temporal_lighting'] = this.analyzeTemporalLighting(resized, grays, tw, th);
    results['frame_stability'] = this.analyzeFrameStability(grays, tw, th);

    // Composite score
    const compositeScore = this.calculateCompositeScore(results);

    // AI indicators
    const aiIndicators = this.countAiIndicators(results);
    const totalIndicators = 16;
    const aiProbability = aiIndicators / totalIndicators;

    const { passed, adjustedScore, warning } = this.determineResult(
      compositeScore, aiProbability, results
    );

    if (warning) results['warning'] = warning;
    results['ai_probability'] = aiProbability;
    results['ai_indicators'] = aiIndicators;

    return {
      name: 'Physics Verification (Video)',
      passed,
      score: adjustedScore,
      details: results,
    };
  }

  // =========================================================================
  // 1. Per-Frame Image Forensics Aggregation
  // =========================================================================
  private analyzeFrames(
    frames: VideoFrame[], w: number, h: number
  ): Record<string, unknown> {
    try {
      const n = frames.length;
      const indices = sampleIndices(n, 10);
      const sampled = indices.map(i => frames[i]);

      if (this.imageVerifier) {
        // Full per-frame analysis via image verifier (requires JPEG encoding)
        // Since we don't have a JPEG encoder here, fall through to lightweight
      }

      // Lightweight: per-frame noise + texture
      const noiseCvs: number[] = [];
      const textureScores: number[] = [];

      for (const frame of sampled) {
        const gray = toGrayscale(frame.data, w, h); // 0-255

        // Laplacian variance (noise/detail indicator)
        const grayF = new Float64Array(gray);
        const lap = laplacian(grayF, w, h);
        let lapVar = 0;
        let lapMean = 0;
        for (let i = 0; i < lap.length; i++) lapMean += lap[i];
        lapMean /= lap.length;
        for (let i = 0; i < lap.length; i++) lapVar += (lap[i] - lapMean) ** 2;
        lapVar /= lap.length;

        // Texture: gradient magnitude mean
        const { gx, gy } = sobelXY(grayF, w, h);
        const gradMag = gradientMagnitude(gx, gy);
        let gradSum = 0;
        for (let i = 0; i < gradMag.length; i++) gradSum += gradMag[i];
        textureScores.push(gradSum / gradMag.length);

        // Noise CV across 4×4 grid
        const gridVars: number[] = [];
        const gh = h >> 2, gw = w >> 2;
        for (let gi = 0; gi < 4; gi++) {
          for (let gj = 0; gj < 4; gj++) {
            let rSum = 0, rSum2 = 0, rCount = 0;
            for (let ry = gi * gh; ry < (gi + 1) * gh && ry < h; ry++) {
              for (let rx = gj * gw; rx < (gj + 1) * gw && rx < w; rx++) {
                const v = lap[ry * w + rx];
                rSum += v;
                rSum2 += v * v;
                rCount++;
              }
            }
            if (rCount > 0) {
              const rMean = rSum / rCount;
              const rVar = rSum2 / rCount - rMean * rMean;
              gridVars.push(Math.max(0, rVar));
            }
          }
        }

        if (gridVars.length > 0) {
          const gvMean = gridVars.reduce((a, b) => a + b, 0) / gridVars.length;
          const gvStd = Math.sqrt(
            gridVars.reduce((a, b) => a + (b - gvMean) ** 2, 0) / gridVars.length
          );
          noiseCvs.push(gvStd / (gvMean + 1e-10));
        }
      }

      const avgNoiseCv = noiseCvs.length > 0 ? mean(noiseCvs) : 0.3;
      const avgTexture = textureScores.length > 0 ? mean(textureScores) : 20;
      const textureStd = textureScores.length > 1 ? std(textureScores) : 0;

      // Score: moderate noise CV + good texture = natural
      const noiseScore = clamp(Math.round(avgNoiseCv * 100), 20, 90);
      const textureS = clamp(Math.round(avgTexture / 30 * 60 + 20), 20, 90);
      const frameScore = Math.round(0.5 * noiseScore + 0.5 * textureS);

      return {
        avg_noise_cv: avgNoiseCv,
        avg_texture: avgTexture,
        texture_std: textureStd,
        n_sampled: sampled.length,
        frame_analysis_score: frameScore,
      };
    } catch (err) {
      return { frame_analysis_score: 50, error: String(err) };
    }
  }

  // =========================================================================
  // 2. Temporal Noise Consistency
  // =========================================================================
  private analyzeTemporalNoise(
    grays: Float64Array[], w: number, h: number
  ): Record<string, unknown> {
    try {
      // Extract noise residuals via median filter subtraction
      const maxFrames = Math.min(grays.length, 60);
      const residuals: Float64Array[] = [];

      for (let i = 0; i < maxFrames; i++) {
        // Scale to 0-255 for median filter, then back
        const gray255 = new Float64Array(w * h);
        for (let j = 0; j < w * h; j++) gray255[j] = grays[i][j] * 255;
        const blurred = medianFilter(gray255, w, h, 2); // 5x5 median
        const residual = new Float64Array(w * h);
        for (let j = 0; j < w * h; j++) {
          residual[j] = grays[i][j] - blurred[j] / 255.0;
        }
        residuals.push(residual);
      }

      if (residuals.length < 3) {
        return { temporal_noise_score: 50, note: 'Too few frames' };
      }

      // Cross-correlation between consecutive frame residuals
      const correlations: number[] = [];
      for (let i = 0; i < residuals.length - 1; i++) {
        const corr = pearsonCorr(residuals[i], residuals[i + 1]);
        correlations.push(isNaN(corr) ? 0 : corr);
      }

      const avgCorr = mean(correlations);
      const corrStd = std(correlations);

      // Non-adjacent frame correlation (should decay)
      let avgFarCorr = avgCorr;
      let decay = 0;
      if (residuals.length > 5) {
        const farCorrs: number[] = [];
        for (let i = 0; i < residuals.length - 3; i += 3) {
          const corr = pearsonCorr(residuals[i], residuals[i + 3]);
          if (!isNaN(corr)) farCorrs.push(corr);
        }
        if (farCorrs.length > 0) {
          avgFarCorr = mean(farCorrs);
          decay = avgCorr - avgFarCorr;
        }
      }

      // Score: natural range (0.1-0.6 with some decay) = high score
      let score: number;
      if (avgCorr < 0.02) {
        score = 25;
      } else if (avgCorr < 0.15) {
        score = Math.round(25 + (avgCorr - 0.02) / 0.13 * 30);
      } else if (avgCorr < 0.65) {
        score = Math.round(55 + (avgCorr - 0.15) / 0.5 * 35);
      } else if (avgCorr < 0.85) {
        score = Math.round(90 - (avgCorr - 0.65) / 0.2 * 30);
      } else {
        score = clamp(Math.round(60 - (avgCorr - 0.85) * 200), 15, 60);
      }

      // Bonus for natural decay pattern
      if (decay > 0.01 && decay < 0.3) {
        score = Math.min(100, score + 5);
      }

      return {
        avg_noise_corr: avgCorr,
        noise_corr_std: corrStd,
        noise_corr_decay: decay,
        temporal_noise_score: score,
      };
    } catch (err) {
      return { temporal_noise_score: 50, error: String(err) };
    }
  }

  // =========================================================================
  // 3. Optical Flow Anomaly Detection
  // =========================================================================
  private analyzeOpticalFlow(
    grays: Float64Array[], w: number, h: number
  ): Record<string, unknown> {
    try {
      if (grays.length < 3) {
        return { optical_flow_score: 50, note: 'Too few frames' };
      }

      // Sample up to 30 frame pairs
      const n = grays.length;
      const step = Math.max(1, Math.floor(n / 30));
      const flowMagnitudes: number[] = [];
      const flowSmoothness: number[] = [];

      for (let i = step; i < n; i += step) {
        const { dx, dy, bw, bh } = blockMatchingFlow(grays[i - step], grays[i], w, h);

        // Average flow magnitude
        let magSum = 0;
        for (let j = 0; j < dx.length; j++) {
          magSum += Math.sqrt(dx[j] * dx[j] + dy[j] * dy[j]);
        }
        const avgMag = magSum / dx.length;
        flowMagnitudes.push(avgMag);

        // Flow smoothness: spatial gradient of flow field
        // Measure how much adjacent blocks differ in flow
        let smoothSum = 0;
        let smoothCount = 0;
        for (let by = 0; by < bh - 1; by++) {
          for (let bx = 0; bx < bw - 1; bx++) {
            const idx = by * bw + bx;
            const idxR = by * bw + bx + 1;
            const idxD = (by + 1) * bw + bx;
            const dxDiff = Math.abs(dx[idx] - dx[idxR]) + Math.abs(dx[idx] - dx[idxD]);
            const dyDiff = Math.abs(dy[idx] - dy[idxR]) + Math.abs(dy[idx] - dy[idxD]);
            smoothSum += dxDiff + dyDiff;
            smoothCount += 2;
          }
        }
        const smoothness = smoothCount > 0 ? smoothSum / smoothCount : 0;
        flowSmoothness.push(smoothness);
      }

      if (flowMagnitudes.length === 0) {
        return { optical_flow_score: 50, note: 'No flow computed' };
      }

      const avgMag = mean(flowMagnitudes);
      const magCv = flowMagnitudes.length > 1
        ? std(flowMagnitudes) / (avgMag + 1e-10) : 0;
      const avgSmooth = mean(flowSmoothness);

      // Jitter: frame-to-frame changes in flow magnitude
      let avgJitter = 0, maxJitter = 0;
      if (flowMagnitudes.length > 2) {
        const diffs: number[] = [];
        for (let i = 1; i < flowMagnitudes.length; i++) {
          diffs.push(Math.abs(flowMagnitudes[i] - flowMagnitudes[i - 1]));
        }
        avgJitter = mean(diffs);
        maxJitter = Math.max(...diffs);
      }

      // Score: smooth flow = natural
      const smoothnessScore = clamp(Math.round(100 - avgSmooth * 50), 20, 95);

      // Moderate jitter = natural; very low or very high = suspect
      let jitterScore: number;
      if (avgJitter < 0.01) {
        jitterScore = 40;
      } else if (avgJitter < 2.0) {
        jitterScore = clamp(Math.round(40 + avgJitter / 2.0 * 50), 40, 90);
      } else {
        jitterScore = clamp(Math.round(90 - (avgJitter - 2.0) * 15), 20, 90);
      }

      const opticalFlowScore = Math.round(0.6 * smoothnessScore + 0.4 * jitterScore);

      return {
        avg_flow_magnitude: avgMag,
        flow_mag_cv: magCv,
        avg_flow_smoothness: avgSmooth,
        avg_jitter: avgJitter,
        max_jitter: maxJitter,
        optical_flow_score: opticalFlowScore,
      };
    } catch (err) {
      return { optical_flow_score: 50, error: String(err) };
    }
  }

  // =========================================================================
  // 4. GOP / Double Encoding Analysis
  // =========================================================================
  private analyzeEncoding(
    grays: Float64Array[], w: number, h: number
  ): Record<string, unknown> {
    try {
      const maxFrames = Math.min(grays.length, 60);
      const blockinessPerFrame: number[] = [];

      for (let f = 0; f < maxFrames; f++) {
        const gray = grays[f];
        // Blockiness: average gradient at 8-pixel boundaries
        const hBlocks: number[] = [];
        for (let x = 8; x < w - 1; x += 8) {
          let sum = 0;
          for (let y = 0; y < h; y++) {
            sum += Math.abs(gray[y * w + x] - gray[y * w + x - 1]);
          }
          hBlocks.push(sum / h);
        }

        const vBlocks: number[] = [];
        for (let y = 8; y < h - 1; y += 8) {
          let sum = 0;
          for (let x = 0; x < w; x++) {
            sum += Math.abs(gray[y * w + x] - gray[(y - 1) * w + x]);
          }
          vBlocks.push(sum / w);
        }

        const allBlocks = [...hBlocks, ...vBlocks];
        const avgBlockiness = allBlocks.length > 0 ? mean(allBlocks) : 0;
        blockinessPerFrame.push(avgBlockiness);
      }

      if (blockinessPerFrame.length < 5) {
        return { encoding_score: 50, note: 'Too few frames' };
      }

      const avgBlockiness = mean(blockinessPerFrame);
      const blockinessCv = std(blockinessPerFrame) / (avgBlockiness + 1e-10);

      // Autocorrelation to detect periodic pattern (GOP structure)
      const centered = blockinessPerFrame.map(v => v - avgBlockiness);
      const acLen = Math.min(30, centered.length);
      let nPeriodicPeaks = 0;

      // Compute normalized autocorrelation
      let ac0 = 0;
      for (let i = 0; i < centered.length; i++) ac0 += centered[i] * centered[i];
      if (ac0 > 1e-15) {
        const autocorr: number[] = [1.0];
        for (let lag = 1; lag < acLen; lag++) {
          let sum = 0;
          for (let i = 0; i < centered.length - lag; i++) {
            sum += centered[i] * centered[i + lag];
          }
          autocorr.push(sum / ac0);
        }

        // Count peaks above 0.2
        for (let i = 1; i < autocorr.length - 1; i++) {
          if (autocorr[i] > autocorr[i - 1] && autocorr[i] > autocorr[i + 1]
              && autocorr[i] > 0.2) {
            nPeriodicPeaks++;
          }
        }
      }

      // Score
      let cvScore: number;
      if (blockinessCv < 0.05) {
        cvScore = 30; // Too uniform = synthetic
      } else if (blockinessCv < 0.3) {
        cvScore = Math.round(30 + (blockinessCv - 0.05) / 0.25 * 50);
      } else {
        cvScore = clamp(Math.round(80 - (blockinessCv - 0.3) * 30), 40, 80);
      }

      const periodicPenalty = Math.min(25, nPeriodicPeaks * 10);
      const encodingScore = clamp(cvScore - periodicPenalty, 0, 100);

      return {
        avg_blockiness: avgBlockiness,
        blockiness_cv: blockinessCv,
        n_periodic_peaks: nPeriodicPeaks,
        encoding_score: encodingScore,
      };
    } catch (err) {
      return { encoding_score: 50, error: String(err) };
    }
  }

  // =========================================================================
  // 5. Temporal Lighting Consistency
  // =========================================================================
  private analyzeTemporalLighting(
    frames: VideoFrame[], grays: Float64Array[], w: number, h: number
  ): Record<string, unknown> {
    try {
      const maxFrames = Math.min(grays.length, 60);
      const brightnessValues: number[] = [];
      const gradientDirections: number[] = [];

      for (let f = 0; f < maxFrames; f++) {
        const gray = grays[f];
        // Mean brightness
        let bSum = 0;
        for (let i = 0; i < w * h; i++) bSum += gray[i];
        brightnessValues.push((bSum / (w * h)) * 255); // scale to 0-255 like Python

        // Dominant gradient direction (proxy for lighting)
        // Use gray in 0-255 scale for Sobel
        const gray255 = new Float64Array(w * h);
        for (let i = 0; i < w * h; i++) gray255[i] = gray[i] * 255;
        const { gx, gy } = sobelXY(gray255, w, h);

        const mag = gradientMagnitude(gx, gy);
        let weightSum = 0;
        let weightedGx = 0, weightedGy = 0;
        for (let i = 0; i < mag.length; i++) {
          weightSum += mag[i];
          weightedGy += gy[i] * mag[i];
          weightedGx += gx[i] * mag[i];
        }
        weightSum += 1e-10;
        const avgDir = Math.atan2(weightedGy / weightSum, weightedGx / weightSum);
        gradientDirections.push(avgDir);
      }

      if (brightnessValues.length < 3) {
        return { temporal_lighting_score: 50, note: 'Too few frames' };
      }

      const brightnessMean = mean(brightnessValues);
      const brightnessCv = std(brightnessValues) / (brightnessMean + 1e-10);

      // Brightness flickering
      const brightnessDiffs: number[] = [];
      for (let i = 1; i < brightnessValues.length; i++) {
        brightnessDiffs.push(Math.abs(brightnessValues[i] - brightnessValues[i - 1]));
      }
      const avgFlicker = mean(brightnessDiffs);
      const maxFlicker = Math.max(...brightnessDiffs);

      // Direction stability
      const dirDiffs: number[] = [];
      for (let i = 1; i < gradientDirections.length; i++) {
        let diff = Math.abs(gradientDirections[i] - gradientDirections[i - 1]);
        diff = Math.min(diff, 2 * Math.PI - diff); // wrap around
        dirDiffs.push(diff);
      }
      const avgDirChange = mean(dirDiffs);

      // Score
      let flickerScore: number;
      if (maxFlicker > 30) {
        flickerScore = clamp(Math.round(100 - (maxFlicker - 30) * 3), 20, 80);
      } else {
        flickerScore = 80;
      }

      let dirScore: number;
      if (avgDirChange < 0.01) {
        dirScore = 50;
      } else if (avgDirChange < 0.3) {
        dirScore = Math.round(50 + (avgDirChange - 0.01) / 0.29 * 35);
      } else {
        dirScore = clamp(Math.round(85 - (avgDirChange - 0.3) * 50), 20, 85);
      }

      const temporalLightingScore = Math.round(0.5 * flickerScore + 0.5 * dirScore);

      return {
        brightness_cv: brightnessCv,
        avg_flicker: avgFlicker,
        max_flicker: maxFlicker,
        avg_dir_change: avgDirChange,
        temporal_lighting_score: temporalLightingScore,
      };
    } catch (err) {
      return { temporal_lighting_score: 50, error: String(err) };
    }
  }

  // =========================================================================
  // 6. Frame-to-Frame Stability
  // =========================================================================
  private analyzeFrameStability(
    grays: Float64Array[], w: number, h: number
  ): Record<string, unknown> {
    try {
      if (grays.length < 3) {
        return { stability_score: 50, note: 'Too few frames' };
      }

      const nccValues: number[] = [];
      const mseValues: number[] = [];

      for (let i = 0; i < grays.length - 1; i++) {
        const f1 = grays[i];
        const f2 = grays[i + 1];
        const ncc = pearsonCorr(f1, f2);
        nccValues.push(isNaN(ncc) ? 1.0 : ncc);

        // MSE
        let mse = 0;
        for (let j = 0; j < w * h; j++) {
          mse += (f1[j] - f2[j]) ** 2;
        }
        mseValues.push(mse / (w * h));
      }

      const avgNcc = mean(nccValues);
      const nccStd = std(nccValues);
      const avgMse = mean(mseValues);

      // Micro-jitter: high-frequency oscillation in NCC
      let avgNccJitter = 0;
      if (nccValues.length > 5) {
        const nccDiffs: number[] = [];
        for (let i = 1; i < nccValues.length; i++) {
          nccDiffs.push(Math.abs(nccValues[i] - nccValues[i - 1]));
        }
        avgNccJitter = mean(nccDiffs);
      }

      // Score
      let nccScore: number;
      if (avgNcc > 0.999) {
        nccScore = 40; // Suspiciously static
      } else if (avgNcc > 0.95) {
        nccScore = Math.round(40 + (0.999 - avgNcc) / 0.049 * 50);
      } else if (avgNcc > 0.7) {
        nccScore = 80; // Good — natural motion
      } else {
        nccScore = clamp(Math.round(80 - (0.7 - avgNcc) * 100), 20, 80);
      }

      // Micro-jitter penalty
      let jitterPenalty = 0;
      if (avgNccJitter > 0.05) {
        jitterPenalty = Math.min(20, Math.round((avgNccJitter - 0.05) * 200));
      }

      const stabilityScore = clamp(nccScore - jitterPenalty, 0, 100);

      return {
        avg_ncc: avgNcc,
        ncc_std: nccStd,
        avg_mse: avgMse,
        avg_ncc_jitter: avgNccJitter,
        stability_score: stabilityScore,
      };
    } catch (err) {
      return { stability_score: 50, error: String(err) };
    }
  }

  // =========================================================================
  // Composite scoring
  // =========================================================================
  private calculateCompositeScore(results: Record<string, unknown>): number {
    let totalScore = 0;
    let totalWeight = 0;

    for (const [method, weight] of Object.entries(VideoVerifier.WEIGHTS)) {
      const sub = results[method];
      if (sub && typeof sub === 'object') {
        const scoreKey = VideoVerifier.SCORE_KEY_MAP[method];
        const score = (sub as Record<string, unknown>)[scoreKey];
        if (typeof score === 'number' && isFinite(score)) {
          totalScore += score * weight;
          totalWeight += weight;
        }
      }
    }

    if (totalWeight > 0) {
      return clamp(Math.round(totalScore / totalWeight), 0, 100);
    }
    return 50;
  }

  // =========================================================================
  // AI indicator counting
  // =========================================================================
  private countAiIndicators(results: Record<string, unknown>): number {
    let indicators = 0;

    const get = (key: string, subKey: string, def: number): number => {
      const sub = results[key];
      if (sub && typeof sub === 'object') {
        const val = (sub as Record<string, unknown>)[subKey];
        if (typeof val === 'number') return val;
      }
      return def;
    };

    // 1. Low per-frame image score
    const frameScore = get('frame_analysis', 'frame_analysis_score', 50);
    if (frameScore < 40) indicators++;

    // 2. Very low per-frame score
    if (frameScore < 25) indicators++;

    // 3. No temporal noise correlation (independent synthesis)
    const noiseCorr = get('temporal_noise', 'avg_noise_corr', 0.3);
    if (noiseCorr < 0.05) indicators++;

    // 3b. Zero noise correlation
    if (noiseCorr < 0.01) indicators++;

    // 4. Too high noise correlation (copied noise)
    if (noiseCorr > 0.75) indicators++;

    // 5. Flow smoothness anomaly
    const flowSmooth = get('optical_flow', 'avg_flow_smoothness', 0.5);
    if (flowSmooth > 3.0) indicators++;

    // 6. Flow jitter (deepfake micro-jitter)
    const maxJitter = get('optical_flow', 'max_jitter', 0.5);
    if (maxJitter > 5.0) indicators++;

    // 7. Too-uniform blockiness (AI-generated)
    const blockCv = get('encoding_analysis', 'blockiness_cv', 0.15);
    if (blockCv < 0.05) indicators++;

    // 8. Periodic blockiness (double encoding)
    const periodic = get('encoding_analysis', 'n_periodic_peaks', 0);
    if (periodic > 2) indicators++;

    // 9. Brightness flickering
    const maxFlicker = get('temporal_lighting', 'max_flicker', 5.0);
    if (maxFlicker > 40) indicators++;

    // 10. Lighting direction instability
    const dirChange = get('temporal_lighting', 'avg_dir_change', 0.05);
    if (dirChange > 0.5) indicators++;

    // 11. Suspiciously static video
    const avgNcc = get('frame_stability', 'avg_ncc', 0.95);
    if (avgNcc > 0.996) indicators++;

    // 12. Frame-level micro-jitter
    const nccJitter = get('frame_stability', 'avg_ncc_jitter', 0.01);
    if (nccJitter > 0.08) indicators++;

    // 13. Combined: low frame score + low noise correlation = AI video
    if (frameScore < 40 && noiseCorr < 0.1) indicators++;

    // 14. Combined: perfect stability + uniform encoding = synthetic
    if (avgNcc > 0.996 && blockCv < 0.05) indicators++;

    // 15. High NCC variance (splice indicator)
    const nccStd = get('frame_stability', 'ncc_std', 0.01);
    if (nccStd > 0.08) indicators++;

    return indicators;
  }

  // =========================================================================
  // Result determination
  // =========================================================================
  private determineResult(
    score: number,
    aiProbability: number,
    results: Record<string, unknown>
  ): { passed: boolean; adjustedScore: number; warning: string | null } {
    const get = (key: string, subKey: string, def: number): number => {
      const sub = results[key];
      if (sub && typeof sub === 'object') {
        const val = (sub as Record<string, unknown>)[subKey];
        if (typeof val === 'number') return val;
      }
      return def;
    };

    // Splice detection: zero temporal correlation + unstable similarity
    const noiseCorr = get('temporal_noise', 'avg_noise_corr', 0.3);
    const nccStd = get('frame_stability', 'ncc_std', 0.01);
    if (noiseCorr < 0.02 && nccStd > 0.05) {
      const adjusted = Math.round(score * 0.6);
      return {
        passed: false,
        adjustedScore: adjusted,
        warning: 'Video splice detected — temporal discontinuity',
      };
    }

    if (aiProbability >= 0.4) {
      const adjusted = Math.round(score * 0.5);
      return {
        passed: false,
        adjustedScore: adjusted,
        warning: 'Strong indicators of AI-generated video',
      };
    }

    if (aiProbability >= 0.3) {
      const adjusted = Math.round(score * 0.7);
      return {
        passed: adjusted >= 50,
        adjustedScore: adjusted,
        warning: 'Some indicators of AI-generated or manipulated video',
      };
    }

    if (aiProbability >= 0.15) {
      const passed = score >= 50;
      return {
        passed,
        adjustedScore: score,
        warning: passed ? null : 'Minor video inconsistencies detected',
      };
    }

    return {
      passed: score >= 40,
      adjustedScore: score,
      warning: null,
    };
  }
}
