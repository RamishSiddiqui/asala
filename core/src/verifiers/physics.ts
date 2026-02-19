import jpeg from 'jpeg-js';
import { LayerResult } from '../types';
import { decodeImage, ImageData } from '../imaging/decode';
import {
  toGrayscale,
  laplacian,
  sobelXY,
  gradientMagnitude,
  medianFilter,
  cannyEdges,
  rgbToHSV,
  rgbToLAB,
  fft2d,
  dct2d,
  resize,
  fftShift,
} from '../imaging/processing';
import {
  mean,
  std,
  variance,
  cv,
  histogram,
  entropy,
  percentile,
  clamp,
} from '../imaging/stats';
import { ErrorLevelAnalysis, ELADetailedResult } from '../crypto/ela';
import { AudioVerifier } from './audio';

// ---------------------------------------------------------------------------
// Threshold configuration (matches Python PhysicsVerifier.THRESHOLDS)
// ---------------------------------------------------------------------------

const THRESHOLDS = {
  noise_cv_low: 0.25,
  noise_variance_floor: 30,
  dct_low: 35,
  lighting_low: 35,
  texture_low: 49,
  geometric_low: 31,
  color_low: 20,
  compression_low: 40,
  regional_ela_cv_high: 0.5,
  regional_ela_cv_low: 0.25,
  // Phase 2 thresholds
  noise_map_cv_high: 0.6,
  ghost_spread_high: 3,
  spectral_residual_high: 0.40,
  channel_corr_low: 0.25,
};

// ---------------------------------------------------------------------------
// Composite score weights (Phase 1 + Phase 2)
// ---------------------------------------------------------------------------

const WEIGHTS: Record<string, number> = {
  // Phase 1 (8 core methods)
  noise_uniformity: 0.12,
  noise_frequency: 0.07,
  frequency_analysis: 0.14,
  geometric_consistency: 0.09,
  lighting_analysis: 0.11,
  texture_analysis: 0.11,
  color_analysis: 0.05,
  compression_analysis: 0.05,
  // Phase 2 (4 methods)
  noise_consistency_map: 0.04,
  jpeg_ghost: 0.04,
  spectral_fingerprint: 0.04,
  channel_correlation: 0.04,
  // Phase 3 (4 new methods) — conservative weights
  benford_dct: 0.02,
  wavelet_ratio: 0.03,
  blocking_grid: 0.02,
  cfa_demosaicing: 0.03,
};

const SCORE_KEY_MAP: Record<string, string> = {
  noise_uniformity: 'uniformity_score',
  noise_frequency: 'frequency_score',
  frequency_analysis: 'dct_score',
  geometric_consistency: 'geometric_score',
  lighting_analysis: 'lighting_score',
  texture_analysis: 'texture_score',
  color_analysis: 'color_score',
  compression_analysis: 'compression_score',
  noise_consistency_map: 'noise_map_score',
  jpeg_ghost: 'ghost_score',
  spectral_fingerprint: 'spectral_score',
  channel_correlation: 'correlation_score',
  // Phase 3
  benford_dct: 'benford_score',
  wavelet_ratio: 'wavelet_score',
  blocking_grid: 'bag_score',
  cfa_demosaicing: 'cfa_score',
};

// ---------------------------------------------------------------------------
// PhysicsVerifier
// ---------------------------------------------------------------------------

export class PhysicsVerifier {
  private audioVerifier = new AudioVerifier();

  /**
   * Verify image for physical consistency.
   * Decodes the buffer, runs all analysis methods, and returns a composite result.
   */
  verifyImage(imageBuffer: Buffer): LayerResult {
    try {
      const img = decodeImage(imageBuffer);
      const gray = toGrayscale(img.data, img.width, img.height);
      const w = img.width;
      const h = img.height;

      // Pre-compute shared Sobel gradients once
      const graySobel = sobelXY(gray, w, h);

      const results: Record<string, Record<string, unknown>> = {};

      // Phase 1 analyses
      results['noise_uniformity'] = this.analyzeNoiseUniformity(gray, w, h);
      results['noise_frequency'] = this.analyzeNoiseFrequency(gray, w, h);
      results['frequency_analysis'] = this.analyzeFrequencyDomain(gray, w, h);
      results['geometric_consistency'] = this.analyzeGeometricConsistency(gray, w, h, graySobel);
      results['lighting_analysis'] = this.analyzeLighting(img.data, gray, w, h);
      results['texture_analysis'] = this.analyzeTexturePatterns(gray, w, h, graySobel);
      results['color_analysis'] = this.analyzeColorDistribution(img.data, w, h);
      results['compression_analysis'] = this.analyzeCompressionArtifacts(imageBuffer, img);

      // Phase 2 analyses
      results['noise_consistency_map'] = this.analyzeNoiseConsistencyMap(gray, w, h);
      results['jpeg_ghost'] = this.analyzeJpegGhosts(imageBuffer, img);
      results['spectral_fingerprint'] = this.analyzeSpectralFingerprint(gray, w, h);
      results['channel_correlation'] = this.analyzeChannelCorrelation(img.data, w, h);

      // Phase 3 analyses
      results['benford_dct'] = this.analyzeBenfordDct(gray, w, h);
      results['wavelet_ratio'] = this.analyzeWaveletSpectralRatio(gray, w, h);
      results['blocking_grid'] = this.analyzeBlockingArtifactGrid(gray, w, h);
      results['cfa_demosaicing'] = this.analyzeCfaDemosaicing(img.data, w, h);

      const finalScore = this.calculateCompositeScore(results);
      const aiIndicators = this.countAiIndicators(results);
      const totalIndicators = 20; // Phase 1: 10, Phase 2: 6, Phase 3: 4
      const aiProbability = aiIndicators / totalIndicators;

      const { passed, adjustedScore, warning } = this.determineResult(
        finalScore,
        aiProbability,
        results
      );

      const details: Record<string, unknown> = { ...results };
      if (warning) details['warning'] = warning;
      details['ai_probability'] = aiProbability;
      details['ai_indicators'] = aiIndicators;

      return {
        name: 'Physics Verification (Image)',
        passed,
        score: adjustedScore,
        details,
      };
    } catch (err) {
      return {
        name: 'Physics Verification (Image)',
        passed: false,
        score: 0,
        details: { error: String(err) },
      };
    }
  }

  /**
   * Verify audio for physical consistency.
   * Delegates to AudioVerifier for full 10-method analysis.
   */
  verifyAudio(audioBuffer: Buffer): LayerResult {
    return this.audioVerifier.verifyAudio(audioBuffer);
  }

  // =========================================================================
  // Phase 1 — Core analysis methods (ported from Python)
  // =========================================================================

  /**
   * Noise uniformity: Laplacian CV + median-residual CV (4x4 grid).
   * Matches Python _analyze_noise_uniformity.
   */
  private analyzeNoiseUniformity(
    gray: Float64Array,
    w: number,
    h: number
  ): Record<string, number> {
    try {
      // Laplacian-based CV ratio on 4x4 grid
      const variances = this.computeGridMetric(gray, w, h, 4, 4, (region, rw, rh) => {
        const lap = laplacian(region, rw, rh);
        return variance(lap);
      });

      const varianceMean = mean(variances);
      const varianceStd = std(variances);
      const cvRatio = varianceStd / (varianceMean + 1e-10);

      // Map cv_ratio [0.1, 0.8] → [20, 100]
      const cvScore = clamp((cvRatio - 0.1) / (0.8 - 0.1) * 80 + 20, 0, 100);

      // Noise-residual CV
      const median = medianFilter(gray, w, h, 2);
      const residual = new Float64Array(w * h);
      for (let i = 0; i < w * h; i++) residual[i] = gray[i] - median[i];

      const residualStds = this.computeGridMetric(residual, w, h, 4, 4, (region) => {
        return std(region);
      });

      const residualMeanStd = mean(residualStds);
      const residualCv = std(residualStds) / (residualMeanStd + 1e-10);

      // Map residual_cv [0.05, 0.5] → [20, 100]
      const residualScore = clamp(
        (residualCv - 0.05) / (0.5 - 0.05) * 80 + 20, 0, 100
      );

      // Variance penalty
      let variancePenalty = 0;
      if (varianceMean < 20) {
        variancePenalty = ((20 - varianceMean) / 20) * 30;
      } else if (varianceMean > 5000) {
        variancePenalty = Math.min(30, ((varianceMean - 5000) / 5000) * 30);
      }

      // Blend: 35% Laplacian CV, 65% residual CV
      const uniformityScore = Math.round(
        clamp(cvScore * 0.35 + residualScore * 0.65 - variancePenalty, 0, 100)
      );

      return {
        variance_mean: varianceMean,
        variance_std: varianceStd,
        cv_ratio: cvRatio,
        residual_cv: residualCv,
        uniformity_score: uniformityScore,
        cv_score: cvScore,
        residual_score: residualScore,
        variance_penalty: variancePenalty,
      };
    } catch {
      return { uniformity_score: 50 };
    }
  }

  /**
   * Noise frequency: 2D FFT → log-magnitude → concentric band energy ratios.
   * Matches Python _analyze_noise_frequency.
   */
  private analyzeNoiseFrequency(
    gray: Float64Array,
    w: number,
    h: number
  ): Record<string, number> {
    try {
      const { re, im, fw, fh } = fft2d(gray, w, h);

      // Compute magnitude
      const mag = new Float64Array(fw * fh);
      for (let i = 0; i < fw * fh; i++) {
        mag[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
      }

      // FFT shift (center DC)
      const magShifted = fftShift(mag, fw, fh);

      // Log magnitude
      const magLog = new Float64Array(fw * fh);
      for (let i = 0; i < fw * fh; i++) {
        magLog[i] = Math.log2(magShifted[i] + 1);
      }

      const centerH = fh >> 1;
      const centerW = fw >> 1;
      const maxRadius = Math.min(fh, fw) >> 1;
      const lowFreqRadius = maxRadius >> 2;
      const highFreqRadius = maxRadius >> 1;

      let lowFreqEnergy = 0, midFreqEnergy = 0, highFreqEnergy = 0;
      for (let y = 0; y < fh; y++) {
        for (let x = 0; x < fw; x++) {
          const dist = Math.sqrt(
            (y - centerH) * (y - centerH) + (x - centerW) * (x - centerW)
          );
          const val = magLog[y * fw + x];
          if (dist <= lowFreqRadius) lowFreqEnergy += val;
          else if (dist <= highFreqRadius) midFreqEnergy += val;
          else highFreqEnergy += val;
        }
      }

      const totalEnergy = lowFreqEnergy + midFreqEnergy + highFreqEnergy + 1e-10;
      const lowFreqRatio = lowFreqEnergy / totalEnergy;
      const midFreqRatio = midFreqEnergy / totalEnergy;
      const highFreqRatio = highFreqEnergy / totalEnergy;

      // Scoring (optimal: low ~70%, mid ~20%, high ~10%)
      const lowScore = 100 - Math.abs(lowFreqRatio * 100 - 70) * 2;
      const midScore = 100 - Math.abs(midFreqRatio * 100 - 20) * 3;
      const highScore = Math.min(100, highFreqRatio * 500);

      const frequencyScore = Math.round(
        clamp(lowScore * 0.4 + midScore * 0.3 + highScore * 0.3, 0, 100)
      );

      return {
        low_freq_ratio: lowFreqRatio,
        mid_freq_ratio: midFreqRatio,
        high_freq_ratio: highFreqRatio,
        frequency_score: frequencyScore,
      };
    } catch {
      return { frequency_score: 50 };
    }
  }

  /**
   * Frequency domain: 2D DCT on 256x256 → DC/AC separation → entropy.
   * Matches Python _analyze_frequency_domain.
   */
  private analyzeFrequencyDomain(
    gray: Float64Array,
    w: number,
    h: number
  ): Record<string, number> {
    try {
      const stdSize = 256;
      const resized = resize(gray, w, h, stdSize, stdSize);
      const dct = dct2d(resized, stdSize, stdSize);

      const dctAbs = new Float64Array(stdSize * stdSize);
      for (let i = 0; i < dctAbs.length; i++) dctAbs[i] = Math.abs(dct[i]);

      const dcEnergy = dctAbs[0];

      // Low freq: top-left 8x8 minus DC
      let lowFreqEnergy = 0;
      for (let y = 0; y < 8; y++) {
        for (let x = 0; x < 8; x++) {
          lowFreqEnergy += dctAbs[y * stdSize + x];
        }
      }
      lowFreqEnergy -= dcEnergy;

      // Mid freq: 8:32, 8:32
      let midFreqEnergy = 0;
      for (let y = 8; y < 32; y++) {
        for (let x = 8; x < 32; x++) {
          midFreqEnergy += dctAbs[y * stdSize + x];
        }
      }

      // Total and high freq
      let totalEnergy = 0;
      for (let i = 0; i < dctAbs.length; i++) totalEnergy += dctAbs[i];
      totalEnergy += 1e-10;

      const highFreqEnergy = totalEnergy - dcEnergy - lowFreqEnergy - midFreqEnergy;

      // Shannon entropy on normalised DCT
      let dctEntropy = 0;
      for (let i = 0; i < dctAbs.length; i++) {
        const p = dctAbs[i] / totalEnergy;
        if (p > 0) dctEntropy -= p * Math.log2(p + 1e-10);
      }

      const highFreqRatio = highFreqEnergy / totalEnergy;
      const acEnergyRatio = (totalEnergy - dcEnergy) / totalEnergy;

      // Scoring
      const hfScore = Math.min(100, highFreqRatio * 300);
      const entropyScore = 100 - Math.abs(dctEntropy - 6) * 10;
      const acScore = Math.min(100, acEnergyRatio * 120);

      const dctScore = Math.round(
        clamp(hfScore * 0.4 + entropyScore * 0.3 + acScore * 0.3, 0, 100)
      );

      return {
        dc_energy: dcEnergy,
        high_freq_ratio: highFreqRatio,
        ac_energy_ratio: acEnergyRatio,
        dct_entropy: dctEntropy,
        dct_score: dctScore,
      };
    } catch {
      return { dct_score: 50 };
    }
  }

  /**
   * Geometric consistency: Canny edges → line estimation + corners.
   * Simplified from Python (no Hough transform or goodFeaturesToTrack).
   */
  private analyzeGeometricConsistency(
    gray: Float64Array,
    w: number,
    h: number,
    graySobel?: { gx: Float64Array; gy: Float64Array }
  ): Record<string, number> {
    try {
      // Multi-threshold Canny
      const edges1 = cannyEdges(gray, w, h, 50, 150);
      const edges2 = cannyEdges(gray, w, h, 100, 200);
      const edges = new Uint8Array(w * h);
      for (let i = 0; i < w * h; i++) {
        edges[i] = edges1[i] | edges2[i];
      }

      // Sobel directions at edge pixels for angle statistics
      const { gx, gy } = graySobel ?? sobelXY(gray, w, h);
      const angles: number[] = [];
      let edgeCount = 0;
      for (let i = 0; i < w * h; i++) {
        if (edges[i]) {
          edgeCount++;
          const angle = Math.atan2(gy[i], gx[i]) * (180 / Math.PI);
          angles.push(angle);
        }
      }

      let angleStd = 50;
      let angleDominance = 0.1;
      let lineCount = 0;
      if (angles.length > 5) {
        angleStd = std(angles);
        const angleHist = histogram(angles, 36, -180, 180);
        let maxBin = 0;
        for (let i = 0; i < 36; i++) {
          if (angleHist[i] > maxBin) maxBin = angleHist[i];
        }
        angleDominance = maxBin / angles.length;
        // Estimate line count from connected edge components (simplified)
        lineCount = this.countEdgeComponents(edges, w, h);
      }

      // Corner detection: high Laplacian response at edge pixels
      const lap = laplacian(gray, w, h);
      let cornerCount = 0;
      const lapThreshold = percentile(
        Float64Array.from(
          Array.from(lap).map(Math.abs).filter((_, i) => edges[i])
        ),
        90
      );
      for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
          const idx = y * w + x;
          if (edges[idx] && Math.abs(lap[idx]) >= lapThreshold) {
            cornerCount++;
          }
        }
      }
      cornerCount = Math.min(cornerCount, 100);
      const cornerDensity = this.calculatePointDensity(
        edges,
        lap,
        lapThreshold,
        w,
        h
      );

      // Scoring (matches Python weights)
      const angleScore = Math.max(0, 100 - angleStd * 1.5);
      const dominanceScore = angleDominance * 100;
      const cornerScore = cornerDensity;
      const lineCountScore =
        lineCount < 50
          ? Math.min(100, lineCount * 2)
          : Math.max(0, 100 - (lineCount - 50));

      const geoScore = Math.round(
        clamp(
          angleScore * 0.3 +
            dominanceScore * 0.2 +
            cornerScore * 0.2 +
            lineCountScore * 0.3,
          0,
          100
        )
      );

      return {
        line_count: lineCount,
        angle_std: angleStd,
        angle_dominance: angleDominance,
        corner_count: cornerCount,
        corner_density: cornerDensity,
        geometric_score: geoScore,
      };
    } catch {
      return { geometric_score: 50 };
    }
  }

  /**
   * Lighting: LAB L-channel → 4×4 grid brightness/contrast + gradient direction.
   * Matches Python _analyze_lighting_consistency.
   */
  private analyzeLighting(
    rgba: Uint8Array,
    gray: Float64Array,
    w: number,
    h: number
  ): Record<string, number> {
    try {
      // Compute L channel from LAB
      const lChannel = new Float64Array(w * h);
      for (let i = 0; i < w * h; i++) {
        const off = i * 4;
        const [L] = rgbToLAB(rgba[off], rgba[off + 1], rgba[off + 2]);
        lChannel[i] = L;
      }

      // 4x4 grid brightness and contrast
      const brightnessValues = this.computeGridMetric(gray, w, h, 4, 4, (region) => mean(region));
      const contrastValues = this.computeGridMetric(gray, w, h, 4, 4, (region) => std(region));

      const brightnessStd = std(brightnessValues);
      const brightnessRange =
        Math.max(...brightnessValues) - Math.min(...brightnessValues);
      const contrastStd = std(contrastValues);

      // Shadow/highlight coverage
      const shadowThreshold = percentile(lChannel, 15);
      const highlightThreshold = percentile(lChannel, 85);
      let shadowCount = 0, highlightCount = 0;
      for (let i = 0; i < w * h; i++) {
        if (lChannel[i] <= shadowThreshold) shadowCount++;
        if (lChannel[i] >= highlightThreshold) highlightCount++;
      }
      const shadowCoverage = shadowCount / (w * h);
      const highlightCoverage = highlightCount / (w * h);

      // Gradient direction consistency on L channel
      const { gx: lgx, gy: lgy } = sobelXY(lChannel, w, h);
      const gradMag = gradientMagnitude(lgx, lgy);
      const gradMagThreshold = percentile(gradMag, 75);

      let dirConsistency = 0.5;
      const significantDirs: number[] = [];
      for (let i = 0; i < w * h; i++) {
        if (gradMag[i] > gradMagThreshold) {
          significantDirs.push(Math.atan2(lgy[i], lgx[i]));
        }
      }
      if (significantDirs.length > 100) {
        const dirStd = std(significantDirs);
        dirConsistency = 1 - Math.min(1, dirStd / Math.PI);
      }

      // Scoring (matches Python weights)
      const brightnessScore = 100 - Math.min(100, brightnessStd * 3);
      const contrastScore = 100 - Math.min(100, contrastStd * 5);
      const directionScore = dirConsistency * 100;
      const balanceScore = Math.max(
        0,
        100 - Math.abs(shadowCoverage - highlightCoverage) * 200
      );

      const lightingScore = Math.round(
        clamp(
          brightnessScore * 0.25 +
            contrastScore * 0.25 +
            directionScore * 0.3 +
            balanceScore * 0.2,
          0,
          100
        )
      );

      return {
        brightness_std: brightnessStd,
        brightness_range: brightnessRange,
        contrast_std: contrastStd,
        shadow_coverage: shadowCoverage,
        highlight_coverage: highlightCoverage,
        gradient_consistency: dirConsistency,
        lighting_score: lightingScore,
      };
    } catch {
      return { lighting_score: 50 };
    }
  }

  /**
   * Texture patterns: Sobel gradient magnitude stats + 4×4 regional CV.
   * Matches Python _analyze_texture_patterns.
   */
  private analyzeTexturePatterns(
    gray: Float64Array,
    w: number,
    h: number,
    graySobel?: { gx: Float64Array; gy: Float64Array }
  ): Record<string, number> {
    try {
      const { gx, gy } = graySobel ?? sobelXY(gray, w, h);
      const gradMag = gradientMagnitude(gx, gy);

      const textureEnergy = mean(gradMag);
      const textureContrast = std(gradMag);
      const gradCv = textureContrast / (textureEnergy + 1e-10);

      // Regional texture variation (4x4 grid)
      const regionEnergies = this.computeGridMetric(gradMag, w, h, 4, 4, (region) => {
        return mean(region);
      });
      const regionCv = cv(regionEnergies);

      // Scoring (matches Python)
      const cvTextureScore = clamp((gradCv - 0.8) / (2.5 - 0.8) * 100, 0, 100);
      const regionScore = clamp((regionCv - 0.1) / (0.7 - 0.1) * 100, 0, 100);
      const energyScore = clamp(
        ((textureEnergy - 5) / (50 - 5)) * 100, 0, 100
      );

      const textureScore = Math.round(
        clamp(
          cvTextureScore * 0.35 + regionScore * 0.35 + energyScore * 0.3,
          0,
          100
        )
      );

      return {
        texture_energy: textureEnergy,
        texture_contrast: textureContrast,
        grad_cv: gradCv,
        region_cv: regionCv,
        texture_score: textureScore,
      };
    } catch {
      return { texture_score: 50 };
    }
  }

  /**
   * Color distribution: HSV histograms → entropy, 4×4 regional HSV std.
   * Matches Python _analyze_color_distribution.
   */
  private analyzeColorDistribution(
    rgba: Uint8Array,
    w: number,
    h: number
  ): Record<string, number> {
    try {
      const n = w * h;
      const hArr = new Float64Array(n);
      const sArr = new Float64Array(n);
      const vArr = new Float64Array(n);
      const aChannelLab = new Float64Array(n);
      const bChannelLab = new Float64Array(n);

      for (let i = 0; i < n; i++) {
        const off = i * 4;
        const r = rgba[off], g = rgba[off + 1], b = rgba[off + 2];
        const [hv, sv, vv] = rgbToHSV(r, g, b);
        hArr[i] = hv;
        sArr[i] = sv;
        vArr[i] = vv;
        const [, aLab, bLab] = rgbToLAB(r, g, b);
        aChannelLab[i] = aLab - 128;
        bChannelLab[i] = bLab - 128;
      }

      // Histograms and entropy
      const hHist = histogram(hArr, 180, 0, 180);
      const sHist = histogram(sArr, 256, 0, 256);
      const vHist = histogram(vArr, 256, 0, 256);
      const hEntropy = entropy(hHist);
      const sEntropy = entropy(sHist);
      const vEntropy = entropy(vHist);

      // Regional HSV std (4x4 grid)
      const hueStds = this.computeGridMetric(hArr, w, h, 4, 4, (region) => mean(region));
      const satStds = this.computeGridMetric(sArr, w, h, 4, 4, (region) => mean(region));
      const valStds = this.computeGridMetric(vArr, w, h, 4, 4, (region) => mean(region));

      const hueStd = std(hueStds);
      const saturationStd = std(satStds);
      const valueStd = std(valStds);

      // LAB color richness
      let aAbsMean = 0, bAbsMean = 0;
      for (let i = 0; i < n; i++) {
        aAbsMean += Math.abs(aChannelLab[i]);
        bAbsMean += Math.abs(bChannelLab[i]);
      }
      aAbsMean /= n;
      bAbsMean /= n;
      const colorRichness = Math.min(100, (aAbsMean + bAbsMean) / 2);

      // Scoring (matches Python)
      const hEntropyScore = 100 - Math.abs(hEntropy - 4.5) * 15;
      const sEntropyScore = Math.min(100, sEntropy * 12);
      const vEntropyScore = Math.min(100, vEntropy * 12);
      const hueVarScore = 100 - Math.min(100, hueStd * 2);
      const satVarScore = 100 - Math.min(100, saturationStd * 1.5);

      const colorScore = Math.round(
        clamp(
          hEntropyScore * 0.15 +
            sEntropyScore * 0.15 +
            vEntropyScore * 0.15 +
            hueVarScore * 0.15 +
            satVarScore * 0.2 +
            colorRichness * 0.2,
          0,
          100
        )
      );

      return {
        hue_entropy: hEntropy,
        saturation_entropy: sEntropy,
        value_entropy: vEntropy,
        hue_std: hueStd,
        saturation_std: saturationStd,
        value_std: valueStd,
        color_richness: colorRichness,
        color_score: colorScore,
      };
    } catch {
      return { color_score: 50 };
    }
  }

  /**
   * Compression artifacts: multi-quality ELA + regional 8×8 grid.
   * Delegates to ErrorLevelAnalysis.analyzeDetailed().
   */
  private analyzeCompressionArtifacts(
    imageBuffer: Buffer,
    img: ImageData
  ): Record<string, unknown> {
    try {
      const result: ELADetailedResult = ErrorLevelAnalysis.analyzeDetailed(
        imageBuffer,
        { data: img.data, width: img.width, height: img.height }
      );
      return {
        ela_scores: result.elaScores,
        ela_variance: result.elaVariance,
        ela_mean: result.elaMean,
        ela_range: result.elaRange,
        ela_gradient: result.elaGradient,
        regional_ela_cv: result.regionalElaCv,
        suspicious_regions: result.suspiciousRegions,
        compression_score: result.compressionScore,
      };
    } catch {
      return { compression_score: 50 };
    }
  }

  // =========================================================================
  // Phase 2 — Advanced detection enhancements
  // =========================================================================

  /**
   * Spatial-aware splice detection via noise inconsistency mapping.
   * Computes per-block noise level estimation and measures spatial variation.
   */
  private analyzeNoiseConsistencyMap(
    gray: Float64Array,
    w: number,
    h: number
  ): Record<string, number> {
    try {
      const blockSize = 32;
      const stride = 8;
      const noiseLevels: number[] = [];
      const positions: Array<[number, number]> = [];

      for (let y = 0; y <= h - blockSize; y += stride) {
        for (let x = 0; x <= w - blockSize; x += stride) {
          // Extract block and compute local noise via high-pass filter
          const block = new Float64Array(blockSize * blockSize);
          for (let by = 0; by < blockSize; by++) {
            for (let bx = 0; bx < blockSize; bx++) {
              block[by * blockSize + bx] = gray[(y + by) * w + (x + bx)];
            }
          }
          const lap = laplacian(block, blockSize, blockSize);
          const noiseStd = std(lap);
          noiseLevels.push(noiseStd);
          positions.push([x, y]);
        }
      }

      if (noiseLevels.length < 4) {
        return { noise_map_score: 50 };
      }

      const noiseMapCv = cv(noiseLevels);
      const noiseLevelRange =
        Math.max(...noiseLevels) - Math.min(...noiseLevels);

      // Connected component analysis on thresholded noise heatmap
      const noiseMapMean = mean(noiseLevels);
      const noiseMapStd = std(noiseLevels);
      const threshold = noiseMapMean + 1.5 * noiseMapStd;
      let islandCount = 0;
      const visited = new Uint8Array(noiseLevels.length);
      const cols = Math.floor((w - blockSize) / stride) + 1;

      for (let i = 0; i < noiseLevels.length; i++) {
        if (noiseLevels[i] > threshold && !visited[i]) {
          islandCount++;
          // BFS to mark connected outlier blocks
          const queue = [i];
          visited[i] = 1;
          while (queue.length > 0) {
            const cur = queue.shift()!;
            const cx = cur % cols;
            const cy = Math.floor(cur / cols);
            for (const [dx, dy] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
              const nx = cx + dx;
              const ny = cy + dy;
              if (nx >= 0 && nx < cols && ny >= 0) {
                const ni = ny * cols + nx;
                if (ni < noiseLevels.length && !visited[ni] && noiseLevels[ni] > threshold) {
                  visited[ni] = 1;
                  queue.push(ni);
                }
              }
            }
          }
        }
      }

      // Score: high CV or many islands → manipulation → lower score
      const cvPenalty = clamp(noiseMapCv * 100, 0, 60);
      const islandPenalty = Math.min(40, islandCount * 10);
      const noiseMapScore = Math.round(clamp(100 - cvPenalty - islandPenalty, 0, 100));

      return {
        noise_map_cv: noiseMapCv,
        noise_island_count: islandCount,
        noise_level_range: noiseLevelRange,
        noise_map_score: noiseMapScore,
      };
    } catch {
      return { noise_map_score: 50 };
    }
  }

  /**
   * JPEG ghost detection: recompress at multiple qualities, find regions
   * that converge at different quality levels (indicating copy-paste from
   * a differently-compressed source).
   */
  private analyzeJpegGhosts(
    imageBuffer: Buffer,
    img: ImageData
  ): Record<string, number> {
    try {
      const { data, width: w, height: h } = img;
      const gridN = 8;
      const qualities = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95];

      // For each quality, compute per-region mean difference
      const regionOptimalQualities = new Float64Array(gridN * gridN);
      const regionMinDiffs = new Float64Array(gridN * gridN);
      regionMinDiffs.fill(Infinity);

      for (const q of qualities) {
        const encoded = this.jpegRoundTrip(data, w, h, q);
        if (!encoded) continue;

        for (let i = 0; i < gridN; i++) {
          for (let j = 0; j < gridN; j++) {
            const y0 = Math.floor((i * h) / gridN);
            const y1 = Math.floor(((i + 1) * h) / gridN);
            const x0 = Math.floor((j * w) / gridN);
            const x1 = Math.floor(((j + 1) * w) / gridN);
            let sum = 0, count = 0;
            for (let y = y0; y < y1; y++) {
              for (let x = x0; x < x1; x++) {
                const off = (y * w + x) * 4;
                sum +=
                  (Math.abs(data[off] - encoded[off]) +
                    Math.abs(data[off + 1] - encoded[off + 1]) +
                    Math.abs(data[off + 2] - encoded[off + 2])) / 3;
                count++;
              }
            }
            const regionMean = count > 0 ? sum / count : 0;
            const rIdx = i * gridN + j;
            if (regionMean < regionMinDiffs[rIdx]) {
              regionMinDiffs[rIdx] = regionMean;
              regionOptimalQualities[rIdx] = q;
            }
          }
        }
      }

      // Compute spread of optimal qualities across regions
      const qualitySpread =
        Math.max(...Array.from(regionOptimalQualities)) -
        Math.min(...Array.from(regionOptimalQualities));

      // Count regions whose optimal quality differs from the mode
      const qHist = new Map<number, number>();
      for (let i = 0; i < regionOptimalQualities.length; i++) {
        const q = regionOptimalQualities[i];
        qHist.set(q, (qHist.get(q) || 0) + 1);
      }
      let modeQ = 0, modeCount = 0;
      for (const [q, c] of qHist) {
        if (c > modeCount) { modeQ = q; modeCount = c; }
      }
      let ghostRegionCount = 0;
      for (let i = 0; i < regionOptimalQualities.length; i++) {
        if (Math.abs(regionOptimalQualities[i] - modeQ) > 10) {
          ghostRegionCount++;
        }
      }

      // Score: high spread or many ghost regions → lower score
      const spreadPenalty = clamp(qualitySpread * 2, 0, 60);
      const ghostPenalty = Math.min(40, ghostRegionCount * 5);
      const ghostScore = Math.round(clamp(100 - spreadPenalty - ghostPenalty, 0, 100));

      return {
        ghost_quality_spread: qualitySpread,
        ghost_region_count: ghostRegionCount,
        ghost_score: ghostScore,
      };
    } catch {
      return { ghost_score: 50 };
    }
  }

  /**
   * GAN spectral fingerprinting: 2D power spectrum → radial profile →
   * power-law fit → residual peaks.
   */
  private analyzeSpectralFingerprint(
    gray: Float64Array,
    w: number,
    h: number
  ): Record<string, number> {
    try {
      // Resize to 256×256 for consistent analysis
      const sz = 256;
      const resized = resize(gray, w, h, sz, sz);
      const { re, im, fw, fh } = fft2d(resized, sz, sz);

      // Power spectrum |FFT|²
      const power = new Float64Array(fw * fh);
      for (let i = 0; i < fw * fh; i++) {
        power[i] = re[i] * re[i] + im[i] * im[i];
      }
      const powerShifted = fftShift(power, fw, fh);

      // Radial profile (azimuthal average)
      const centerX = fw >> 1;
      const centerY = fh >> 1;
      const maxR = Math.min(centerX, centerY);
      const radialSum = new Float64Array(maxR);
      const radialCount = new Float64Array(maxR);

      for (let y = 0; y < fh; y++) {
        for (let x = 0; x < fw; x++) {
          const r = Math.sqrt(
            (x - centerX) * (x - centerX) + (y - centerY) * (y - centerY)
          );
          const ri = Math.floor(r);
          if (ri > 0 && ri < maxR) {
            radialSum[ri] += powerShifted[y * fw + x];
            radialCount[ri]++;
          }
        }
      }

      const radialProfile = new Float64Array(maxR);
      for (let r = 1; r < maxR; r++) {
        radialProfile[r] =
          radialCount[r] > 0
            ? Math.log10(radialSum[r] / radialCount[r] + 1e-10)
            : 0;
      }

      // Fit power law: log(P(f)) = a - b*log(f)
      // Simple linear regression on log-log
      let sumX = 0, sumY = 0, sumXX = 0, sumXY = 0, n = 0;
      for (let r = 2; r < maxR; r++) {
        if (radialProfile[r] > 0) {
          const logR = Math.log10(r);
          sumX += logR;
          sumY += radialProfile[r];
          sumXX += logR * logR;
          sumXY += logR * radialProfile[r];
          n++;
        }
      }

      let spectralResidualEnergy = 0;
      let powerLawFitR2 = 0;
      let spectralPeakCount = 0;

      if (n > 2) {
        const b = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX + 1e-10);
        const a = (sumY - b * sumX) / n;

        // Compute residuals and R²
        let ssTot = 0, ssRes = 0;
        const meanY = sumY / n;
        const residuals = new Float64Array(maxR);
        for (let r = 2; r < maxR; r++) {
          if (radialProfile[r] > 0) {
            const predicted = a + b * Math.log10(r);
            const res = radialProfile[r] - predicted;
            residuals[r] = res;
            ssRes += res * res;
            ssTot += (radialProfile[r] - meanY) * (radialProfile[r] - meanY);
          }
        }
        powerLawFitR2 = 1 - ssRes / (ssTot + 1e-10);
        spectralResidualEnergy = ssRes / n;

        // Count peaks in the residual (local maxima > 1.5 std)
        const resStd = Math.sqrt(ssRes / n);
        for (let r = 3; r < maxR - 1; r++) {
          if (
            residuals[r] > 1.5 * resStd &&
            residuals[r] > residuals[r - 1] &&
            residuals[r] > residuals[r + 1]
          ) {
            spectralPeakCount++;
          }
        }
      }

      // Score: high residual, low R², many peaks → GAN → lower score
      const residualPenalty = clamp(spectralResidualEnergy * 200, 0, 40);
      const fitPenalty = clamp((1 - powerLawFitR2) * 30, 0, 30);
      const peakPenalty = Math.min(30, spectralPeakCount * 5);
      const spectralScore = Math.round(
        clamp(100 - residualPenalty - fitPenalty - peakPenalty, 0, 100)
      );

      return {
        spectral_residual_energy: spectralResidualEnergy,
        spectral_peak_count: spectralPeakCount,
        power_law_fit_r2: powerLawFitR2,
        spectral_score: spectralScore,
      };
    } catch {
      return { spectral_score: 50 };
    }
  }

  /**
   * Cross-channel correlation: R/G/B noise residuals → pairwise correlations.
   * Real cameras produce correlated channel noise; GANs produce independent noise.
   */
  private analyzeChannelCorrelation(
    rgba: Uint8Array,
    w: number,
    h: number
  ): Record<string, number> {
    try {
      const n = w * h;

      // Extract per-channel grayscale arrays
      const rCh = new Float64Array(n);
      const gCh = new Float64Array(n);
      const bCh = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const off = i * 4;
        rCh[i] = rgba[off];
        gCh[i] = rgba[off + 1];
        bCh[i] = rgba[off + 2];
      }

      // Noise residual = original - median filtered
      const rMedian = medianFilter(rCh, w, h, 2);
      const gMedian = medianFilter(gCh, w, h, 2);
      const bMedian = medianFilter(bCh, w, h, 2);

      const rNoise = new Float64Array(n);
      const gNoise = new Float64Array(n);
      const bNoise = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        rNoise[i] = rCh[i] - rMedian[i];
        gNoise[i] = gCh[i] - gMedian[i];
        bNoise[i] = bCh[i] - bMedian[i];
      }

      // Pairwise Pearson correlation
      const corrRG = this.pearsonCorr(rNoise, gNoise);
      const corrRB = this.pearsonCorr(rNoise, bNoise);
      const corrGB = this.pearsonCorr(gNoise, bNoise);

      // Bayer pattern periodicity: check for 2x2 periodicity in noise residual
      // Average the green channel noise residual in a 2x2 pattern
      let bayerSum = 0, bayerCount = 0;
      for (let y = 0; y < h - 1; y += 2) {
        for (let x = 0; x < w - 1; x += 2) {
          const tl = gNoise[y * w + x];
          const tr = gNoise[y * w + x + 1];
          const bl = gNoise[(y + 1) * w + x];
          const br = gNoise[(y + 1) * w + x + 1];
          // In Bayer CFA, green pixels at (0,0) and (1,1) differ from (0,1) and (1,0)
          bayerSum += Math.abs((tl + br) - (tr + bl));
          bayerCount++;
        }
      }
      const bayerPeriodicity = bayerCount > 0 ? bayerSum / bayerCount : 0;

      // Score: high correlation + Bayer periodicity → real camera → higher score
      const avgCorr = (Math.abs(corrRG) + Math.abs(corrRB) + Math.abs(corrGB)) / 3;
      // Real: avgCorr 0.4-0.8, GAN: 0.1-0.3
      const corrScore = clamp((avgCorr - 0.1) / (0.6 - 0.1) * 80 + 20, 0, 100);
      const bayerScore = clamp(bayerPeriodicity * 20, 0, 30);
      const correlationScore = Math.round(clamp(corrScore * 0.8 + bayerScore * 0.2, 0, 100));

      return {
        channel_noise_corr_rg: corrRG,
        channel_noise_corr_rb: corrRB,
        channel_noise_corr_gb: corrGB,
        bayer_periodicity: bayerPeriodicity,
        correlation_score: correlationScore,
      };
    } catch {
      return { correlation_score: 50 };
    }
  }

  // =========================================================================
  // Phase 3 — Advanced image analysis methods
  // =========================================================================

  /**
   * Benford's Law analysis of DCT coefficients.
   * First significant digits of AC coefficients in 8×8 block DCTs
   * of natural images follow a generalized Benford distribution.
   */
  private analyzeBenfordDct(
    gray: Float64Array,
    w: number,
    h: number
  ): Record<string, number> {
    try {
      // Collect first significant digits from 8×8 block DCT AC coefficients
      const digits: number[] = [];
      for (let y = 0; y <= h - 8; y += 8) {
        for (let x = 0; x <= w - 8; x += 8) {
          // Extract 8×8 block
          const block = new Float64Array(64);
          for (let by = 0; by < 8; by++) {
            for (let bx = 0; bx < 8; bx++) {
              block[by * 8 + bx] = gray[(y + by) * w + (x + bx)];
            }
          }
          // Compute DCT of this 8×8 block
          const dctBlock = dct2d(block, 8, 8);
          // Skip DC (index 0), process AC coefficients
          for (let i = 1; i < 64; i++) {
            const absval = Math.abs(dctBlock[i]);
            if (absval >= 1.0) {
              // Extract first significant digit
              const d = parseInt(absval.toExponential(6).charAt(0), 10);
              if (d >= 1 && d <= 9) {
                digits.push(d);
              }
            }
          }
        }
      }

      if (digits.length < 100) {
        return { benford_score: 50 };
      }

      // Observed distribution
      const counts = new Float64Array(9);
      for (const d of digits) counts[d - 1]++;
      const total = digits.length;
      const observed = new Float64Array(9);
      for (let i = 0; i < 9; i++) observed[i] = counts[i] / total;

      // Expected Benford distribution
      const expected = new Float64Array(9);
      for (let d = 1; d <= 9; d++) {
        expected[d - 1] = Math.log10(1 + 1 / d);
      }

      // KL-divergence: D_KL(observed || expected)
      let klDiv = 0;
      for (let i = 0; i < 9; i++) {
        klDiv += (observed[i] + 1e-10) * Math.log((observed[i] + 1e-10) / (expected[i] + 1e-10));
      }

      // Chi-squared statistic
      let chi2 = 0;
      for (let i = 0; i < 9; i++) {
        const diff = counts[i] - total * expected[i];
        chi2 += (diff * diff) / (total * expected[i] + 1e-10);
      }

      // Score: cap at 60 (neutral-default). Only extreme KL is diagnostic.
      const benfordScore = Math.round(clamp(60 - klDiv * 50, 20, 60));

      return {
        benford_kl_divergence: klDiv,
        benford_chi_squared: chi2,
        benford_digit_count: digits.length,
        benford_score: benfordScore,
      };
    } catch {
      return { benford_score: 50 };
    }
  }

  /**
   * Wavelet spectral ratio for diffusion model detection.
   * Uses a simplified Haar wavelet decomposition (3 levels).
   * Measures cross-scale energy decay — diffusion models show
   * faster-than-expected falloff in fine-scale detail.
   */
  private analyzeWaveletSpectralRatio(
    gray: Float64Array,
    w: number,
    h: number
  ): Record<string, number> {
    try {
      // Resize to 256×256 for consistent analysis
      const sz = 256;
      let current = resize(gray, w, h, sz, sz);
      let cw = sz;
      let ch = sz;

      // 3-level Haar wavelet decomposition
      const levelEnergies: Array<{ h: number; v: number; d: number; total: number }> = [];
      let llEnergy = 0;

      for (let level = 0; level < 3; level++) {
        const halfW = cw >> 1;
        const halfH = ch >> 1;
        if (halfW < 4 || halfH < 4) break;

        // Row-wise transform
        const rowTransformed = new Float64Array(cw * ch);
        for (let y = 0; y < ch; y++) {
          for (let x = 0; x < halfW; x++) {
            const a = current[y * cw + 2 * x];
            const b = current[y * cw + 2 * x + 1];
            rowTransformed[y * cw + x] = (a + b) * 0.5;           // Low-pass
            rowTransformed[y * cw + halfW + x] = (a - b) * 0.5;   // High-pass
          }
        }

        // Column-wise transform
        const ll = new Float64Array(halfW * halfH);
        const lh = new Float64Array(halfW * halfH);  // Horizontal detail
        const hl = new Float64Array(halfW * halfH);  // Vertical detail
        const hh = new Float64Array(halfW * halfH);  // Diagonal detail

        for (let x = 0; x < halfW; x++) {
          for (let y = 0; y < halfH; y++) {
            const a = rowTransformed[(2 * y) * cw + x];
            const b = rowTransformed[(2 * y + 1) * cw + x];
            ll[y * halfW + x] = (a + b) * 0.5;
            hl[y * halfW + x] = (a - b) * 0.5;

            const c = rowTransformed[(2 * y) * cw + halfW + x];
            const d = rowTransformed[(2 * y + 1) * cw + halfW + x];
            lh[y * halfW + x] = (c + d) * 0.5;
            hh[y * halfW + x] = (c - d) * 0.5;
          }
        }

        // Compute energies for this level
        const energyOf = (arr: Float64Array) => {
          let sum = 0;
          for (let i = 0; i < arr.length; i++) sum += arr[i] * arr[i];
          return sum / arr.length;
        };

        const eH = energyOf(hl);
        const eV = energyOf(lh);
        const eD = energyOf(hh);
        levelEnergies.push({ h: eH, v: eV, d: eD, total: eH + eV + eD });

        // Next level operates on LL subband
        current = ll;
        cw = halfW;
        ch = halfH;
      }

      // LL energy from the final approximation
      llEnergy = 0;
      for (let i = 0; i < current.length; i++) llEnergy += current[i] * current[i];
      llEnergy /= current.length;

      if (levelEnergies.length < 2) {
        return { wavelet_score: 50 };
      }

      // Finest-scale detail energy (last level)
      const finest = levelEnergies[levelEnergies.length - 1];

      // Wavelet Spectral Ratio
      const wsr = finest.total / (llEnergy + 1e-10);

      // Cross-scale energy decay
      const decayRates: number[] = [];
      for (let i = 0; i < levelEnergies.length - 1; i++) {
        const coarser = levelEnergies[i].total;
        const finer = levelEnergies[i + 1].total;
        if (coarser > 1e-10) decayRates.push(finer / coarser);
      }
      const avgDecay = decayRates.length > 0
        ? decayRates.reduce((a, b) => a + b, 0) / decayRates.length
        : 1.0;

      // Diagonal-to-total ratio at finest scale
      const diagRatio = finest.d / (finest.total + 1e-10);

      // Score
      const decayScore = clamp((avgDecay - 0.01) / (0.8 - 0.01) * 80 + 20, 0, 100);
      const wsrScore = clamp(wsr * 5000, 0, 40);
      const waveletScore = Math.round(clamp(decayScore * 0.8 + wsrScore * 0.2, 0, 100));

      return {
        wavelet_spectral_ratio: wsr,
        finest_detail_energy: finest.total,
        approx_energy: llEnergy,
        cross_scale_decay: avgDecay,
        diag_ratio: diagRatio,
        wavelet_score: waveletScore,
      };
    } catch {
      return { wavelet_score: 50 };
    }
  }

  /**
   * Blocking Artifact Grid (BAG) analysis for double JPEG detection.
   * Computes blockiness B(offset) for all 8 possible offsets,
   * detects primary and secondary grids.
   */
  private analyzeBlockingArtifactGrid(
    gray: Float64Array,
    w: number,
    h: number
  ): Record<string, unknown> {
    try {
      if (h < 16 || w < 16) return { bag_score: 50 };

      // Compute blockiness for each offset (0..7) in both directions
      const hBlockiness = new Float64Array(8);
      const vBlockiness = new Float64Array(8);

      for (let offset = 0; offset < 8; offset++) {
        // Horizontal blockiness
        let hSum = 0, hCount = 0;
        for (let x = offset; x < w - 1; x += 8) {
          let colDiff = 0;
          for (let y = 0; y < h; y++) {
            colDiff += Math.abs(gray[y * w + x] - gray[y * w + x + 1]);
          }
          hSum += colDiff / h;
          hCount++;
        }
        hBlockiness[offset] = hCount > 0 ? hSum / hCount : 0;

        // Vertical blockiness
        let vSum = 0, vCount = 0;
        for (let y = offset; y < h - 1; y += 8) {
          let rowDiff = 0;
          for (let x = 0; x < w; x++) {
            rowDiff += Math.abs(gray[y * w + x] - gray[(y + 1) * w + x]);
          }
          vSum += rowDiff / w;
          vCount++;
        }
        vBlockiness[offset] = vCount > 0 ? vSum / vCount : 0;
      }

      // Combined blockiness per offset
      const blockiness = new Float64Array(8);
      for (let i = 0; i < 8; i++) blockiness[i] = hBlockiness[i] + vBlockiness[i];

      // Primary grid: offset with highest blockiness
      let primaryOffset = 0;
      for (let i = 1; i < 8; i++) {
        if (blockiness[i] > blockiness[primaryOffset]) primaryOffset = i;
      }
      const primaryStrength = blockiness[primaryOffset];

      // Secondary grid: highest excluding primary
      let secondaryOffset = -1;
      let secondaryStrength = 0;
      for (let i = 0; i < 8; i++) {
        if (i !== primaryOffset && blockiness[i] > secondaryStrength) {
          secondaryStrength = blockiness[i];
          secondaryOffset = i;
        }
      }

      // Tertiary average (excluding primary and secondary)
      let tertiarySum = 0, tertiaryCount = 0;
      for (let i = 0; i < 8; i++) {
        if (i !== primaryOffset && i !== secondaryOffset) {
          tertiarySum += blockiness[i];
          tertiaryCount++;
        }
      }
      const tertiaryAvg = tertiaryCount > 0 ? tertiarySum / tertiaryCount : 0;
      const baseline = tertiaryAvg;

      const primarySnr = primaryStrength / (baseline + 1e-10);
      const secondarySnr = secondaryStrength / (baseline + 1e-10);
      const secondaryAboveTertiary = secondaryStrength / (tertiaryAvg + 1e-10);
      const gridDiff = Math.abs(primaryOffset - secondaryOffset);

      const dualGrid =
        gridDiff > 0 &&
        secondaryAboveTertiary > 1.5 &&
        primarySnr > 2.0 &&
        secondarySnr > 1.8;

      // Score: neutral (50) when no dual grid detected
      let bagScore: number;
      if (dualGrid) {
        const dualPenalty = Math.min(50, (secondaryAboveTertiary - 1.0) * 30);
        bagScore = Math.round(clamp(100 - dualPenalty - 20, 0, 100));
      } else {
        bagScore = 50;
      }

      return {
        primary_grid_offset: primaryOffset,
        primary_grid_snr: primarySnr,
        secondary_grid_offset: secondaryOffset,
        secondary_grid_snr: secondarySnr,
        dual_grid_detected: dualGrid,
        grid_alignment_diff: gridDiff,
        bag_score: bagScore,
      };
    } catch {
      return { bag_score: 50 };
    }
  }

  /**
   * CFA / demosaicing peak detection.
   * Real cameras leave periodic correlation peaks in the 2D FFT of
   * inter-channel difference images at Bayer frequencies.
   */
  private analyzeCfaDemosaicing(
    rgba: Uint8Array,
    w: number,
    h: number
  ): Record<string, number> {
    try {
      if (h < 32 || w < 32) return { cfa_score: 50 };

      const n = w * h;
      // Extract R, G, B channels as float arrays
      const rCh = new Float64Array(n);
      const gCh = new Float64Array(n);
      const bCh = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const off = i * 4;
        rCh[i] = rgba[off];
        gCh[i] = rgba[off + 1];
        bCh[i] = rgba[off + 2];
      }

      // Inter-channel difference images: G-R and G-B
      const diffGR = new Float64Array(n);
      const diffGB = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        diffGR[i] = gCh[i] - rCh[i];
        diffGB[i] = gCh[i] - bCh[i];
      }

      const peakSnrs: number[] = [];

      for (const diffImg of [diffGR, diffGB]) {
        // 2D FFT
        const { re, im, fw, fh } = fft2d(diffImg, w, h);

        // Magnitude (shifted)
        const mag = new Float64Array(fw * fh);
        for (let i = 0; i < fw * fh; i++) {
          mag[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
        }
        const magShifted = fftShift(mag, fw, fh);

        const cy = fh >> 1;
        const cx = fw >> 1;

        // Peak energy at expected Bayer locations
        const peakWindow = 3;
        const peakEnergy = (py: number, px: number): number => {
          let maxVal = 0;
          for (let dy = -peakWindow; dy <= peakWindow; dy++) {
            for (let dx = -peakWindow; dx <= peakWindow; dx++) {
              const ny = py + dy;
              const nx = px + dx;
              if (ny >= 0 && ny < fh && nx >= 0 && nx < fw) {
                maxVal = Math.max(maxVal, magShifted[ny * fw + nx]);
              }
            }
          }
          return maxVal;
        };

        // Background energy (mid-frequency ring)
        const rInner = Math.min(fh, fw) / 6;
        const rOuter = Math.min(fh, fw) / 3;
        const bgValues: number[] = [];
        for (let y = 0; y < fh; y++) {
          for (let x = 0; x < fw; x++) {
            const dist = Math.sqrt((y - cy) * (y - cy) + (x - cx) * (x - cx));
            if (dist > rInner && dist < rOuter) {
              bgValues.push(magShifted[y * fw + x]);
            }
          }
        }
        // Median of background
        bgValues.sort((a, b) => a - b);
        const background = bgValues.length > 0 ? bgValues[bgValues.length >> 1] : 1.0;

        // Check all Bayer peak locations
        const peaks: number[] = [];
        // (π,π) corners
        for (const [py, px] of [[0, 0], [0, fw - 1], [fh - 1, 0], [fh - 1, fw - 1]]) {
          peaks.push(peakEnergy(py, px));
        }
        // (π,0) top/bottom center
        for (const [py, px] of [[0, cx], [fh - 1, cx]]) {
          peaks.push(peakEnergy(py, px));
        }
        // (0,π) left/right center
        for (const [py, px] of [[cy, 0], [cy, fw - 1]]) {
          peaks.push(peakEnergy(py, px));
        }

        const maxPeak = Math.max(...peaks);
        const snr = maxPeak / (background + 1e-10);
        peakSnrs.push(snr);
      }

      const avgSnr = peakSnrs.reduce((a, b) => a + b, 0) / peakSnrs.length;
      const cfaPresent = avgSnr > 2.0;

      // Cap at 55: CFA peaks can be mimicked by geometric structure
      const snrScore = clamp((avgSnr - 1.0) / (5.0 - 1.0) * 35 + 20, 0, 55);
      const cfaScore = Math.round(clamp(snrScore, 0, 55));

      return {
        cfa_peak_snr_gr: peakSnrs[0] ?? 0,
        cfa_peak_snr_gb: peakSnrs[1] ?? 0,
        cfa_avg_snr: avgSnr,
        cfa_present: cfaPresent ? 1 : 0,
        cfa_score: cfaScore,
      };
    } catch {
      return { cfa_score: 50 };
    }
  }

  // =========================================================================
  // Decision logic (matches Python)
  // =========================================================================

  // Declarative indicator table for simple threshold checks.
  // Each entry: [resultKey, metricKey, op, threshold|null, default, weight]
  // null threshold → resolved from THRESHOLDS at runtime.
  private static readonly INDICATOR_TABLE: Array<
    [string, string, '<' | '>', number | null, number, number]
  > = [
    ['noise_uniformity', 'residual_cv', '<', 0.10, 0.5, 1],
    ['noise_uniformity', 'uniformity_score', '<', 15, 50, 1],
    ['frequency_analysis', 'dct_score', '<', null, 50, 1],
    ['lighting_analysis', 'lighting_score', '<', null, 50, 1],
    ['texture_analysis', 'texture_score', '<', null, 50, 1],
    ['texture_analysis', 'texture_score', '<', 25, 50, 1],
    ['geometric_consistency', 'geometric_score', '<', null, 50, 1],
    ['color_analysis', 'color_score', '<', null, 50, 1],
    ['compression_analysis', 'compression_score', '<', null, 50, 1],
    ['noise_consistency_map', 'noise_map_cv', '>', null, 0.3, 1],
    ['jpeg_ghost', 'ghost_quality_spread', '>', null, 0, 2],
    ['spectral_fingerprint', 'spectral_residual_energy', '>', null, 0, 1],
    ['benford_dct', 'benford_kl_divergence', '>', 0.8, 0, 1],
    ['wavelet_ratio', 'cross_scale_decay', '<', 0.10, 0.5, 1],
    ['cfa_demosaicing', 'cfa_avg_snr', '<', 1.5, 2.5, 1],
  ];

  // Map from null threshold entries to THRESHOLDS keys.
  private static readonly THRESHOLD_KEY_MAP: Record<string, string> = {
    'frequency_analysis.dct_score': 'dct_low',
    'lighting_analysis.lighting_score': 'lighting_low',
    'texture_analysis.texture_score': 'texture_low',
    'geometric_consistency.geometric_score': 'geometric_low',
    'color_analysis.color_score': 'color_low',
    'compression_analysis.compression_score': 'compression_low',
    'noise_consistency_map.noise_map_cv': 'noise_map_cv_high',
    'jpeg_ghost.ghost_quality_spread': 'ghost_spread_high',
    'spectral_fingerprint.spectral_residual_energy': 'spectral_residual_high',
  };

  private countAiIndicators(results: Record<string, Record<string, unknown>>): number {
    let indicators = 0;

    // Simple threshold checks from declarative table
    for (const [resultKey, metricKey, op, threshold, def, weight] of PhysicsVerifier.INDICATOR_TABLE) {
      const val = ((results[resultKey] || {})[metricKey] as number) ?? def;
      let thresh = threshold;
      if (thresh === null) {
        const key = PhysicsVerifier.THRESHOLD_KEY_MAP[`${resultKey}.${metricKey}`];
        thresh = key ? (THRESHOLDS as Record<string, number>)[key] ?? 0 : 0;
      }
      if (op === '<' && val < thresh) indicators += weight;
      else if (op === '>' && val > thresh) indicators += weight;
    }

    // Compound rule: noise CV + variance floor
    const noise = results['noise_uniformity'] || {};
    const cvRatioVal = (noise['cv_ratio'] as number) ?? 0.5;
    const varianceMeanVal = (noise['variance_mean'] as number) ?? 50;
    if (cvRatioVal < THRESHOLDS.noise_cv_low && varianceMeanVal > THRESHOLDS.noise_variance_floor) {
      indicators++;
    }

    // Two-branch rule: regional ELA CV
    const regionalCv = ((results['compression_analysis'] || {})['regional_ela_cv'] as number) ?? 0.35;
    if (regionalCv < THRESHOLDS.regional_ela_cv_low) {
      indicators += 3;
    } else if (regionalCv > THRESHOLDS.regional_ela_cv_high) {
      indicators++;
    }

    // Derived metric: cross-channel correlation
    const avgCorr =
      (Math.abs(((results['channel_correlation'] || {})['channel_noise_corr_rg'] as number) ?? 0.5) +
        Math.abs(((results['channel_correlation'] || {})['channel_noise_corr_rb'] as number) ?? 0.5) +
        Math.abs(((results['channel_correlation'] || {})['channel_noise_corr_gb'] as number) ?? 0.5)) / 3;
    if (avgCorr < THRESHOLDS.channel_corr_low) indicators += 2;

    // Boolean indicator: BAG dual grid
    const dualGrid = (results['blocking_grid'] || {})['dual_grid_detected'] as boolean;
    if (dualGrid) indicators++;

    // Combined GAN rule: low ELA CV + low texture
    const textureScore = ((results['texture_analysis'] || {})['texture_score'] as number) ?? 50;
    if (regionalCv < THRESHOLDS.regional_ela_cv_low && textureScore < THRESHOLDS.texture_low) {
      indicators++;
    }

    return indicators;
  }

  private calculateCompositeScore(
    results: Record<string, Record<string, unknown>>
  ): number {
    let totalScore = 0;
    let totalWeight = 0;

    for (const [analysisType, weight] of Object.entries(WEIGHTS)) {
      const result = results[analysisType];
      if (result) {
        const scoreKey = SCORE_KEY_MAP[analysisType] || 'score';
        const score = result[scoreKey];
        if (typeof score === 'number') {
          totalScore += score * weight;
          totalWeight += weight;
        }
      }
    }

    if (totalWeight > 0) {
      return Math.max(0, Math.min(100, Math.round(totalScore / totalWeight)));
    }
    return 50;
  }

  private determineResult(
    score: number,
    aiProbability: number,
    results: Record<string, Record<string, unknown>>
  ): { passed: boolean; adjustedScore: number; warning: string | null } {
    // --- Targeted manipulation detection ---
    // High regional ELA CV (>0.5) + non-zero JPEG ghost spread (>=3) is
    // specific to copy-paste / splice manipulation.
    const regionalCv = ((results['compression_analysis'] || {})['regional_ela_cv'] as number) ?? 0.35;
    const ghostSpread = ((results['jpeg_ghost'] || {})['ghost_quality_spread'] as number) ?? 0;

    if (regionalCv > 0.5 && ghostSpread >= 3) {
      const adjusted = Math.round(score * 0.7);
      return {
        passed: adjusted >= 50,
        adjustedScore: adjusted,
        warning: 'Compression inconsistencies detected - likely manipulated (splice/edit)',
      };
    }

    // --- General probability-based brackets ---
    if (aiProbability >= 0.4) {
      return {
        passed: false,
        adjustedScore: Math.round(score * 0.5),
        warning: 'Strong physical inconsistencies detected - likely AI-generated',
      };
    }
    if (aiProbability >= 0.3) {
      const adjusted = Math.round(score * 0.7);
      return {
        passed: adjusted >= 50,
        adjustedScore: adjusted,
        warning: 'Some physical inconsistencies detected - possibly AI-generated',
      };
    }
    if (aiProbability >= 0.1) {
      const passed = score >= 45;
      return {
        passed,
        adjustedScore: score,
        warning: passed
          ? null
          : 'Minor physical inconsistencies detected - review recommended',
      };
    }
    return {
      passed: score >= 40,
      adjustedScore: score,
      warning: null,
    };
  }

  // =========================================================================
  // Helpers
  // =========================================================================

  /** Extract a grid region from a flat array (row i, col j of rows×cols grid). */
  private extractRegion(
    arr: Float64Array,
    w: number,
    h: number,
    row: number,
    col: number,
    rows: number,
    cols: number
  ): Float64Array {
    const y0 = Math.floor((row * h) / rows);
    const y1 = Math.floor(((row + 1) * h) / rows);
    const x0 = Math.floor((col * w) / cols);
    const x1 = Math.floor(((col + 1) * w) / cols);
    const rw = x1 - x0;
    const rh = y1 - y0;
    const region = new Float64Array(rw * rh);
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) {
        region[(y - y0) * rw + (x - x0)] = arr[y * w + x];
      }
    }
    return region;
  }

  /**
   * Apply a metric function to each cell of a rows×cols grid.
   */
  private computeGridMetric(
    arr: Float64Array,
    w: number,
    h: number,
    rows: number,
    cols: number,
    fn: (region: Float64Array, rw: number, rh: number) => number
  ): number[] {
    const values: number[] = [];
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const region = this.extractRegion(arr, w, h, i, j, rows, cols);
        const rw = Math.floor(w / cols);
        const rh = Math.floor(h / rows);
        values.push(fn(region, rw, rh));
      }
    }
    return values;
  }

  /** Count connected components of edge pixels (simplified line count). */
  private countEdgeComponents(
    edges: Uint8Array,
    w: number,
    h: number
  ): number {
    const visited = new Uint8Array(w * h);
    let components = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const idx = y * w + x;
        if (edges[idx] && !visited[idx]) {
          components++;
          // BFS flood fill
          const stack = [idx];
          visited[idx] = 1;
          while (stack.length > 0) {
            const cur = stack.pop()!;
            const cx = cur % w;
            const cy = (cur - cx) / w;
            for (let dy = -1; dy <= 1; dy++) {
              for (let dx = -1; dx <= 1; dx++) {
                if (dy === 0 && dx === 0) continue;
                const nx = cx + dx;
                const ny = cy + dy;
                if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                  const ni = ny * w + nx;
                  if (edges[ni] && !visited[ni]) {
                    visited[ni] = 1;
                    stack.push(ni);
                  }
                }
              }
            }
          }
        }
      }
    }
    return components;
  }

  /** Calculate corner density score from edge/Laplacian data. */
  private calculatePointDensity(
    edges: Uint8Array,
    lap: Float64Array,
    threshold: number,
    w: number,
    h: number
  ): number {
    const gridSize = 10;
    const gridH = Math.max(1, Math.floor(h / gridSize));
    const gridW = Math.max(1, Math.floor(w / gridSize));
    const counts = new Float64Array(gridH * gridW);

    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const idx = y * w + x;
        if (edges[idx] && Math.abs(lap[idx]) >= threshold) {
          const gi = Math.min(Math.floor(y / gridSize), gridH - 1);
          const gj = Math.min(Math.floor(x / gridSize), gridW - 1);
          counts[gi * gridW + gj]++;
        }
      }
    }

    const nonEmpty: number[] = [];
    for (let i = 0; i < counts.length; i++) {
      if (counts[i] > 0) nonEmpty.push(counts[i]);
    }
    if (nonEmpty.length > 0) {
      const v = variance(nonEmpty);
      return Math.max(0, 100 - v * 5);
    }
    return 0;
  }

  /** Pearson correlation coefficient between two arrays. */
  private pearsonCorr(a: Float64Array, b: Float64Array): number {
    const n = a.length;
    if (n === 0) return 0;
    const ma = mean(a);
    const mb = mean(b);
    let sumAB = 0, sumAA = 0, sumBB = 0;
    for (let i = 0; i < n; i++) {
      const da = a[i] - ma;
      const db = b[i] - mb;
      sumAB += da * db;
      sumAA += da * da;
      sumBB += db * db;
    }
    const denom = Math.sqrt(sumAA * sumBB);
    return denom > 1e-10 ? sumAB / denom : 0;
  }

  /** JPEG round-trip: encode then decode, return decoded RGBA. */
  private jpegRoundTrip(
    rgba: Uint8Array,
    w: number,
    h: number,
    quality: number
  ): Uint8Array | null {
    try {
      const raw = { data: rgba, width: w, height: h };
      const encoded = jpeg.encode(raw as jpeg.RawImageData<Uint8Array>, quality);
      const decoded = jpeg.decode(encoded.data, { useTArray: true });
      return new Uint8Array(decoded.data);
    } catch {
      return null;
    }
  }

}
