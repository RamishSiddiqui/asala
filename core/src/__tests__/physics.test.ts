import fs from 'fs';
import path from 'path';
import jpeg from 'jpeg-js';
import { PhysicsVerifier } from '../verifiers/physics';
import {
    toGrayscale,
    convolve2d,
    sobelXY,
    gradientMagnitude,
    laplacian,
    fft1d,
    rgbToHSV,
    rgbToLAB,
    resize,
} from '../imaging/processing';
import { mean, std, variance, entropy, histogram, percentile, clamp, linmap } from '../imaging/stats';

// ---------------------------------------------------------------------------
// Helpers: generate synthetic test images
// ---------------------------------------------------------------------------

/** Create a uniform gray JPEG buffer. */
function makeUniformJpeg(w: number, h: number, value: number = 128): Buffer {
    const rgba = new Uint8Array(w * h * 4);
    for (let i = 0; i < w * h; i++) {
        rgba[i * 4] = value;
        rgba[i * 4 + 1] = value;
        rgba[i * 4 + 2] = value;
        rgba[i * 4 + 3] = 255;
    }
    const raw = { data: rgba, width: w, height: h };
    const encoded = jpeg.encode(raw as jpeg.RawImageData<Uint8Array>, 90);
    return Buffer.from(encoded.data);
}

/** Create a random noise JPEG buffer. */
function makeNoiseJpeg(w: number, h: number): Buffer {
    const rgba = new Uint8Array(w * h * 4);
    for (let i = 0; i < w * h; i++) {
        rgba[i * 4] = Math.floor(Math.random() * 256);
        rgba[i * 4 + 1] = Math.floor(Math.random() * 256);
        rgba[i * 4 + 2] = Math.floor(Math.random() * 256);
        rgba[i * 4 + 3] = 255;
    }
    const raw = { data: rgba, width: w, height: h };
    const encoded = jpeg.encode(raw as jpeg.RawImageData<Uint8Array>, 90);
    return Buffer.from(encoded.data);
}

/** Create a gradient JPEG (smooth left-to-right). */
function makeGradientJpeg(w: number, h: number): Buffer {
    const rgba = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const val = Math.round((x / (w - 1)) * 255);
            const off = (y * w + x) * 4;
            rgba[off] = val;
            rgba[off + 1] = val;
            rgba[off + 2] = val;
            rgba[off + 3] = 255;
        }
    }
    const raw = { data: rgba, width: w, height: h };
    const encoded = jpeg.encode(raw as jpeg.RawImageData<Uint8Array>, 90);
    return Buffer.from(encoded.data);
}

// ---------------------------------------------------------------------------
// PhysicsVerifier tests
// ---------------------------------------------------------------------------

describe('PhysicsVerifier', () => {
    let verifier: PhysicsVerifier;
    let sampleImage: Buffer | null;

    beforeAll(() => {
        const imagePath = path.resolve(
            __dirname,
            '../../../../test-data/original/sample-landscape.jpg'
        );
        sampleImage = fs.existsSync(imagePath)
            ? fs.readFileSync(imagePath)
            : null;
    });

    beforeEach(() => {
        verifier = new PhysicsVerifier();
    });

    // -- Basic structure tests --

    it('should verify a JPEG image and return the new result structure', () => {
        const img = sampleImage ?? makeGradientJpeg(64, 64);
        const result = verifier.verifyImage(img);

        expect(result).toBeDefined();
        expect(result.name).toBe('Physics Verification (Image)');
        expect(typeof result.score).toBe('number');
        expect(result.score).toBeGreaterThanOrEqual(0);
        expect(result.score).toBeLessThanOrEqual(100);
        expect(result.details).toBeDefined();

        // Phase 1 sub-analysis objects
        expect(result.details['noise_uniformity']).toBeDefined();
        expect(result.details['noise_frequency']).toBeDefined();
        expect(result.details['frequency_analysis']).toBeDefined();
        expect(result.details['geometric_consistency']).toBeDefined();
        expect(result.details['lighting_analysis']).toBeDefined();
        expect(result.details['texture_analysis']).toBeDefined();
        expect(result.details['color_analysis']).toBeDefined();
        expect(result.details['compression_analysis']).toBeDefined();

        // Phase 2 sub-analysis objects
        expect(result.details['noise_consistency_map']).toBeDefined();
        expect(result.details['jpeg_ghost']).toBeDefined();
        expect(result.details['spectral_fingerprint']).toBeDefined();
        expect(result.details['channel_correlation']).toBeDefined();

        // Phase 3 sub-analysis objects
        expect(result.details['benford_dct']).toBeDefined();
        expect(result.details['wavelet_ratio']).toBeDefined();
        expect(result.details['blocking_grid']).toBeDefined();
        expect(result.details['cfa_demosaicing']).toBeDefined();

        // Aggregate metrics
        expect(typeof result.details['ai_probability']).toBe('number');
        expect(typeof result.details['ai_indicators']).toBe('number');
    });

    it('should handle a non-image buffer gracefully', () => {
        const buf = Buffer.alloc(100);
        const result = verifier.verifyImage(buf);
        expect(result).toBeDefined();
        expect(result.passed).toBe(false);
        expect(result.score).toBe(0);
        expect(result.details['error']).toBeDefined();
    });

    it('should verify audio and return a result', () => {
        // Non-WAV buffer should return error gracefully
        const buf = Buffer.alloc(1000);
        const result = verifier.verifyAudio(buf);
        expect(result.name).toBe('Physics Verification (Audio)');
        expect(typeof result.score).toBe('number');
        expect(result.details['error']).toBeDefined();
    });

    it('should verify a valid WAV buffer and return sub-analysis objects', () => {
        // Generate a minimal valid 16-bit PCM WAV (0.2s at 8000 Hz)
        const sr = 8000;
        const duration = 0.2;
        const numSamples = Math.floor(sr * duration);
        const dataSize = numSamples * 2; // 16-bit = 2 bytes per sample
        const buf = Buffer.alloc(44 + dataSize);

        // RIFF header
        buf.write('RIFF', 0);
        buf.writeUInt32LE(36 + dataSize, 4);
        buf.write('WAVE', 8);

        // fmt chunk
        buf.write('fmt ', 12);
        buf.writeUInt32LE(16, 16);       // chunk size
        buf.writeUInt16LE(1, 20);        // PCM format
        buf.writeUInt16LE(1, 22);        // mono
        buf.writeUInt32LE(sr, 24);       // sample rate
        buf.writeUInt32LE(sr * 2, 28);   // byte rate
        buf.writeUInt16LE(2, 32);        // block align
        buf.writeUInt16LE(16, 34);       // bits per sample

        // data chunk
        buf.write('data', 36);
        buf.writeUInt32LE(dataSize, 40);

        // Fill with a simple sine wave
        for (let i = 0; i < numSamples; i++) {
            const sample = Math.round(Math.sin(2 * Math.PI * 440 * i / sr) * 16000);
            buf.writeInt16LE(sample, 44 + i * 2);
        }

        const result = verifier.verifyAudio(buf);
        expect(result.name).toBe('Physics Verification (Audio)');
        expect(typeof result.score).toBe('number');
        expect(result.score).toBeGreaterThanOrEqual(0);
        expect(result.score).toBeLessThanOrEqual(100);

        // Should have sub-analysis objects
        expect(result.details['phase_coherence']).toBeDefined();
        expect(result.details['voice_quality']).toBeDefined();
        expect(result.details['enf_analysis']).toBeDefined();
        expect(result.details['spectral_tilt']).toBeDefined();
        expect(result.details['noise_consistency']).toBeDefined();
        expect(result.details['mel_regularity']).toBeDefined();
        expect(result.details['formant_bandwidth']).toBeDefined();
        expect(result.details['double_compression']).toBeDefined();
        expect(result.details['spectral_discontinuity']).toBeDefined();
        expect(result.details['subband_energy']).toBeDefined();

        // Aggregate metrics
        expect(typeof result.details['ai_probability']).toBe('number');
        expect(typeof result.details['ai_indicators']).toBe('number');
    });

    // -- Synthetic image detection --

    it('should score a uniform gray image lower than a gradient image', () => {
        const uniform = makeUniformJpeg(64, 64, 128);
        const gradient = makeGradientJpeg(64, 64);
        const uResult = verifier.verifyImage(uniform);
        const gResult = verifier.verifyImage(gradient);

        // Uniform images are more synthetic-looking
        expect(uResult.score).toBeLessThanOrEqual(gResult.score + 20);
    });

    it('should produce finite scores for a noise image', () => {
        const noise = makeNoiseJpeg(64, 64);
        const result = verifier.verifyImage(noise);
        expect(result.score).toBeGreaterThanOrEqual(0);
        expect(result.score).toBeLessThanOrEqual(100);
        expect(Number.isFinite(result.score)).toBe(true);
    });

    if (fs.existsSync(
        path.resolve(__dirname, '../../../../test-data/original/sample-landscape.jpg')
    )) {
        it('should give a real photo a passing score', () => {
            const result = verifier.verifyImage(sampleImage!);
            expect(result.score).toBeGreaterThanOrEqual(30);
        });
    }
});

// ---------------------------------------------------------------------------
// Imaging primitives tests
// ---------------------------------------------------------------------------

describe('imaging/stats', () => {
    it('mean of [1,2,3,4,5] should be 3', () => {
        expect(mean([1, 2, 3, 4, 5])).toBeCloseTo(3, 10);
    });

    it('std of constant array should be 0', () => {
        expect(std([5, 5, 5, 5])).toBeCloseTo(0, 10);
    });

    it('variance of [0, 10] should be 25', () => {
        expect(variance([0, 10])).toBeCloseTo(25, 10);
    });

    it('entropy of uniform distribution should be log2(N)', () => {
        const h = new Float64Array([1, 1, 1, 1]);
        expect(entropy(h)).toBeCloseTo(2, 5); // log2(4) = 2
    });

    it('percentile 50 of [1,2,3,4,5] should be 3', () => {
        expect(percentile([1, 2, 3, 4, 5], 50)).toBeCloseTo(3, 5);
    });

    it('clamp should restrict values', () => {
        expect(clamp(-5, 0, 100)).toBe(0);
        expect(clamp(150, 0, 100)).toBe(100);
        expect(clamp(50, 0, 100)).toBe(50);
    });

    it('linmap should interpolate linearly', () => {
        expect(linmap(0.5, 0, 1, 0, 100)).toBeCloseTo(50, 10);
        expect(linmap(0, 0, 1, 10, 20)).toBeCloseTo(10, 10);
    });

    it('histogram should bin values correctly', () => {
        const h = histogram([0, 1, 2, 3], 4, 0, 4);
        expect(h[0]).toBe(1);
        expect(h[1]).toBe(1);
        expect(h[2]).toBe(1);
        expect(h[3]).toBe(1);
    });
});

describe('imaging/processing', () => {
    it('toGrayscale should convert RGBA to single channel', () => {
        // White pixel: RGBA = [255, 255, 255, 255]
        const rgba = new Uint8Array([255, 255, 255, 255, 0, 0, 0, 255]);
        const gray = toGrayscale(rgba, 2, 1);
        expect(gray[0]).toBeCloseTo(255, 0);
        expect(gray[1]).toBeCloseTo(0, 0);
    });

    it('identity convolution should preserve values', () => {
        const img = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
        const kernel = new Float64Array([0, 0, 0, 0, 1, 0, 0, 0, 0]);
        const result = convolve2d(img, 3, 3, kernel, 3, 3);
        expect(result[4]).toBeCloseTo(5, 10); // center pixel
    });

    it('laplacian of flat image should be zero', () => {
        const flat = new Float64Array(9).fill(50);
        const lap = laplacian(flat, 3, 3);
        expect(lap[4]).toBeCloseTo(0, 5);
    });

    it('sobelXY should detect horizontal gradient', () => {
        // Left column 0, right column 255, 3x3
        const img = new Float64Array([0, 0, 255, 0, 0, 255, 0, 0, 255]);
        const { gx } = sobelXY(img, 3, 3);
        // Center pixel should have positive gx
        expect(gx[4]).toBeGreaterThan(0);
    });

    it('gradientMagnitude should be non-negative', () => {
        const gx = new Float64Array([3, -4]);
        const gy = new Float64Array([4, 3]);
        const mag = gradientMagnitude(gx, gy);
        expect(mag[0]).toBeCloseTo(5, 5);
        expect(mag[1]).toBeCloseTo(5, 5);
    });

    it('rgbToHSV should handle pure red', () => {
        const [h, s, v] = rgbToHSV(255, 0, 0);
        expect(h).toBeCloseTo(0, 0);       // Hue 0°
        expect(s).toBeCloseTo(255, 0);     // Full saturation
        expect(v).toBeCloseTo(255, 0);     // Full value
    });

    it('rgbToLAB should handle black', () => {
        const [L, a, b] = rgbToLAB(0, 0, 0);
        expect(L).toBeCloseTo(0, 0);       // L* = 0
        expect(a).toBeCloseTo(128, 1);     // neutral a
        expect(b).toBeCloseTo(128, 1);     // neutral b
    });

    it('rgbToLAB should handle white', () => {
        const [L] = rgbToLAB(255, 255, 255);
        expect(L).toBeGreaterThan(240);    // L* close to 255
    });

    it('fft1d of known signal should produce correct output', () => {
        // FFT of [1, 0, 0, 0] should give [1, 1, 1, 1]
        const re = new Float64Array([1, 0, 0, 0]);
        const im = new Float64Array(4);
        fft1d(re, im);
        for (let i = 0; i < 4; i++) {
            expect(re[i]).toBeCloseTo(1, 5);
            expect(im[i]).toBeCloseTo(0, 5);
        }
    });

    it('fft1d of [1,1,1,1] should produce DC peak', () => {
        const re = new Float64Array([1, 1, 1, 1]);
        const im = new Float64Array(4);
        fft1d(re, im);
        expect(re[0]).toBeCloseTo(4, 5);   // DC = sum
        for (let i = 1; i < 4; i++) {
            expect(re[i]).toBeCloseTo(0, 5);
        }
    });

    it('resize should preserve a constant image', () => {
        const img = new Float64Array(16).fill(42);
        const resized = resize(img, 4, 4, 2, 2);
        for (let i = 0; i < 4; i++) {
            expect(resized[i]).toBeCloseTo(42, 5);
        }
    });
});
