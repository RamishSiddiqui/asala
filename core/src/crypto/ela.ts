import jpeg from 'jpeg-js';

export interface ELADetailedResult {
  /** Per-quality mean ELA scores (quality levels 95, 85, 75, 65). */
  elaScores: number[];
  elaVariance: number;
  elaMean: number;
  elaRange: number;
  elaGradient: number;
  /** Coefficient of variation of per-region ELA means (8x8 grid at quality 90). */
  regionalElaCv: number;
  /** Number of regions with ELA mean >2x or <0.5x the global mean. */
  suspiciousRegions: number;
  /** Composite compression score [0-100]. */
  compressionScore: number;
}

/**
 * Error Level Analysis (ELA) Utilities
 * detecting manipulation by analyzing compression artifacts
 */
export class ErrorLevelAnalysis {
    /**
     * Calculate ELA score for an image buffer (backward-compatible).
     * @param imageBuffer Original image buffer
     * @param quality Quality level for re-compression (default 95)
     * @returns Score between 0-100 (higher means more consistent / less manipulation)
     */
    static analyze(imageBuffer: Buffer, quality: number = 95): number {
        try {
            const original = jpeg.decode(imageBuffer, { useTArray: true });
            const recompressed = jpeg.encode(original, quality);
            const decodedRecompressed = jpeg.decode(recompressed.data, { useTArray: true });
            return this.calculateDifference(original, decodedRecompressed);
        } catch {
            return 0;
        }
    }

    /**
     * Multi-quality ELA with regional analysis.
     * Matches the Python _analyze_compression_artifacts algorithm.
     *
     * @param imageBuffer  Raw image bytes (JPEG or pre-decoded RGBA via decodeImage)
     * @param decodedRGBA  Optional pre-decoded RGBA pixel data and dimensions.
     *                     When provided, imageBuffer is ignored for decoding but
     *                     is still used for initial JPEG decode if decodedRGBA is
     *                     not given.
     */
    static analyzeDetailed(
        imageBuffer: Buffer,
        decodedRGBA?: { data: Uint8Array; width: number; height: number }
    ): ELADetailedResult {
        try {
            // Decode or use provided pixels
            let width: number, height: number, data: Uint8Array;
            if (decodedRGBA) {
                ({ data, width, height } = decodedRGBA);
            } else {
                const dec = jpeg.decode(imageBuffer, { useTArray: true });
                data = new Uint8Array(dec.data);
                width = dec.width;
                height = dec.height;
            }

            // ---- Global multi-quality ELA ----
            const qualities = [95, 85, 75, 65];
            const elaScores: number[] = [];

            for (const q of qualities) {
                const raw = { data, width, height };
                const encoded = jpeg.encode(raw as jpeg.RawImageData<Uint8Array>, q);
                const redecoded = jpeg.decode(encoded.data, { useTArray: true });
                const meanDiff = this.meanPixelDifference(data, new Uint8Array(redecoded.data), width, height);
                elaScores.push(meanDiff);
            }

            const elaVariance = this.arrVariance(elaScores);
            const elaMean = this.arrMean(elaScores);
            const elaRange = Math.max(...elaScores) - Math.min(...elaScores);
            const elaGradient =
                (elaScores[elaScores.length - 1] - elaScores[0]) /
                (qualities[qualities.length - 1] - qualities[0]);

            // ---- Regional ELA at quality=90 ----
            const encoded90 = jpeg.encode(
                { data, width, height } as jpeg.RawImageData<Uint8Array>,
                90
            );
            const redecoded90 = jpeg.decode(encoded90.data, { useTArray: true });
            const diff90 = this.perPixelDifference(data, new Uint8Array(redecoded90.data), width, height);

            const gridN = 8;
            const regionMeans: number[] = [];
            for (let i = 0; i < gridN; i++) {
                for (let j = 0; j < gridN; j++) {
                    const y0 = Math.floor((i * height) / gridN);
                    const y1 = Math.floor(((i + 1) * height) / gridN);
                    const x0 = Math.floor((j * width) / gridN);
                    const x1 = Math.floor(((j + 1) * width) / gridN);
                    let sum = 0, count = 0;
                    for (let y = y0; y < y1; y++) {
                        for (let x = x0; x < x1; x++) {
                            sum += diff90[y * width + x];
                            count++;
                        }
                    }
                    regionMeans.push(count > 0 ? sum / count : 0);
                }
            }

            const globalElaMean = this.arrMean(regionMeans);
            const regionalElaCv =
                this.arrStd(regionMeans) / (globalElaMean + 1e-10);
            let suspiciousHigh = 0, suspiciousLow = 0;
            for (const rm of regionMeans) {
                if (rm > 2 * globalElaMean) suspiciousHigh++;
                if (rm < 0.5 * globalElaMean) suspiciousLow++;
            }
            const suspiciousRegions = suspiciousHigh + suspiciousLow;

            // ---- Scoring ----
            const varianceScore = 100 - Math.min(100, elaVariance * 20);
            const meanScore = 100 - Math.min(100, elaMean * 5);
            const gradientScore = Math.min(100, elaGradient * 50);
            const rangeScore = Math.min(100, elaRange * 30);
            const regionalScore = Math.max(0, Math.min(100, 100 - regionalElaCv * 100));

            const compressionScore = Math.round(
                Math.max(
                    0,
                    Math.min(
                        100,
                        varianceScore * 0.2 +
                        meanScore * 0.2 +
                        gradientScore * 0.2 +
                        rangeScore * 0.2 +
                        regionalScore * 0.2
                    )
                )
            );

            return {
                elaScores,
                elaVariance,
                elaMean,
                elaRange,
                elaGradient,
                regionalElaCv,
                suspiciousRegions,
                compressionScore,
            };
        } catch {
            return {
                elaScores: [],
                elaVariance: 0,
                elaMean: 0,
                elaRange: 0,
                elaGradient: 0,
                regionalElaCv: 0.35,
                suspiciousRegions: 0,
                compressionScore: 50,
            };
        }
    }

    // -------------------------------------------------------------------
    // Private helpers
    // -------------------------------------------------------------------

    private static calculateDifference(
        original: jpeg.UintArrRet,
        recompressed: jpeg.UintArrRet
    ): number {
        if (original.width !== recompressed.width || original.height !== recompressed.height) {
            return 0;
        }

        let totalDiff = 0;
        const size = original.width * original.height * 4;
        const data1 = original.data;
        const data2 = recompressed.data;

        for (let i = 0; i < size; i += 4) {
            totalDiff +=
                Math.abs(data1[i] - data2[i]) +
                Math.abs(data1[i + 1] - data2[i + 1]) +
                Math.abs(data1[i + 2] - data2[i + 2]);
        }

        const avgDiff = totalDiff / (original.width * original.height);
        const score = Math.max(0, 100 - avgDiff * 5);
        return Math.round(score);
    }

    /** Mean of RGB pixel difference (single channel mean across image). */
    private static meanPixelDifference(
        a: Uint8Array,
        b: Uint8Array,
        w: number,
        h: number
    ): number {
        let sum = 0;
        const n = w * h;
        for (let i = 0; i < n; i++) {
            const off = i * 4;
            sum +=
                (Math.abs(a[off] - b[off]) +
                    Math.abs(a[off + 1] - b[off + 1]) +
                    Math.abs(a[off + 2] - b[off + 2])) / 3;
        }
        return sum / n;
    }

    /** Per-pixel grayscale ELA difference (mean of abs R/G/B diff). */
    private static perPixelDifference(
        a: Uint8Array,
        b: Uint8Array,
        w: number,
        h: number
    ): Float64Array {
        const n = w * h;
        const out = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            const off = i * 4;
            out[i] =
                (Math.abs(a[off] - b[off]) +
                    Math.abs(a[off + 1] - b[off + 1]) +
                    Math.abs(a[off + 2] - b[off + 2])) / 3;
        }
        return out;
    }

    private static arrMean(arr: number[]): number {
        if (arr.length === 0) return 0;
        let s = 0;
        for (const v of arr) s += v;
        return s / arr.length;
    }

    private static arrVariance(arr: number[]): number {
        if (arr.length === 0) return 0;
        const m = this.arrMean(arr);
        let s = 0;
        for (const v of arr) { const d = v - m; s += d * d; }
        return s / arr.length;
    }

    private static arrStd(arr: number[]): number {
        return Math.sqrt(this.arrVariance(arr));
    }
}
