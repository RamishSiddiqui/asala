import jpeg from 'jpeg-js';

/**
 * Error Level Analysis (ELA) Utilities
 * detecting manipulation by analyzing compression artifacts
 */
export class ErrorLevelAnalysis {
    /**
     * Calculate ELA score for an image buffer
     * @param imageBuffer Original image buffer
     * @param quality Quality level for re-compression (default 95)
     * @returns Score between 0-100 (higher means more manipulation detected)
     */
    static analyze(imageBuffer: Buffer, quality: number = 95): number {
        try {
            // 1. Decode original image
            const original = jpeg.decode(imageBuffer, { useTArray: true });

            // 2. Re-compress image at known quality
            const recompressed = jpeg.encode(original, quality);

            // 3. Decode re-compressed image
            const decodedRecompressed = jpeg.decode(recompressed.data, { useTArray: true });

            // 4. Calculate difference
            return this.calculateDifference(original, decodedRecompressed);
        } catch (error) {
            console.error('ELA Analysis failed:', error);
            return 0; // Fallback for non-JPEG or errors
        }
    }

    /**
     * Calculate pixel-by-pixel difference
     */
    private static calculateDifference(original: jpeg.UintArrRet, recompressed: jpeg.UintArrRet): number {
        if (original.width !== recompressed.width || original.height !== recompressed.height) {
            return 0;
        }

        let totalDiff = 0;
        const size = original.width * original.height * 4; // RGBA
        const data1 = original.data;
        const data2 = recompressed.data;

        // Skip Alpha channel (every 4th byte)
        for (let i = 0; i < size; i += 4) {
            const rDiff = Math.abs(data1[i] - data2[i]);
            const gDiff = Math.abs(data1[i + 1] - data2[i + 1]);
            const bDiff = Math.abs(data1[i + 2] - data2[i + 2]);

            // Weight difference by intensity
            totalDiff += (rDiff + gDiff + bDiff);
        }

        // Normalize score
        // Max possible diff per pixel is 255*3 = 765
        // Average difference per pixel
        const avgDiff = totalDiff / (original.width * original.height);

        // Scale to 0-100 range
        // A typical "untouched" JPEG re-compressed at 95% might have avg diff of 2-5
        // A manipulated one might be higher. 
        // This is a heuristic mapping.
        const score = Math.max(0, 100 - (avgDiff * 5));

        return Math.round(score);
    }
}
