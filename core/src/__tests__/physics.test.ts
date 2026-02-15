import fs from 'fs';
import path from 'path';
import { PhysicsVerifier } from '../verifiers/physics';

describe('PhysicsVerifier', () => {
    let verifier: PhysicsVerifier;
    let sampleImage: Buffer;

    beforeAll(() => {
        // Read sample image from test-data
        // Relative path from core/src/__tests__/physics.test.ts to test-data/original/sample-landscape.jpg
        // core/src/__tests__ -> core/src -> core -> asala -> test-data
        const imagePath = path.resolve(__dirname, '../../../../test-data/original/sample-landscape.jpg');

        if (fs.existsSync(imagePath)) {
            sampleImage = fs.readFileSync(imagePath);
        } else {
            console.warn('Sample image not found, skipping real image tests');
            // Create a dummy buffer that behaves like a file but won't parse as JPEG
            sampleImage = Buffer.alloc(100);
        }
    });

    beforeEach(() => {
        verifier = new PhysicsVerifier();
    });

    it('should verify image and return a result', () => {
        const result = verifier.verifyImage(sampleImage);

        expect(result).toBeDefined();
        expect(result.name).toBe('Physics Verification (Image)');
        expect(typeof result.score).toBe('number');
        expect(result.details).toBeDefined();
        expect(result.details.noiseScore).toBeDefined();
        expect(result.details.lightingScore).toBeDefined();
        expect(result.details.aberrationScore).toBeDefined();
    });

    it('should verify random buffer gracefully', () => {
        const randomBuffer = Buffer.alloc(1000); // Not a JPEG
        const result = verifier.verifyImage(randomBuffer);

        expect(result).toBeDefined();
        // ELA should catch it's not a JPEG and return 0 (or handle it)
        // Verify it doesn't crash
        expect(result.passed).toBeDefined();
    });
});
