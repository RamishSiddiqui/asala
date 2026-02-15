import fs from 'fs';
import path from 'path';
import { Asala } from '../core/src/index';

async function main() {
    console.log('--- Testing Physics Verification (TypeScript) ---');

    // Paths
    const imagePath = path.join(__dirname, '../test-data/original/sample-landscape.jpg');
    const manifestPath = path.join(__dirname, '../test-data/signed/sample-landscape.jpg.manifest.json');

    if (!fs.existsSync(imagePath) || !fs.existsSync(manifestPath)) {
        console.error('Test data not found!');
        console.error('Image:', imagePath);
        console.error('Manifest:', manifestPath);
        process.exit(1);
    }

    // Read files
    const imageBuffer = fs.readFileSync(imagePath);
    const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));

    // Initialize Asala
    const asala = new Asala();

    console.log('Verifying image:', path.basename(imagePath));
    console.log('Image size:', imageBuffer.length, 'bytes');

    try {
        const result = await asala.verify(imageBuffer, manifest);

        console.log('\n--- Verification Result ---');
        console.log('Status:', result.status);
        console.log('Confidence:', result.confidence);

        // Find Physics Layer
        const physicsLayer = result.layers.find(l => l.name.startsWith('Physics Verification'));

        if (physicsLayer) {
            console.log('\n[Physics Verification Layer]');
            console.log('Passed:', physicsLayer.passed);
            console.log('Score:', physicsLayer.score);
            console.log('Details:', JSON.stringify(physicsLayer.details, null, 2));
        } else {
            console.error('\n[ERROR] Physics Verification Layer NOT found!');
        }

    } catch (error) {
        console.error('Verification failed:', error);
    }
}

main();
