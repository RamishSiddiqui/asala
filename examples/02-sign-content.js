const { Asala } = require('@asala/core');
const fs = require('fs');
const path = require('path');

// Initialize
const asala = new Asala();

// Example: Sign content
async function signContent(imagePath, creatorName) {
  console.log(`Signing: ${imagePath}`);
  console.log(`Creator: ${creatorName}\n`);
  
  try {
    // Generate keys (in production, use existing keys)
    console.log('Generating key pair...');
    const { publicKey, privateKey } = asala.generateKeyPair();
    
    // Save keys for reference
    const keysDir = path.join(__dirname, 'keys');
    if (!fs.existsSync(keysDir)) {
      fs.mkdirSync(keysDir, { recursive: true });
    }
    
    fs.writeFileSync(path.join(keysDir, 'example-private.pem'), privateKey);
    fs.writeFileSync(path.join(keysDir, 'example-public.pem'), publicKey);
    console.log('Keys saved to ./keys/\n');
    
    // Read content
    const content = fs.readFileSync(imagePath);
    
    // Sign content
    console.log('Creating signature...');
    const manifest = asala.signContent(content, privateKey, creatorName);
    
    // Save manifest
    const manifestPath = `${imagePath}.manifest.json`;
    fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
    
    console.log('\n✓ Content signed successfully!');
    console.log('==========================');
    console.log(`Manifest saved to: ${manifestPath}`);
    console.log(`Content hash: ${manifest.contentHash}`);
    console.log(`Signatures: ${manifest.signatures.length}`);
    console.log(`\nYou can now verify this content with:`);
    console.log(`  asala verify ${imagePath} --manifest ${manifestPath}`);
    
  } catch (error) {
    console.error('Signing failed:', error.message);
  }
}

// Run example
const imagePath = process.argv[2];
const creatorName = process.argv[3] || 'Example Creator';

if (!imagePath) {
  console.log('Usage: node 02-sign-content.js <image-path> [creator-name]');
  console.log('Example: node 02-sign-content.js ./photo.jpg "John Doe"');
  process.exit(1);
}

signContent(imagePath, creatorName);
