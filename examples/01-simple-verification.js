const { Asala } = require('@asala/core');
const fs = require('fs');
const path = require('path');

// Initialize verifier
const asala = new Asala();

// Example: Verify an image
async function verifyImage(imagePath) {
  console.log(`Verifying: ${imagePath}`);
  
  try {
    const content = fs.readFileSync(imagePath);
    const result = await asala.verify(content);
    
    console.log('\nVerification Result:');
    console.log('===================');
    console.log(`Status: ${result.status}`);
    console.log(`Confidence: ${result.confidence}%`);
    
    if (result.warnings.length > 0) {
      console.log('\nWarnings:');
      result.warnings.forEach(w => console.log(`  - ${w}`));
    }
    
    if (result.errors.length > 0) {
      console.log('\nErrors:');
      result.errors.forEach(e => console.log(`  - ${e}`));
    }
    
    if (result.manifest) {
      console.log('\nProvenance Data:');
      console.log(`  Creator: ${result.manifest.createdBy}`);
      console.log(`  Created: ${result.manifest.createdAt}`);
      console.log(`  Signatures: ${result.manifest.signatures.length}`);
    }
    
  } catch (error) {
    console.error('Verification failed:', error.message);
  }
}

// Run example
const sampleImage = process.argv[2] || './sample-image.jpg';
verifyImage(sampleImage);
