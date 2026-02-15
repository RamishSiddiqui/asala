# Examples

This directory contains practical examples of using Asala.

## Basic Examples

### 1. Simple Verification

```javascript
const { Asala } = require('@asala/core');
const fs = require('fs');

const asala = new Asala();
const content = fs.readFileSync('./photo.jpg');

asala.verify(content).then(result => {
  console.log('Status:', result.status);
  console.log('Confidence:', result.confidence + '%');
});
```

### 2. Signing Content

```javascript
const { Asala } = require('@asala/core');
const fs = require('fs');

const asala = new Asala();
const { privateKey, publicKey } = asala.generateKeyPair();

// Save keys
fs.writeFileSync('private.pem', privateKey);
fs.writeFileSync('public.pem', publicKey);

// Sign content
const content = fs.readFileSync('./photo.jpg');
const manifest = asala.signContent(content, privateKey, 'John Doe');

// Save manifest
fs.writeJsonSync('photo.jpg.manifest.json', manifest, { spaces: 2 });
```

## Advanced Examples

### 3. Batch Verification

```javascript
const { Asala } = require('@asala/core');
const fs = require('fs').promises;
const path = require('path');

async function verifyBatch(directory) {
  const asala = new Asala();
  const files = await fs.readdir(directory);
  const imageFiles = files.filter(f => 
    ['.jpg', '.jpeg', '.png'].includes(path.extname(f).toLowerCase())
  );

  const results = await Promise.all(
    imageFiles.map(async (file) => {
      const content = await fs.readFile(path.join(directory, file));
      const result = await asala.verify(content);
      return { file, ...result };
    })
  );

  // Summary
  const verified = results.filter(r => r.status === 'verified').length;
  const unverified = results.filter(r => r.status === 'unverified').length;
  
  console.log(`\nBatch Verification Complete:`);
  console.log(`  Verified: ${verified}/${results.length}`);
  console.log(`  Unverified: ${unverified}/${results.length}`);
  
  return results;
}

verifyBatch('./my-photos');
```

### 4. Web Integration

```html
<!DOCTYPE html>
<html>
<head>
  <title>Asala Demo</title>
  <script src="https://unpkg.com/@asala/core@latest/dist/index.js"></script>
</head>
<body>
  <input type="file" id="fileInput" accept="image/*">
  <div id="result"></div>

  <script>
    const asala = new Asala();
    
    document.getElementById('fileInput').addEventListener('change', async (e) => {
      const file = e.target.files[0];
      const arrayBuffer = await file.arrayBuffer();
      const buffer = Buffer.from(arrayBuffer);
      
      const result = await asala.verify(buffer);
      
      const resultDiv = document.getElementById('result');
      resultDiv.innerHTML = `
        <h3>Verification Result</h3>
        <p>Status: ${result.status}</p>
        <p>Confidence: ${result.confidence}%</p>
      `;
    });
  </script>
</body>
</html>
```

### 5. Express.js Middleware

```javascript
const express = require('express');
const { Asala } = require('@asala/core');
const multer = require('multer');

const app = express();
const upload = multer();
const asala = new Asala();

// Verification endpoint
app.post('/verify', upload.single('file'), async (req, res) => {
  try {
    const result = await asala.verify(req.file.buffer);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(3000, () => {
  console.log('Verification API running on port 3000');
});
```

## Real-World Use Cases

### News Organization

```javascript
// Sign all published photos
const newsOrg = {
  name: 'Global News Network',
  keys: loadKeysFromHSM(), // Hardware Security Module
  
  async publishPhoto(photoPath, metadata) {
    const content = fs.readFileSync(photoPath);
    
    const manifest = new ManifestBuilder(
      hashContent(content),
      'image',
      this.name
    )
    .addCreationInfo(
      metadata.camera,
      metadata.software,
      metadata.location
    )
    .addAssertion('c2pa.metadata', {
      photojournalist: metadata.author,
      caption: metadata.caption,
      dateTaken: metadata.date
    })
    .sign(this.keys.private, this.name)
    .build();
    
    // Embed manifest in image
    return embedManifest(photoPath, manifest);
  }
};
```

### Social Media Platform

```javascript
// Verify all uploaded content
const verificationMiddleware = async (req, res, next) => {
  const asala = new Asala();
  const result = await asala.verify(req.file.buffer);
  
  // Attach verification result to request
  req.verification = result;
  
  // Flag potentially fake content
  if (result.status === 'tampered') {
    req.flags = req.flags || [];
    req.flags.push('potential_manipulation');
  }
  
  next();
};

app.post('/upload', upload.single('media'), verificationMiddleware, (req, res) => {
  res.json({
    uploaded: true,
    verification: req.verification,
    flags: req.flags || []
  });
});
```

## Running the Examples

```bash
# Install dependencies
npm install @asala/core

# Run an example
node examples/01-simple-verification.js

# Or use the CLI
asala verify ./examples/sample-image.jpg
```
