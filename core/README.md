# @asala/core

Core cryptographic provenance library for content authenticity verification.

## Installation

```bash
npm install @asala/core
```

## Usage

```typescript
import { Asala } from '@asala/core';

const asala = new Asala();

// Sign content
const manifest = asala.signContent(
  contentBuffer,
  privateKey,
  'creator-name'
);

// Verify content
const result = await asala.verify(contentBuffer, manifest);

if (result.status === 'verified') {
  console.log('Content is authentic!');
}
```

## Features

- Cryptographic signing and verification (RSA-2048, SHA-256)
- Physics-based image analysis (16 methods)
- Audio verification (10 methods)
- Video verification (6 methods)
- Content type auto-detection
- C2PA-compatible manifest structure

## Documentation

See the [main README](https://github.com/RamishSiddiqui/asala#readme) for full documentation.

## License

GPL-3.0
