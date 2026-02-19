# Asala - Documentation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [How It Works](#how-it-works)
4. [Getting Started](#getting-started)
5. [API Reference](#api-reference)
6. [Examples](#examples)
7. [Contributing](#contributing)

## Overview

Asala is an open-source solution for verifying digital content authenticity using cryptographic provenance. Unlike AI-based detection systems that try to identify fakes, we use mathematical proofs that are inherently unbreakable.

### Why Cryptographic Provenance?

- **AI models can always improve** at generating realistic fakes
- **Mathematics cannot be fooled** - cryptographic proofs are based on hard math problems
- **No training required** - verification is instant and deterministic
- **Privacy-preserving** - verification can happen locally on your device

## Architecture

Asala uses a multi-layered approach:

### Layer 1: Cryptographic Provenance (Primary)
- Content signed at creation with private keys
- Immutable chain of custody for any edits
- Mathematical signature verification

### Layer 2: Physics-Based Verification (Secondary)
- **Image analysis** (16 methods): noise, frequency, ELA, geometric, lighting, texture, color, spectral fingerprint, channel correlation, Benford DCT, wavelet, CFA demosaicing
- **Audio analysis** (10 methods): phase coherence, voice quality, ENF, spectral tilt, noise consistency, mel regularity, formant bandwidth, double compression, splice detection, sub-band energy
- **Video analysis** (6 methods): per-frame analysis, temporal noise, optical flow, encoding, temporal lighting, frame stability
- **Parallel processing**: All methods run concurrently via `max_workers` parameter (Python `ThreadPoolExecutor`, off by default)

### Layer 3: Distributed Consensus (Tertiary)
- Multi-party verification network
- Reputation-weighted validation
- Time-locked existence proofs

## How It Works

### For Content Creators

1. **Create content** using C2PA-enabled tools or devices
2. **Sign content** with your private key at creation time
3. **Publish** - the signature travels with the content
4. **Edit history** - any changes are also signed

### For Content Consumers

1. **View content** anywhere on the web
2. **Verify automatically** - browser extension checks in real-time
3. **See results** - green badge = verified, yellow = unknown, red = tampered
4. **View details** - click for full provenance chain

### Verification Process

```
Content + Signature → Hash Verification → Chain Integrity → Trust Check → Result
```

## Getting Started

### Installation

```bash
# Install CLI globally
npm install -g @asala/cli

# Or use npx
npx @asala/cli verify ./image.jpg
```

### Quick Start

**Verify content:**
```bash
asala verify ./photo.jpg
```

**Sign your content:**
```bash
# Generate keys first
asala keys --generate

# Sign content
asala sign ./photo.jpg --key ./keys/private.pem --creator "Your Name"
```

**Install browser extension:**
- Download from Chrome Web Store or Firefox Add-ons
- Automatically verifies content on social media

## API Reference

### Core Library (`@asala/core`)

```typescript
import { Asala } from '@asala/core';

const asala = new Asala();

// Verify content
const result = await asala.verify(contentBuffer, manifest);

// Sign content
const manifest = asala.signContent(content, privateKey, creator);

// Generate keys
const { publicKey, privateKey } = asala.generateKeyPair();
```

### CLI Commands

```bash
# Verify
asala verify <file> [options]
  -m, --manifest <path>    Path to manifest file
  -t, --trust <keys...>    Trusted public keys
  -p, --physics           Enable physics-based verification
  -w, --workers <n>       Number of parallel threads for analysis (default: 1)
  -j, --json              Output as JSON
  -v, --verbose           Verbose output

# Sign
asala sign <file> [options]
  -k, --key <path>        Path to private key (required)
  -o, --output <path>     Output file path
  -c, --creator <name>    Creator name
  -d, --device <device>   Device name

# Keys
asala keys [options]
  -g, --generate          Generate new key pair
  -o, --output <dir>      Output directory

# Manifest
asala manifest <file> [options]
  -e, --extract           Extract manifest to file
  -o, --output <path>     Output path
```

## Examples

### Example 1: Verify an Image

```bash
asala verify ./vacation-photo.jpg
```

Output:
```
============================================================
  Content Verification Report
============================================================

File: /path/to/vacation-photo.jpg
Status: VERIFIED
Confidence: 95%

✓ Signature Verification: 100%
✓ Chain Integrity: 100%
✓ Trust Verification: 100%

Provenance Data:
  ID: urn:uuid:abc123...
  Creator: John Doe
  Created: 12/15/2023, 2:30:45 PM
  Signatures: 1

============================================================
```

### Example 2: Sign Content

```bash
# Generate keys
asala keys --generate --output ./my-keys

# Sign a photo
asala sign ./my-photo.jpg \
  --key ./my-keys/private.pem \
  --creator "John Doe" \
  --device "iPhone 14 Pro"
```

### Example 3: Programmatic Usage (TypeScript)

```typescript
import { Asala } from '@asala/core';
import fs from 'fs';

async function verifyImage(imagePath: string) {
  const asala = new Asala();

  // Read image
  const content = fs.readFileSync(imagePath);

  // Verify
  const result = await asala.verify(content);

  if (result.status === 'verified') {
    console.log('Content is authentic!');
    console.log(`Confidence: ${result.confidence}%`);
  } else {
    console.log('Content could not be verified');
    console.log('Warnings:', result.warnings);
  }
}
```

### Example 4: Parallel Verification (Python)

```python
from asala import Asala

# Enable parallel analysis with 4 threads
asala = Asala(max_workers=4)

with open('photo.jpg', 'rb') as f:
    content = f.read()

result = asala.verify(content)
print(f"Status: {result.status.value}")
print(f"Confidence: {result.confidence}%")
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/asala.git
cd asala

# Install Node.js dependencies
npm install

# Install Python dependencies
pip install -e ".[dev]"

# Build all TypeScript packages
npm run build

# Run TypeScript tests
npm run test

# Run Python tests
pytest python/tests -v
```

### Project Structure

```
asala/
├── core/          # TypeScript core library (crypto + physics + audio + video)
│   └── src/
│       ├── crypto/       # Hashing, signing, ELA
│       ├── imaging/      # Pure-JS image processing (FFT, DCT, convolution)
│       ├── types/        # TypeScript interfaces
│       └── verifiers/    # Physics, audio, video verifiers
├── python/        # Python implementation
│   ├── asala/        # Package (verify, crypto, physics, audio, video)
│   └── tests/        # pytest suite (108 tests)
├── cli/           # Node.js CLI tool
├── extension/     # Browser extension (Chrome/Firefox)
├── web/           # Next.js web interface
├── docs/          # Sphinx documentation
└── examples/      # Example usage
```

## Resources

- [C2PA Specification](https://c2pa.org/specifications/)
- [Content Authenticity Initiative](https://contentauthenticity.org/)
- [Project GitHub](https://github.com/your-org/asala)

## License

MIT License - See [LICENSE](../LICENSE) for details.
