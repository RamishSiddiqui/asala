# Asala

**Cryptographic Content Authenticity Verification** - An open-source, multi-layered solution to sign and verify the authenticity of digital content (images, videos, audio, documents) without relying on AI detection.

> **Asala** (أصالة) - Arabic for "Authenticity"

## 🎯 Core Philosophy

Instead of trying to detect fakes (which models will always improve at), we **mathematically prove authenticity** using:
- **Cryptographic provenance** (C2PA standard)
- **Immutable chain of trust**
- **Physics-based verification**
- **Distributed consensus**

## 📦 Packages

| Package | Language | Description |
|---------|----------|-------------|
| `@asala/core` | TypeScript | Cryptographic verification engine + physics-based analysis |
| `@asala/cli` | TypeScript | Command-line tool for verification and signing |
| `@asala/extension` | TypeScript | Browser extension for real-time verification |
| `@asala/web` | TypeScript | Next.js web interface for manual verification |
| `asala` | Python | Full Python implementation (PyPI) with image, audio, and video verification |

## 🚀 Quick Start

```bash
# Install CLI (Node.js)
npm install -g @asala/cli

# Install CLI (Python)
pip install asala

# Verify content
asala verify ./image.jpg

# Verify with physics-based analysis
asala verify ./image.jpg --physics

# Verify with parallel processing (4 threads)
asala verify ./image.jpg --physics --workers 4

# Sign content (content creators)
asala sign ./video.mp4 --key ./private.pem
```

## 🛡️ How It Works

### Layer 1: Cryptographic Provenance (Unbreakable)
- Content signed at creation using hardware-backed private keys
- Chain of custody for any edits/transformations
- Mathematical proof of authenticity via digital signatures

### Layer 2: Physics-Based Verification (Implemented)
- **Image analysis** (16 methods): noise uniformity, frequency/DCT analysis, ELA, geometric consistency, lighting, texture, color distribution, spectral fingerprinting, channel correlation, Benford's law, wavelet ratios, CFA demosaicing detection
- **Audio analysis** (10 methods): phase coherence, voice quality, ENF analysis, spectral tilt, noise consistency, mel regularity, formant bandwidth, double compression, spectral discontinuity, sub-band energy
- **Video analysis** (6 methods): per-frame image analysis, temporal noise, optical flow, encoding analysis, temporal lighting, frame stability
- **Parallel processing**: All analysis methods within each verifier can run concurrently via `ThreadPoolExecutor` (`max_workers` parameter, off by default)

### Layer 3: Distributed Consensus (Roadmap)
- Multi-party verification network (Planned v0.2.0)
- Reputation-weighted validation (Planned)
- Time-locked existence proofs (Planned)

## 🤝 Contributing

We welcome contributors! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## 📄 License

GPLv3 License - See [LICENSE](./LICENSE) for details.

## 🔗 Resources

- [C2PA Specification](https://c2pa.org/specifications/specifications/1.3/specs/C2PA_Specification.html)
- [Documentation](./docs/README.md)
- [Examples](./examples/)

---

**Built for the community. Verifiable by math. Trust through authenticity.**
