# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1]

### Added

#### Parallel Processing for Verification Pipeline
- `max_workers` parameter on `Asala`, `PhysicsVerifier`, `AudioVerifier`, and `VideoVerifier` classes
- `ThreadPoolExecutor`-based parallelism for all independent analysis methods within each verifier
- Image: 16 methods run concurrently; Audio: 10 methods; Video: 6 top-level methods + per-frame image analysis
- Off by default (`max_workers=1`); set `max_workers > 1` to enable parallel execution
- CLI `--workers` / `-w` flag on the `verify` command to control thread count
- Thread-safe by design: all analysis methods read from pre-computed shared arrays, no shared mutable state
- Uses `ThreadPoolExecutor` (not `ProcessPoolExecutor`) because numpy/scipy/OpenCV release the GIL during C-level operations

#### Physics-Based Image Verification (Layer 2)
- Phase 1: 8 core analysis methods — noise uniformity, noise frequency (FFT), frequency domain (DCT), geometric consistency, lighting analysis, texture patterns, color distribution, compression artifacts (multi-quality ELA)
- Phase 2: 4 advanced detection methods — noise consistency mapping (splice detection), JPEG ghost detection, spectral fingerprinting (GAN detection), cross-channel correlation analysis
- Phase 3: 4 forensic methods — Benford's law on DCT coefficients, wavelet detail ratio, blocking artifact grid detection, CFA demosaicing pattern analysis
- Composite scoring with calibrated weights and 16 AI indicator slots
- 100% accuracy on real-world images (9/9 real, 4/4 StyleGAN2)

#### Audio Verification
- 10 signal-processing analysis methods: phase coherence, voice quality (jitter/shimmer/HNR), ENF hum detection, spectral tilt, noise floor consistency, mel-spectrogram regularity, formant bandwidth, double compression detection, spectral discontinuity (splice detection), sub-band energy distribution
- Detects synthetic/TTS audio, pure tones, and splice edits
- 100% benchmark accuracy (5/5)

#### Video Verification
- 6 temporal analysis methods: per-frame image analysis, temporal noise consistency, optical flow analysis, encoding artifact detection, temporal lighting analysis, frame stability (NCC)
- Detects looped/static video, flickering artifacts, temporal inconsistencies
- 100% benchmark accuracy (5/5)

#### TypeScript Ports
- Pure-JS imaging library (`core/src/imaging/`): decode (JPEG/PNG), processing (convolution, Sobel, Laplacian, FFT, DCT, Canny, color conversion), stats (mean, std, entropy, histogram)
- Complete PhysicsVerifier port with all 16 image analysis methods
- AudioVerifier port with all 10 signal-processing methods
- VideoVerifier port with block-matching optical flow (pure JS, no OpenCV)
- `pngjs` dependency added for PNG decode support

#### Testing
- Comprehensive pytest suite: 108 tests across 8 test files
- Test coverage: types, crypto, manifest, CLI, physics, audio, video, verify
- Shared fixtures in `conftest.py` (key pairs, sample content generators)
- 79%+ code coverage on full suite

#### Tooling
- isort configuration (black-compatible profile)
- bandit security scanning configuration
- mypy overrides for untyped C extension modules (cv2, pywt, scipy)
- Coverage threshold set to 60%

### Fixed
- Audio/video content types not routed to physics verification in `verify.py`
- Content type detection expanded: WAV, MP4, AVI, FLAC, OGG, WebP, MP3
- NaN guard on `np.corrcoef` in audio spectral consistency
- `scipy.fft.dct` import moved outside loop in audio double-compression analysis
- StopIteration crash in physics benchmark when no JPEGs found
- CLI `--device` flag wired through to `sign_content()`
- Dead code removed from physics score key computation
- Fragile noise map reshape replaced with explicit size handling

### Security
- RSA-2048 signature algorithm
- SHA-256 hashing
- Content hash verification to detect tampering
- Chain of custody tracking

## [0.0.1] - 2024-02-15

### Added
- Core cryptographic verification library
  - Content signing with RSA private keys
  - Signature verification with public keys
  - SHA-256 content hashing
  - Manifest creation and validation
  
- TypeScript/JavaScript implementation
  - `@asala/core` - Core library
  - `@asala/cli` - Command-line tool
  - `@asala/extension` - Browser extension
  - `@asala/web` - Web interface
  
- Python implementation
  - `asala` package on PyPI
  - Full API compatibility with TypeScript version
  - Command-line interface
  
- Browser extension
  - Chrome and Firefox support
  - Real-time content verification
  - Visual badges for verified content
  - Popup interface for details
  
- Documentation
  - Sphinx documentation with Furo theme
  - API reference for both languages
  - Architecture documentation
  - Quick start guide
  - Contributing guidelines
  
- Testing
  - Jest test suite for TypeScript
  - pytest test suite for Python
  - 80%+ code coverage targets
  - Integration tests
  
- Build system
  - npm workspaces for Node.js packages
  - setuptools for Python package
  - Binary building with pkg and PyInstaller
  - Docker support
  
- CI/CD
  - GitHub Actions for testing
  - Automated releases to npm and PyPI
  - Multi-platform binary builds

### Security
- Implemented cryptographic provenance using C2PA standards
- RSA-2048 bit key generation
- Private key extraction from signing keys
- Content hash mismatch detection

[Unreleased]: https://github.com/your-org/asala/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/your-org/asala/releases/tag/v0.0.1
