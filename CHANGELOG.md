# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial implementation of cryptographic content verification
- Multi-language support (TypeScript/JavaScript and Python)
- Browser extension for real-time content verification
- Command-line interface for verification and signing
- Comprehensive test suite (Jest for TypeScript, pytest for Python)
- Sphinx documentation with Furo theme
- CI/CD pipelines for automated testing and releases
- Binary building for Windows, macOS, and Linux

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
