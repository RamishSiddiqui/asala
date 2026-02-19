# Contributing to Asala

Thank you for your interest in contributing to Asala! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code:

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Respect different viewpoints and experiences

## How to Contribute

### Reporting Bugs

Before creating a bug report:

1. Check if the issue already exists
2. Try the latest version to see if it's been fixed
3. Collect information about the bug (steps to reproduce, error messages, environment)

When creating a bug report, include:
- Clear title and description
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, browser, versions)
- Screenshots if applicable

### Suggesting Features

Feature requests are welcome! Please:

1. Check if the feature has already been suggested
2. Provide a clear use case
3. Explain why it would be useful
4. Consider how it fits with the project's goals

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`npm test`)
5. Commit with clear messages (`git commit -m 'Add amazing feature'`)
6. Push to your fork (`git push origin feature/amazing-feature`)
7. Open a Pull Request

#### Pull Request Guidelines

- Follow existing code style
- Add tests for new features
- Update documentation as needed
- Keep changes focused and atomic
- Reference related issues

## Development Setup

```bash
# Clone the repository
git clone https://github.com/your-org/asala.git
cd asala

# Install Node.js dependencies (runs build via postinstall)
npm install

# Install Python dependencies
pip install -e ".[dev]"

# Build all TypeScript packages
npm run build

# Run all tests
npm run test              # TypeScript (Jest)
pytest python/tests -v    # Python (pytest)
```

### Project Structure

```
asala/
├── core/              # TypeScript core library
│   └── src/
│       ├── crypto/       # Hashing, signing, ELA
│       ├── imaging/      # Pure-JS image processing (FFT, DCT, convolution)
│       ├── types/        # TypeScript interfaces
│       └── verifiers/    # Physics, audio, video verifiers
├── python/            # Python implementation
│   ├── asala/            # Package (verify, crypto, physics, audio, video)
│   └── tests/            # pytest suite (108 tests)
├── cli/               # Node.js CLI
├── extension/         # Browser extension (Chrome/Firefox)
├── web/               # Next.js web interface
├── docs/              # Sphinx documentation
└── examples/          # Usage examples
```

### Testing

```bash
# TypeScript
npm test                                    # All packages
npm run test --workspace=@asala/core        # Core only

# Python
pytest python/tests -v                      # All tests
pytest python/tests -m "not slow"           # Skip slow tests
pytest python/tests --cov=asala             # With coverage
```

### Code Style

**TypeScript:**
- ESLint for linting
- Prettier for formatting

```bash
npm run lint
npm run format
```

**Python:**
- black for formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking
- bandit for security scanning

```bash
black python/                 # Format
isort python/                 # Sort imports
flake8 python/asala           # Lint
mypy python/asala             # Type check
bandit -r python/asala        # Security scan
```

## Areas for Contribution

### High Priority

- [ ] C2PA manifest embedding in images
- [ ] Distributed consensus layer (Layer 3)
- [ ] Browser extension improvements
- [ ] Mobile app development
- [x] Performance optimizations for physics verification (parallel processing via `max_workers`)

### Documentation

- [ ] API documentation improvements
- [ ] Tutorial videos
- [ ] Translation to other languages
- [ ] Integration guides

### Testing

- [ ] Unit tests for edge cases
- [ ] Integration tests
- [ ] Browser compatibility testing
- [ ] Performance benchmarks

## Questions?

- Join our Discord: [Link]
- Check existing discussions
- Open a GitHub Discussion

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to making the internet more trustworthy!
