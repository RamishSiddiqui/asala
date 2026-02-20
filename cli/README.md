# @asala/cli

Command-line tool for content verification and signing.

## Installation

```bash
npm install -g @asala/cli
```

## Usage

### Verify content

```bash
asala verify ./image.jpg
asala verify ./image.jpg --manifest ./manifest.json
asala verify ./image.jpg --trust ./public-key.pem
```

### Sign content

```bash
asala sign ./image.jpg --key ./private-key.pem --creator "Your Name"
```

### Generate keys

```bash
asala keys --generate --output ./keys
```

### View manifest

```bash
asala manifest ./image.jpg
asala manifest ./image.jpg --extract --output ./manifest.json
```

## Options

- `--json` - Output results as JSON
- `--verbose` - Verbose output
- `--trust <keys...>` - Specify trusted public keys

## Documentation

See the [main README](https://github.com/RamishSiddiqui/asala#readme) for full documentation.

## License

GPL-3.0
