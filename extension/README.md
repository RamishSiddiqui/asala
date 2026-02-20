# @asala/extension

Browser extension for real-time content verification (Chrome MV3 / Firefox).

## Installation

### Chrome

1. Clone this repository
2. Run `npm install && npm run build:chrome` in the extension directory
3. Open Chrome and navigate to `chrome://extensions/`
4. Enable "Developer mode"
5. Click "Load unpacked" and select the `dist` directory

### Firefox

1. Clone this repository
2. Run `npm install && npm run build:firefox` in the extension directory
3. Open Firefox and navigate to `about:debugging`
4. Click "This Firefox"
5. Click "Load Temporary Add-on" and select `dist/manifest.json`

## Features

- Real-time content verification
- Visual badges for verified content
- Popup interface for verification details
- Support for images, videos, and audio

## Development

```bash
npm install
npm run dev  # Watch mode
npm run build  # Production build
```

## Documentation

See the [main README](https://github.com/RamishSiddiqui/asala#readme) for full documentation.

## License

GPL-3.0
