# Sample Test Data for Asala

This directory contains sample images and media for testing the verification system.

## Contents

- `original/` - Original unmodified images
- `signed/` - Images with embedded C2PA manifests
- `tampered/` - Modified images (for testing detection)
- `corrupted/` - Corrupted manifests (for testing error handling)

## Test Images

### Sample-01: Basic JPEG
- File: `original/sample-01.jpg`
- Size: 1920x1080
- Format: JPEG
- Description: Simple landscape photo for basic testing

### Sample-02: PNG with Transparency
- File: `original/sample-02.png`
- Size: 512x512
- Format: PNG with alpha channel
- Description: Logo/Icon for testing transparency handling

### Sample-03: Signed Image
- File: `signed/sample-03-signed.jpg`
- Size: 1920x1080
- Signed by: Test Creator
- Description: Image with valid C2PA manifest embedded

### Sample-04: Tampered Image
- File: `tampered/sample-04-tampered.jpg`
- Original: `original/sample-01.jpg`
- Modification: Resized and watermarked
- Description: Should fail verification

## Generating Test Data

### Create Signed Image
```bash
# Using CLI
asala sign sample-01.jpg --key keys/private.pem --creator "Test Creator"
```

### Verify Image
```bash
asala verify sample-03-signed.jpg
```

### Detect Tampering
```bash
asala verify sample-04-tampered.jpg
# Expected: Verification failed - Content hash mismatch
```

## License

All sample images are created for testing purposes and are released under CC0 (public domain).
