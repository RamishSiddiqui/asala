#!/bin/bash

# Build script for Unmask Verify binaries
# This script builds standalone executables for all platforms

set -e

echo "Building Unmask Verify binaries..."
echo "=================================="

# Build TypeScript/Node.js packages first
echo ""
echo "Step 1: Building Node.js packages..."
npm run build

# Build binaries for different platforms
echo ""
echo "Step 2: Building standalone binaries..."

# Windows
echo "  Building Windows binary..."
npm run binary:win

# macOS
echo "  Building macOS binary..."
npm run binary:mac

# Linux
echo "  Building Linux binary..."
npm run binary:linux

echo ""
echo "Step 3: Building Python wheel..."
cd python
python -m build
cd ..

echo ""
echo "Step 4: Building Python binary with PyInstaller..."
pip install pyinstaller
pyinstaller \
    --onefile \
    --name unmask-python \
    --distpath bin \
    python/unmask_verify/cli.py

echo ""
echo "=================================="
echo "Build complete! Binaries created in ./bin/"
echo ""
echo "Binaries:"
ls -lh bin/
