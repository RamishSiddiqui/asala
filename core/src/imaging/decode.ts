/**
 * Image decoding utilities.
 * Supports JPEG (via jpeg-js) and PNG (via pngjs).
 */
import jpeg from 'jpeg-js';
import { PNG } from 'pngjs';

export interface ImageData {
  width: number;
  height: number;
  data: Uint8Array; // RGBA interleaved
}

/**
 * Decode an image buffer, auto-detecting format from magic bytes.
 * Supports JPEG (FF D8) and PNG (89 50).
 */
export function decodeImage(buffer: Buffer): ImageData {
  if (buffer.length < 4) {
    throw new Error('Buffer too small to be a valid image');
  }
  if (buffer[0] === 0xff && buffer[1] === 0xd8) {
    return decodeJPEG(buffer);
  }
  if (buffer[0] === 0x89 && buffer[1] === 0x50) {
    return decodePNG(buffer);
  }
  throw new Error('Unsupported image format (expected JPEG or PNG)');
}

function decodeJPEG(buffer: Buffer): ImageData {
  const decoded = jpeg.decode(buffer, { useTArray: true });
  return {
    width: decoded.width,
    height: decoded.height,
    data: new Uint8Array(decoded.data),
  };
}

function decodePNG(buffer: Buffer): ImageData {
  const png = PNG.sync.read(buffer);
  return {
    width: png.width,
    height: png.height,
    data: new Uint8Array(png.data),
  };
}
