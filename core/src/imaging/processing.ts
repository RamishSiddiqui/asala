/**
 * Pure-JS image processing primitives.
 * Operates on flat Float64Array grayscale buffers unless noted otherwise.
 */

// ---------------------------------------------------------------------------
// Grayscale conversion
// ---------------------------------------------------------------------------

/** Convert RGBA Uint8Array to single-channel float array (ITU-R BT.601). */
export function toGrayscale(rgba: Uint8Array, w: number, h: number): Float64Array {
  const gray = new Float64Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const off = i * 4;
    gray[i] = 0.299 * rgba[off] + 0.587 * rgba[off + 1] + 0.114 * rgba[off + 2];
  }
  return gray;
}

// ---------------------------------------------------------------------------
// Convolution
// ---------------------------------------------------------------------------

/** Generic 2D convolution with zero-padded borders. */
export function convolve2d(
  img: Float64Array,
  w: number,
  h: number,
  kernel: Float64Array,
  kw: number,
  kh: number
): Float64Array {
  const out = new Float64Array(w * h);
  const khalf = Math.floor(kh / 2);
  const kwhalf = Math.floor(kw / 2);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let ky = 0; ky < kh; ky++) {
        for (let kx = 0; kx < kw; kx++) {
          const iy = y + ky - khalf;
          const ix = x + kx - kwhalf;
          if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
            sum += img[iy * w + ix] * kernel[ky * kw + kx];
          }
        }
      }
      out[y * w + x] = sum;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Laplacian
// ---------------------------------------------------------------------------

/** Laplacian with hardcoded 3×3 kernel [[0,1,0],[1,-4,1],[0,1,0]]. */
export function laplacian(gray: Float64Array, w: number, h: number): Float64Array {
  const out = new Float64Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = y * w + x;
      out[idx] =
        gray[idx - w] +          // top
        gray[idx - 1] +          // left
        -4 * gray[idx] +         // center
        gray[idx + 1] +          // right
        gray[idx + w];           // bottom
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Sobel
// ---------------------------------------------------------------------------

/** Sobel X and Y gradient arrays with hardcoded 3×3 kernels. */
export function sobelXY(
  gray: Float64Array,
  w: number,
  h: number
): { gx: Float64Array; gy: Float64Array } {
  const gx = new Float64Array(w * h);
  const gy = new Float64Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = y * w + x;
      const tl = gray[idx - w - 1];
      const tc = gray[idx - w];
      const tr = gray[idx - w + 1];
      const ml = gray[idx - 1];
      const mr = gray[idx + 1];
      const bl = gray[idx + w - 1];
      const bc = gray[idx + w];
      const br = gray[idx + w + 1];
      // Sobel X: [[-1,0,1],[-2,0,2],[-1,0,1]]
      gx[idx] = -tl + tr - 2 * ml + 2 * mr - bl + br;
      // Sobel Y: [[-1,-2,-1],[0,0,0],[1,2,1]]
      gy[idx] = -tl - 2 * tc - tr + bl + 2 * bc + br;
    }
  }
  return { gx, gy };
}

/** sqrt(gx^2 + gy^2) element-wise. */
export function gradientMagnitude(gx: Float64Array, gy: Float64Array): Float64Array {
  const out = new Float64Array(gx.length);
  for (let i = 0; i < gx.length; i++) {
    out[i] = Math.sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Median filter
// ---------------------------------------------------------------------------

/**
 * In-place quickselect: rearranges arr[lo..hi] so that arr[k] is the
 * k-th smallest element.  O(n) expected.
 */
function quickselect(arr: number[], lo: number, hi: number, k: number): void {
  while (lo < hi) {
    const pivot = arr[lo + ((hi - lo) >> 1)];
    let i = lo, j = hi;
    while (i <= j) {
      while (arr[i] < pivot) i++;
      while (arr[j] > pivot) j--;
      if (i <= j) {
        const tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
        i++; j--;
      }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else break;
  }
}

/** Sliding-window median filter (default 5x5 with radius=2). */
export function medianFilter(
  gray: Float64Array,
  w: number,
  h: number,
  radius: number = 2
): Float64Array {
  const out = new Float64Array(w * h);
  const size = 2 * radius + 1;
  const buf: number[] = new Array(size * size);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let count = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const ny = y + dy;
          const nx = x + dx;
          if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
            buf[count++] = gray[ny * w + nx];
          }
        }
      }
      const k = count >> 1;
      quickselect(buf, 0, count - 1, k);
      out[y * w + x] = buf[k];
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Gaussian blur
// ---------------------------------------------------------------------------

/** Gaussian blur via separable 1D passes (O(2k) instead of O(k²) per pixel). */
export function gaussianBlur(
  gray: Float64Array,
  w: number,
  h: number,
  sigma: number
): Float64Array {
  const ks = Math.max(3, Math.ceil(sigma * 6) | 1); // odd kernel size
  const half = Math.floor(ks / 2);

  // Build 1D Gaussian kernel
  const kernel1d = new Float64Array(ks);
  let sum = 0;
  for (let i = -half; i <= half; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel1d[i + half] = v;
    sum += v;
  }
  for (let i = 0; i < ks; i++) kernel1d[i] /= sum;

  // Horizontal pass
  const tmp = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = -half; k <= half; k++) {
        const ix = x + k;
        if (ix >= 0 && ix < w) {
          s += gray[y * w + ix] * kernel1d[k + half];
        }
      }
      tmp[y * w + x] = s;
    }
  }

  // Vertical pass
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = -half; k <= half; k++) {
        const iy = y + k;
        if (iy >= 0 && iy < h) {
          s += tmp[iy * w + x] * kernel1d[k + half];
        }
      }
      out[y * w + x] = s;
    }
  }

  return out;
}

// ---------------------------------------------------------------------------
// Canny edge detection (simplified)
// ---------------------------------------------------------------------------

/**
 * Simplified Canny edge detection.
 * Returns a binary edge map (0 or 255) as Uint8Array.
 */
export function cannyEdges(
  gray: Float64Array,
  w: number,
  h: number,
  lowT: number,
  highT: number
): Uint8Array {
  // 1. Gaussian blur
  const blurred = gaussianBlur(gray, w, h, 1.4);

  // 2. Sobel gradients
  const { gx, gy } = sobelXY(blurred, w, h);
  const mag = gradientMagnitude(gx, gy);

  // 3. Gradient direction (quantised to 0, 45, 90, 135 degrees)
  const dir = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    let angle = Math.atan2(gy[i], gx[i]) * (180 / Math.PI);
    if (angle < 0) angle += 180;
    if (angle < 22.5 || angle >= 157.5) dir[i] = 0;       // horizontal
    else if (angle < 67.5) dir[i] = 45;
    else if (angle < 112.5) dir[i] = 90;                    // vertical
    else dir[i] = 135;
  }

  // 4. Non-maximum suppression
  const nms = new Float64Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = y * w + x;
      const m = mag[idx];
      let n1 = 0, n2 = 0;
      switch (dir[idx]) {
        case 0:   n1 = mag[idx - 1]; n2 = mag[idx + 1]; break;
        case 45:  n1 = mag[(y - 1) * w + (x + 1)]; n2 = mag[(y + 1) * w + (x - 1)]; break;
        case 90:  n1 = mag[(y - 1) * w + x]; n2 = mag[(y + 1) * w + x]; break;
        case 135: n1 = mag[(y - 1) * w + (x - 1)]; n2 = mag[(y + 1) * w + (x + 1)]; break;
      }
      nms[idx] = (m >= n1 && m >= n2) ? m : 0;
    }
  }

  // 5. Double thresholding + hysteresis
  const edge = new Uint8Array(w * h); // 0=none, 1=weak, 2=strong
  for (let i = 0; i < w * h; i++) {
    if (nms[i] >= highT) edge[i] = 2;
    else if (nms[i] >= lowT) edge[i] = 1;
  }

  // Hysteresis: BFS from strong edges into weak edges
  const result = new Uint8Array(w * h);
  const stack: number[] = [];
  for (let i = 0; i < w * h; i++) {
    if (edge[i] === 2) {
      result[i] = 255;
      stack.push(i);
    }
  }
  while (stack.length > 0) {
    const idx = stack.pop()!;
    const x = idx % w;
    const y = (idx - x) / w;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (dy === 0 && dx === 0) continue;
        const ny = y + dy, nx = x + dx;
        if (ny >= 0 && ny < h && nx >= 0 && nx < w) {
          const ni = ny * w + nx;
          if (edge[ni] === 1 && result[ni] === 0) {
            result[ni] = 255;
            edge[ni] = 2;
            stack.push(ni);
          }
        }
      }
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Color space conversions
// ---------------------------------------------------------------------------

/** RGB [0-255] → HSV with OpenCV-compatible ranges: H [0-180], S [0-255], V [0-255]. */
export function rgbToHSV(r: number, g: number, b: number): [number, number, number] {
  const rf = r / 255, gf = g / 255, bf = b / 255;
  const cmax = Math.max(rf, gf, bf);
  const cmin = Math.min(rf, gf, bf);
  const d = cmax - cmin;

  let h: number;
  if (d === 0) h = 0;
  else if (cmax === rf) h = 60 * (((gf - bf) / d + 6) % 6);
  else if (cmax === gf) h = 60 * ((bf - rf) / d + 2);
  else h = 60 * ((rf - gf) / d + 4);

  h = h / 2; // scale to [0, 180]
  const s = cmax === 0 ? 0 : (d / cmax) * 255;
  const v = cmax * 255;
  return [h, s, v];
}

/**
 * RGB [0-255] → CIE L*a*b* with OpenCV-compatible ranges:
 * L [0-255] (L* 0-100 scaled), a [0-255] (128 = neutral), b [0-255] (128 = neutral).
 */
export function rgbToLAB(r: number, g: number, b: number): [number, number, number] {
  // sRGB → linear
  const linearize = (c: number) => {
    const cf = c / 255;
    return cf > 0.04045 ? Math.pow((cf + 0.055) / 1.055, 2.4) : cf / 12.92;
  };
  const rl = linearize(r), gl = linearize(g), bl = linearize(b);

  // Linear RGB → XYZ (D65)
  const x = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl;
  const y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl;
  const z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl;

  // D65 white point
  const xn = 0.95047, yn = 1.0, zn = 1.08883;
  const fv = (t: number) => (t > 0.008856 ? Math.cbrt(t) : 7.787037 * t + 16 / 116);

  const fx = fv(x / xn);
  const fy = fv(y / yn);
  const fz = fv(z / zn);

  const L = 116 * fy - 16;         // [0, 100]
  const a = 500 * (fx - fy);       // roughly [-128, 127]
  const bv = 200 * (fy - fz);      // roughly [-128, 127]

  // Scale to OpenCV uint8 range
  return [L * 255 / 100, a + 128, bv + 128];
}

// ---------------------------------------------------------------------------
// FFT (Cooley-Tukey radix-2, in-place)
// ---------------------------------------------------------------------------

function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/**
 * In-place Cooley-Tukey radix-2 FFT.
 * `re` and `im` must have length that is a power of 2.
 */
export function fft1d(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  if (n <= 1) return;

  // Bit-reversal permutation
  let j = 0;
  for (let i = 1; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j) {
      let tmp = re[i]; re[i] = re[j]; re[j] = tmp;
      tmp = im[i]; im[i] = im[j]; im[j] = tmp;
    }
  }

  // Butterfly stages
  for (let len = 2; len <= n; len *= 2) {
    const halfLen = len >> 1;
    const angle = -2 * Math.PI / len;
    const wRe = Math.cos(angle);
    const wIm = Math.sin(angle);

    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let k = 0; k < halfLen; k++) {
        const evenIdx = i + k;
        const oddIdx = i + k + halfLen;
        const tRe = curRe * re[oddIdx] - curIm * im[oddIdx];
        const tIm = curRe * im[oddIdx] + curIm * re[oddIdx];
        re[oddIdx] = re[evenIdx] - tRe;
        im[oddIdx] = im[evenIdx] - tIm;
        re[evenIdx] += tRe;
        im[evenIdx] += tIm;
        const newCur = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = newCur;
      }
    }
  }
}

/**
 * 2D FFT via row-then-column 1D FFTs.
 * Input is zero-padded to the next power-of-2 dimensions.
 * Returns complex arrays plus the padded dimensions.
 */
export function fft2d(
  gray: Float64Array,
  w: number,
  h: number
): { re: Float64Array; im: Float64Array; fw: number; fh: number } {
  const fw = nextPow2(w);
  const fh = nextPow2(h);
  const size = fw * fh;

  const re = new Float64Array(size);
  const im = new Float64Array(size);

  // Copy and zero-pad
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      re[y * fw + x] = gray[y * w + x];
    }
  }

  // Row-wise FFT
  const rowRe = new Float64Array(fw);
  const rowIm = new Float64Array(fw);
  for (let y = 0; y < fh; y++) {
    const off = y * fw;
    for (let x = 0; x < fw; x++) { rowRe[x] = re[off + x]; rowIm[x] = 0; }
    fft1d(rowRe, rowIm);
    for (let x = 0; x < fw; x++) { re[off + x] = rowRe[x]; im[off + x] = rowIm[x]; }
  }

  // Column-wise FFT
  const colRe = new Float64Array(fh);
  const colIm = new Float64Array(fh);
  for (let x = 0; x < fw; x++) {
    for (let y = 0; y < fh; y++) { colRe[y] = re[y * fw + x]; colIm[y] = im[y * fw + x]; }
    fft1d(colRe, colIm);
    for (let y = 0; y < fh; y++) { re[y * fw + x] = colRe[y]; im[y * fw + x] = colIm[y]; }
  }

  return { re, im, fw, fh };
}

// ---------------------------------------------------------------------------
// DCT (Type-II via reorder + FFT)
// ---------------------------------------------------------------------------

/** 1D Type-II DCT of length N (N must be power of 2) via FFT reorder trick. */
function dct1d(x: Float64Array): Float64Array {
  const N = x.length;
  // Reorder: even-indexed first, odd-indexed reversed
  // v serves as both reorder buffer and FFT real input (avoids extra copy)
  const v = new Float64Array(N);
  for (let k = 0; k < (N >> 1); k++) {
    v[k] = x[2 * k];
  }
  for (let k = 0; k < (N >> 1); k++) {
    v[N - 1 - k] = x[2 * k + 1];
  }

  // N-point FFT (v is modified in-place as the real part)
  const im = new Float64Array(N);
  fft1d(v, im);

  // Multiply by twiddle factor and write result into v (reuse buffer)
  for (let k = 0; k < N; k++) {
    const angle = (-Math.PI * k) / (2 * N);
    v[k] = 2 * (v[k] * Math.cos(angle) - im[k] * Math.sin(angle));
  }
  return v;
}

/**
 * 2D Type-II DCT via row-column 1D DCTs.
 * Resizes to a power-of-2 square if needed.
 */
export function dct2d(gray: Float64Array, w: number, h: number): Float64Array {
  const out = new Float64Array(w * h);

  // Copy input
  for (let i = 0; i < w * h; i++) out[i] = gray[i];

  // Pre-allocate row/col buffers (reused across iterations)
  const row = new Float64Array(w);
  const col = new Float64Array(h);

  // Row-wise DCT
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) row[x] = out[y * w + x];
    const dctRow = dct1d(row);
    for (let x = 0; x < w; x++) out[y * w + x] = dctRow[x];
  }

  // Column-wise DCT
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) col[y] = out[y * w + x];
    const dctCol = dct1d(col);
    for (let y = 0; y < h; y++) out[y * w + x] = dctCol[y];
  }

  return out;
}

// ---------------------------------------------------------------------------
// Bilinear resize
// ---------------------------------------------------------------------------

/** Bilinear interpolation resize for grayscale images. */
export function resize(
  gray: Float64Array,
  w: number,
  h: number,
  newW: number,
  newH: number
): Float64Array {
  const result = new Float64Array(newW * newH);
  const xRatio = w / newW;
  const yRatio = h / newH;

  for (let y = 0; y < newH; y++) {
    for (let x = 0; x < newW; x++) {
      const srcX = x * xRatio;
      const srcY = y * yRatio;
      const x0 = Math.floor(srcX);
      const y0 = Math.floor(srcY);
      const x1 = Math.min(x0 + 1, w - 1);
      const y1 = Math.min(y0 + 1, h - 1);
      const xf = srcX - x0;
      const yf = srcY - y0;
      result[y * newW + x] =
        gray[y0 * w + x0] * (1 - xf) * (1 - yf) +
        gray[y0 * w + x1] * xf * (1 - yf) +
        gray[y1 * w + x0] * (1 - xf) * yf +
        gray[y1 * w + x1] * xf * yf;
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// FFT shift (swap quadrants for centered spectrum)
// ---------------------------------------------------------------------------

/** Swap quadrants of a 2D complex FFT result so DC is at center. */
export function fftShift(
  data: Float64Array,
  w: number,
  h: number
): Float64Array {
  const out = new Float64Array(w * h);
  const hw = w >> 1;
  const hh = h >> 1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const ny = (y + hh) % h;
      const nx = (x + hw) % w;
      out[ny * w + nx] = data[y * w + x];
    }
  }
  return out;
}
