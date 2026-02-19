/**
 * Statistical helper functions for image analysis.
 */

export function mean(arr: Float64Array | number[]): number {
  const len = arr.length;
  if (len === 0) return 0;
  let sum = 0;
  for (let i = 0; i < len; i++) sum += arr[i];
  return sum / len;
}

export function variance(arr: Float64Array | number[]): number {
  const len = arr.length;
  if (len === 0) return 0;
  const m = mean(arr);
  let sum = 0;
  for (let i = 0; i < len; i++) {
    const d = arr[i] - m;
    sum += d * d;
  }
  return sum / len;
}

export function std(arr: Float64Array | number[]): number {
  return Math.sqrt(variance(arr));
}

export function cv(arr: Float64Array | number[]): number {
  const m = mean(arr);
  if (Math.abs(m) < 1e-10) return 0;
  return std(arr) / m;
}

export function histogram(
  arr: Float64Array | number[],
  bins: number,
  rangeMin?: number,
  rangeMax?: number
): Float64Array {
  let lo = rangeMin;
  let hi = rangeMax;
  if (lo === undefined || hi === undefined) {
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < arr.length; i++) {
      if (arr[i] < mn) mn = arr[i];
      if (arr[i] > mx) mx = arr[i];
    }
    if (lo === undefined) lo = mn;
    if (hi === undefined) hi = mx;
  }
  const hist = new Float64Array(bins);
  const range = hi - lo || 1;
  for (let i = 0; i < arr.length; i++) {
    let bin = Math.floor(((arr[i] - lo) / range) * bins);
    if (bin >= bins) bin = bins - 1;
    if (bin < 0) bin = 0;
    hist[bin]++;
  }
  return hist;
}

export function entropy(hist: Float64Array | number[]): number {
  let total = 0;
  for (let i = 0; i < hist.length; i++) total += hist[i];
  if (total === 0) return 0;
  let ent = 0;
  for (let i = 0; i < hist.length; i++) {
    const p = hist[i] / total;
    if (p > 0) ent -= p * Math.log2(p);
  }
  return ent;
}

/**
 * In-place quickselect: rearranges arr so that arr[k] is the k-th
 * smallest element.  O(n) expected.
 */
function qselect(arr: number[], lo: number, hi: number, k: number): void {
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

export function percentile(arr: Float64Array | number[], p: number): number {
  if (arr.length === 0) return 0;
  const data = Array.from(arr);
  const idx = (p / 100) * (data.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) {
    qselect(data, 0, data.length - 1, lo);
    return data[lo];
  }
  // Need both lo and hi values — select lo first, then hi from remaining
  qselect(data, 0, data.length - 1, lo);
  const loVal = data[lo];
  // After qselect for lo, data[lo+1..end] are all >= data[lo],
  // so the min of data[lo+1..end] is data[hi]
  qselect(data, lo + 1, data.length - 1, hi);
  const hiVal = data[hi];
  return loVal + (hiVal - loVal) * (idx - lo);
}

export function clamp(val: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, val));
}

export function linmap(
  val: number,
  inLo: number,
  inHi: number,
  outLo: number,
  outHi: number
): number {
  const t = (val - inLo) / (inHi - inLo);
  return outLo + t * (outHi - outLo);
}
