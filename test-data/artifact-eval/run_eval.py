"""Evaluate physics verifier against real and AI-generated images from public sources."""
import sys, importlib.util, os
from pathlib import Path

# Load physics verifier directly (bypass asala.__init__)
_pkg_dir = Path(__file__).resolve().parents[2] / "python" / "asala"

def _load(name, fp):
    spec = importlib.util.spec_from_file_location(name, fp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_load("asala.types", str(_pkg_dir / "types.py"))
physics = _load("asala.physics", str(_pkg_dir / "physics.py"))

verifier = physics.PhysicsVerifier()
base = Path(__file__).resolve().parent

print("=" * 78)
print("ARTIFACT DATASET EVALUATION - Physics Verifier Layer 2")
print("=" * 78)

def print_result(fpath, result, expect_real):
    if expect_real:
        status = "PASS" if result.passed else "FAIL"
    else:
        status = "PASS" if not result.passed else "FAIL"
    warn = result.details.get("warning", "")
    ai_prob = result.details.get("ai_probability", 0)
    indicators = result.details.get("ai_indicators", 0)
    print(f"  [{status}] {fpath.name:30s}  score={result.score:3d}  ai_prob={ai_prob:.2f}  indicators={indicators}")
    if warn:
        print(f"         {warn}")
    sub_scores = []
    for key in ['noise_uniformity', 'noise_frequency', 'frequency_analysis',
                 'geometric_consistency', 'lighting_analysis', 'texture_analysis',
                 'color_analysis', 'compression_analysis']:
        sub = result.details.get(key, {})
        if isinstance(sub, dict):
            for k, v in sub.items():
                if k.endswith('_score') and isinstance(v, (int, float)):
                    short = key.replace('_analysis', '').replace('_consistency', '').replace('noise_', 'n_')
                    sub_scores.append(f"{short}={int(v)}")
    print(f"         [{', '.join(sub_scores)}]")

# --- Real Images ---
print("\n" + "-" * 78)
print("REAL PHOTOS (Unsplash via picsum.photos)  [Expected: passed=True, score>=50]")
print("-" * 78)

real_dir = base / "real"
real_results = []
for fpath in sorted(real_dir.iterdir()):
    if fpath.suffix.lower() in ('.jpg', '.jpeg', '.png'):
        result = verifier.verify_image(fpath.read_bytes())
        real_results.append((fpath.name, result))
        print_result(fpath, result, expect_real=True)

# --- AI-Generated Images ---
print("\n" + "-" * 78)
print("AI-GENERATED IMAGES  [Expected: passed=False, score<50]")
print("-" * 78)

ai_dir = base / "ai-generated"
ai_results = []
for fpath in sorted(ai_dir.iterdir()):
    if fpath.suffix.lower() in ('.jpg', '.jpeg', '.png'):
        result = verifier.verify_image(fpath.read_bytes())
        ai_results.append((fpath.name, result))
        print_result(fpath, result, expect_real=False)

# --- Summary ---
print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)

real_correct = sum(1 for _, r in real_results if r.passed)
real_total = len(real_results)
ai_correct = sum(1 for _, r in ai_results if not r.passed)
ai_total = len(ai_results)
total_correct = real_correct + ai_correct
total = real_total + ai_total

print(f"  Real photos correctly identified:    {real_correct}/{real_total}")
print(f"  AI images correctly identified:      {ai_correct}/{ai_total}")
print(f"  Overall accuracy:                    {total_correct}/{total} ({100*total_correct/max(total,1):.1f}%)")

if real_results:
    real_scores = [r.score for _, r in real_results]
    print(f"\n  Real scores:  min={min(real_scores)}  max={max(real_scores)}  avg={sum(real_scores)/len(real_scores):.1f}")
if ai_results:
    ai_scores = [r.score for _, r in ai_results]
    print(f"  AI scores:    min={min(ai_scores)}  max={max(ai_scores)}  avg={sum(ai_scores)/len(ai_scores):.1f}")

if real_results and ai_results:
    gap = min(real_scores) - max(ai_scores)
    print(f"\n  Score gap (real_min - ai_max): {gap}")
    if gap > 0:
        print("  -> Clean separation!")
    else:
        print("  -> OVERLAP exists")

print("\n" + "=" * 78)
