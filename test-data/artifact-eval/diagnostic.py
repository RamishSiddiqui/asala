"""Diagnostic: compare real photos vs StyleGAN2 faces on key metrics."""
import sys, importlib.util
from pathlib import Path

_pkg_dir = Path(__file__).resolve().parents[2] / "python" / "asala"

def _load(name, fp):
    spec = importlib.util.spec_from_file_location(name, fp)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_load("asala.types", str(_pkg_dir / "types.py"))
physics = _load("asala.physics", str(_pkg_dir / "physics.py"))
v = physics.PhysicsVerifier()

base = Path(__file__).resolve().parent

print("=" * 110)
print("DIAGNOSTIC: Real Photos vs StyleGAN2 Faces")
print("=" * 110)

def get_metrics(result):
    d = result.details
    n = d.get('noise_uniformity', {})
    t = d.get('texture_analysis', {})
    g = d.get('geometric_consistency', {})
    c = d.get('compression_analysis', {})
    return {
        'score': result.score,
        'passed': result.passed,
        'cv_ratio': n.get('cv_ratio', 0),
        'residual_cv': n.get('residual_cv', 0),
        'noise_score': n.get('uniformity_score', 0),
        'grad_cv': t.get('grad_cv', 0),
        'region_cv': t.get('region_cv', 0),
        'texture_score': t.get('texture_score', 0),
        'geo_score': g.get('geometric_score', 0),
        'ela_cv': c.get('regional_ela_cv', 0),
        'compression_score': c.get('compression_score', 0),
        'indicators': d.get('ai_indicators', 0),
    }

header = f"  {'':2s} {'Image':27s} {'Score':>5s} {'noise':>5s} {'cv_r':>6s} {'res_cv':>6s} {'txt':>5s} {'g_cv':>6s} {'r_cv':>6s} {'geo':>5s} {'ela_cv':>6s} {'ind':>3s}"
print(header)
print("-" * 110)

for label, subdir in [("REAL", "real"), ("STYLEGAN2", "ai-generated")]:
    d = base / subdir
    if not d.exists():
        continue
    print(f"\n  --- {label} ---")
    for fpath in sorted(d.iterdir()):
        if fpath.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        r = v.verify_image(fpath.read_bytes())
        m = get_metrics(r)
        tag = "OK" if (label == "REAL" and m['passed']) or (label != "REAL" and not m['passed']) else "XX"
        print(f"  {tag} {fpath.name:27s} {m['score']:5d} {m['noise_score']:5d} "
              f"{m['cv_ratio']:6.3f} {m['residual_cv']:6.3f} {m['texture_score']:5d} "
              f"{m['grad_cv']:6.2f} {m['region_cv']:6.3f} {m['geo_score']:5d} "
              f"{m['ela_cv']:6.3f} {m['indicators']:3d}")

print("\n" + "=" * 110)
print("KEY FINDINGS")
print("=" * 110)
print("""
StyleGAN2 faces from thispersondoesnotexist.com are production-quality
1024x1024 images. Unlike our simple programmatic test synthetics:

  1. StyleGAN2 INJECTS spatially-varying noise at multiple scales
     -> noise cv_ratio and residual_cv are HIGH, bypassing noise analysis
  2. Generates realistic fine texture (hair, skin pores)
     -> gradient CV is natural-looking
  3. Has proper geometric structure (symmetric faces)
     -> moderate-to-high geometric scores
  4. Uses progressive growing for natural frequency distribution

Detection of production GANs requires more advanced methods:
  - GAN fingerprinting (model-specific spectral signatures)
  - Facial landmark consistency analysis
  - Pupil/teeth regularity checks
  - High-frequency spectral analysis in Fourier domain
""")
