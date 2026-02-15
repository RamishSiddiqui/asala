import os
import sys
import json
from pathlib import Path

# Add python directory to path to import asala
sys.path.append(os.path.join(os.path.dirname(__file__), '../python'))

from asala.verify import Asala
from asala.types import VerificationOptions, ContentType

def main():
    print("--- Testing Physics Verification (Python) ---")

    # Paths
    base_dir = Path(__file__).parent.parent
    image_path = base_dir / "test-data" / "original" / "sample-landscape.jpg"
    manifest_path = base_dir / "test-data" / "signed" / "sample-landscape.jpg.manifest.json"

    if not image_path.exists() or not manifest_path.exists():
        print("Test data not found!")
        print(f"Image: {image_path}")
        print(f"Manifest: {manifest_path}")
        sys.exit(1)

    # Read files
    with open(image_path, "rb") as f:
        image_buffer = f.read()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    
    # We need to reconstruct ContentManifest object from JSON
    # This is a bit tricky since we don't have a direct JSON deserializer yet
    # For now, let's skip manifest verification and focus on physics
    # Or simplified: use verify_without_manifest but forced to check physics? 
    # verify_without_manifest doesn't support physics check in current implementation.
    
    # Let's verify directly using PhysicsVerifier for this test to be simpler
    from asala.physics import PhysicsVerifier
    
    verifier = PhysicsVerifier()
    print(f"Verifying image: {image_path.name}")
    print(f"Image size: {len(image_buffer)} bytes")
    
    result = verifier.verify_image(image_buffer)
    
    print("\n[Physics Verification Layer]")
    print(f"Passed: {result.passed}")
    print(f"Score: {result.score}")
    print(f"Details: {json.dumps(result.details, indent=2)}")

if __name__ == "__main__":
    main()
