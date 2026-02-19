Python API
==========

The Python API provides full access to Asala's functionality.

Installation
------------

.. code-block:: bash

    pip install asala

Quick Start
-----------

.. code-block:: python

    from asala import Asala

    # Initialize
    asala = Asala()
    
    # Generate keys
    public_key, private_key = asala.generate_key_pair()
    
    # Sign content
    with open('photo.jpg', 'rb') as f:
        content = f.read()
    
    manifest = asala.sign_content(content, private_key, "Your Name")
    
    # Verify
    result = asala.verify(content, manifest)
    print(f"Status: {result.status.value}")

Module Structure
----------------

.. code-block::

    asala/
    ├── __init__.py          # Main exports
    ├── verify.py            # Asala class (main entry point)
    ├── crypto.py            # CryptoUtils (hashing, signing, key generation)
    ├── manifest.py          # ManifestBuilder (fluent manifest construction)
    ├── types.py             # Type definitions (dataclasses, enums)
    ├── physics.py           # PhysicsVerifier (16 image analysis methods)
    ├── audio.py             # AudioVerifier (10 audio analysis methods)
    ├── video.py             # VideoVerifier (6 video analysis methods)
    └── cli.py               # Command-line interface

Detailed Usage
--------------

Key Management
^^^^^^^^^^^^^^

.. code-block:: python

    from asala import Asala

    asala = Asala()
    
    # Generate RSA key pair
    public_key, private_key = asala.generate_key_pair()
    
    # Save keys
    with open('private.pem', 'w') as f:
        f.write(private_key)
    
    with open('public.pem', 'w') as f:
        f.write(public_key)
    
    # Load keys
    with open('private.pem', 'r') as f:
        private_key = f.read()

Content Signing
^^^^^^^^^^^^^^^

Basic signing:

.. code-block:: python

    from asala import Asala

    asala = Asala()
    
    with open('image.jpg', 'rb') as f:
        content = f.read()
    
    manifest = asala.sign_content(
        content=content,
        private_key=private_key,
        creator="John Doe"
    )

With explicit content type:

.. code-block:: python

    from asala.types import ContentType

    manifest = asala.sign_content(
        content=content,
        private_key=private_key,
        creator="John Doe",
        content_type=ContentType.VIDEO
    )

With metadata:

.. code-block:: python

    from asala.manifest import ManifestBuilder

    manifest_builder = ManifestBuilder(
        content_hash=asala.crypto.hash_content(content),
        content_type=ContentType.IMAGE,
        creator="John Doe"
    )
    
    manifest_builder.add_metadata({
        'location': {'lat': 40.7128, 'lng': -74.0060},
        'camera': 'Canon EOS R5',
        'settings': {'iso': 100, 'aperture': 'f/2.8'}
    })
    
    manifest_builder.add_creation_info(
        device='Canon EOS R5',
        software='Adobe Lightroom',
        location={'lat': 40.7128, 'lng': -74.0060}
    )
    
    manifest_builder.sign(private_key, "John Doe")
    manifest = manifest_builder.build()

Chain of Custody
^^^^^^^^^^^^^^^^

Track edits and transformations:

.. code-block:: python

    from asala.manifest import ManifestBuilder

    # Start with original
    builder = ManifestBuilder(
        content_hash=original_hash,
        content_type=ContentType.IMAGE,
        creator="Photographer"
    )
    
    # Original signature
    builder.sign(photographer_key, "Photographer")
    
    # Editor resizes
    builder.add_chain_link(
        action="resize",
        actor="Editor",
        private_key=editor_key
    )
    
    # Publisher adds watermark
    builder.add_chain_link(
        action="watermark",
        actor="Publisher",
        private_key=publisher_key
    )
    
    manifest = builder.build()

Content Verification
^^^^^^^^^^^^^^^^^^^^

Basic verification:

.. code-block:: python

    result = asala.verify(content, manifest)
    
    if result.status.value == 'verified':
        print("✓ Content is authentic")
    else:
        print(f"✗ Verification failed: {result.status.value}")

With options:

.. code-block:: python

    from asala.types import VerificationOptions

    options = VerificationOptions(
        include_metadata=True,
        include_chain_analysis=True,
        trust_store=['trusted-key-1', 'trusted-key-2']
    )
    
    result = asala.verify(content, manifest, options)

Analyzing results:

.. code-block:: python

    result = asala.verify(content, manifest)
    
    print(f"Status: {result.status.value}")
    print(f"Confidence: {result.confidence}%")
    
    # Check individual layers
    for layer in result.layers:
        status = "✓" if layer.passed else "✗"
        print(f"{status} {layer.name}: {layer.score:.0f}%")
        
        # Layer-specific details
        for key, value in layer.details.items():
            print(f"    {key}: {value}")
    
    # Warnings and errors
    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  ⚠ {warning}")
    
    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  ✗ {error}")

Physics-Based Verification
^^^^^^^^^^^^^^^^^^^^^^^^^^

Image verification:

.. code-block:: python

    from asala.physics import PhysicsVerifier

    verifier = PhysicsVerifier()

    with open('photo.jpg', 'rb') as f:
        image_bytes = f.read()

    result = verifier.verify_image(image_bytes)

    print(f"Score: {result.score}")
    print(f"Passed: {result.passed}")
    print(f"AI probability: {result.details['ai_probability']}")
    print(f"AI indicators: {result.details['ai_indicators']}/16")

Audio verification:

.. code-block:: python

    from asala.audio import AudioVerifier

    verifier = AudioVerifier()

    with open('recording.wav', 'rb') as f:
        audio_bytes = f.read()

    result = verifier.verify_audio(audio_bytes)

    print(f"Score: {result.score}")
    print(f"AI indicators: {result.details['ai_indicators']}/10")

Video verification:

.. code-block:: python

    from asala.video import VideoVerifier

    verifier = VideoVerifier()

    with open('clip.mp4', 'rb') as f:
        video_bytes = f.read()

    result = verifier.verify_video(video_bytes)

    print(f"Score: {result.score}")
    print(f"AI indicators: {result.details['ai_indicators']}/16")

Parallel Processing
^^^^^^^^^^^^^^^^^^^

All verifiers accept a ``max_workers`` parameter to run analysis methods
concurrently using ``ThreadPoolExecutor``. This is off by default
(``max_workers=1``). Set it to a higher value to enable parallelism:

.. code-block:: python

    from asala import Asala

    # Enable parallel verification with 4 threads
    asala = Asala(max_workers=4)
    result = asala.verify(content)

You can also set ``max_workers`` on individual verifiers:

.. code-block:: python

    from asala.physics import PhysicsVerifier
    from asala.audio import AudioVerifier
    from asala.video import VideoVerifier

    # Parallel image analysis (16 methods across 4 threads)
    physics = PhysicsVerifier(max_workers=4)
    result = physics.verify_image(image_bytes)

    # Parallel audio analysis (10 methods across 4 threads)
    audio = AudioVerifier(max_workers=4)
    result = audio.verify_audio(audio_bytes)

    # Parallel video analysis (6 methods + per-frame across 4 threads)
    video = VideoVerifier(max_workers=4)
    result = video.verify_video(video_bytes)

Thread safety: all analysis methods only read from pre-computed shared arrays
(``img``, ``gray``, ``lab``, ``samples``, ``stft_*``). numpy, scipy, and
OpenCV release the GIL during C-level operations, so threads achieve real
parallelism without the serialization overhead of multiprocessing.

Cryptographic Utilities
^^^^^^^^^^^^^^^^^^^^^^^

Direct access to crypto functions:

.. code-block:: python

    from asala import CryptoUtils

    # Hash content
    content_hash = CryptoUtils.hash_content(b"content")
    
    # Hash string
    string_hash = CryptoUtils.hash_string("text")
    
    # Sign and verify
    signature = CryptoUtils.sign_content("message", private_key)
    is_valid = CryptoUtils.verify_signature("message", signature, public_key)
    
    # Verify chain integrity
    is_valid = CryptoUtils.verify_chain_integrity(chain_links, original_hash)

Advanced Examples
-----------------

Custom Verification Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from asala import Asala
    from asala.types import VerificationStatus

    class CustomVerifier:
        def __init__(self, trust_store=None):
            self.asala = Asala()
            self.trust_store = trust_store or []
        
        def verify_batch(self, files):
            results = []
            for file_path in files:
                content = Path(file_path).read_bytes()
                
                # Try to load manifest
                manifest_path = f"{file_path}.manifest.json"
                manifest = None
                if Path(manifest_path).exists():
                    manifest = self._load_manifest(manifest_path)
                
                result = self.asala.verify(
                    content,
                    manifest,
                    trust_store=self.trust_store
                )
                
                results.append({
                    'file': file_path,
                    'result': result
                })
            
            return results
        
        def generate_report(self, results):
            verified = sum(1 for r in results 
                         if r['result'].status == VerificationStatus.VERIFIED)
            
            report = f"""
            Verification Report
            ===================
            Total files: {len(results)}
            Verified: {verified}
            Failed: {len(results) - verified}
            
            Details:
            """
            
            for item in results:
                icon = "✓" if item['result'].status == VerificationStatus.VERIFIED else "✗"
                report += f"\n{icon} {item['file']}: {item['result'].status.value}"
            
            return report

Integration with Web Frameworks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

FastAPI:

.. code-block:: python

    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import JSONResponse
    from asala import Asala

    app = FastAPI()
    asala = Asala()

    @app.post("/verify")
    async def verify_file(file: UploadFile = File(...)):
        content = await file.read()
        result = asala.verify(content)
        
        return {
            "filename": file.filename,
            "status": result.status.value,
            "confidence": result.confidence,
            "verified": result.status.value == 'verified'
        }

Flask:

.. code-block:: python

    from flask import Flask, request, jsonify
    from asala import Asala

    app = Flask(__name__)
    asala = Asala()

    @app.route('/verify', methods=['POST'])
    def verify():
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        content = file.read()
        result = asala.verify(content)
        
        return jsonify({
            'status': result.status.value,
            'confidence': result.confidence
        })

Django:

.. code-block:: python

    from django.http import JsonResponse
    from django.views import View
    from asala import Asala

    class VerifyView(View):
        def post(self, request):
            asala = Asala()
            
            uploaded_file = request.FILES['file']
            content = uploaded_file.read()
            result = asala.verify(content)
            
            return JsonResponse({
                'status': result.status.value,
                'confidence': result.confidence
            })

Performance Considerations
--------------------------

- **Key Generation**: One-time operation, ~100ms
- **Signing**: <10ms for typical content
- **Verification**: <5ms per signature
- **Memory Usage**: Minimal, proportional to content size
- **Parallel Processing**: Set ``max_workers > 1`` to run analysis methods
  concurrently. Uses ``ThreadPoolExecutor`` — numpy/scipy/OpenCV release the
  GIL so threads achieve real parallelism without process serialization overhead

Best Practices
--------------

1. **Keep private keys secure**: Never commit to version control
2. **Backup keys**: Store encrypted backups offline
3. **Use trust stores**: Verify against known good keys
4. **Chain edits**: Always sign transformations
5. **Regular verification**: Verify content periodically
