API Reference
=============

Python API
----------

Asala
^^^^^^^^^^^^

.. autoclass:: asala.Asala
   :members:
   :undoc-members:
   :show-inheritance:

CryptoUtils
^^^^^^^^^^^

.. autoclass:: asala.CryptoUtils
   :members:
   :undoc-members:
   :show-inheritance:

ManifestBuilder
^^^^^^^^^^^^^^^

.. autoclass:: asala.manifest.ManifestBuilder
   :members:
   :undoc-members:
   :show-inheritance:

PhysicsVerifier
^^^^^^^^^^^^^^^

.. autoclass:: asala.physics.PhysicsVerifier
   :members:
   :undoc-members:
   :show-inheritance:

AudioVerifier
^^^^^^^^^^^^^

.. autoclass:: asala.audio.AudioVerifier
   :members:
   :undoc-members:
   :show-inheritance:

VideoVerifier
^^^^^^^^^^^^^

.. autoclass:: asala.video.VideoVerifier
   :members:
   :undoc-members:
   :show-inheritance:

Types
-----

ContentManifest
^^^^^^^^^^^^^^^

.. autoclass:: asala.types.ContentManifest
   :members:
   :undoc-members:

SignatureInfo
^^^^^^^^^^^^^

.. autoclass:: asala.types.SignatureInfo
   :members:
   :undoc-members:

VerificationResult
^^^^^^^^^^^^^^^^^^

.. autoclass:: asala.types.VerificationResult
   :members:
   :undoc-members:

VerificationStatus
^^^^^^^^^^^^^^^^^^

.. autoclass:: asala.types.VerificationStatus
   :members:
   :undoc-members:

ContentType
^^^^^^^^^^^

.. autoclass:: asala.types.ContentType
   :members:
   :undoc-members:

LayerResult
^^^^^^^^^^^

.. autoclass:: asala.types.LayerResult
   :members:
   :undoc-members:

Node.js API
-----------

Asala (TypeScript/JavaScript)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: typescript

    class Asala {
        verify(
            content: Buffer,
            manifest?: ContentManifest,
            options?: VerificationOptions
        ): Promise<VerificationResult>;
        
        signContent(
            content: Buffer,
            privateKey: string,
            creator: string,
            contentType?: ContentType
        ): ContentManifest;
        
        generateKeyPair(): { publicKey: string; privateKey: string };
    }

Types
^^^^^

.. code-block:: typescript

    interface VerificationResult {
        status: VerificationStatus;
        confidence: number;
        manifest?: ContentManifest;
        warnings: string[];
        errors: string[];
        layers: LayerResult[];
    }
    
    interface ContentManifest {
        id: string;
        contentHash: string;
        contentType: ContentType;
        createdAt: string;
        createdBy: string;
        signatures: SignatureInfo[];
        assertions: Assertion[];
        chain: ChainLink[];
    }
    
    enum VerificationStatus {
        VERIFIED = 'verified',
        UNVERIFIED = 'unverified',
        TAMPERED = 'tampered',
        INVALID_SIGNATURE = 'invalid_signature',
        MISSING_PROVENANCE = 'missing_provenance'
    }

Examples
--------

Basic Verification
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from asala import Asala

    asala = Asala()
    
    with open('image.jpg', 'rb') as f:
        content = f.read()
    
    result = asala.verify(content)
    
    print(f"Status: {result.status.value}")
    print(f"Confidence: {result.confidence}%")

Signing with Chain
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from asala import Asala
    from asala.manifest import ManifestBuilder

    asala = Asala()
    
    # Generate keys
    public_key, private_key = asala.generate_key_pair()
    
    # Create and sign
    with open('photo.jpg', 'rb') as f:
        content = f.read()
    
    manifest = asala.sign_content(content, private_key, "Photographer")
    
    # Add edit chain
    manifest_builder = ManifestBuilder(
        manifest.content_hash,
        manifest.content_type,
        manifest.created_by
    )
    
    # Add resize operation
    manifest_builder.add_chain_link("resize", "Editor", private_key)
    
    # Verify
    result = asala.verify(content, manifest)

Batch Verification
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pathlib import Path
    from asala import Asala

    asala = Asala()
    
    results = []
    for image_path in Path('./photos').glob('*.jpg'):
        content = image_path.read_bytes()
        result = asala.verify(content)
        results.append({
            'file': image_path.name,
            'status': result.status.value,
            'confidence': result.confidence
        })
    
    # Summary
    verified = sum(1 for r in results if r['status'] == 'verified')
    print(f"Verified: {verified}/{len(results)}")

Error Handling
^^^^^^^^^^^^^^

.. code-block:: python

    from asala import Asala
    from asala.types import VerificationStatus

    asala = Asala()
    
    result = asala.verify(content, manifest)
    
    if result.status == VerificationStatus.TAMPERED:
        print("⚠ Content has been tampered with!")
        for error in result.errors:
            print(f"  Error: {error}")
    
    elif result.status == VerificationStatus.INVALID_SIGNATURE:
        print("⚠ Invalid signature detected!")
    
    elif result.status == VerificationStatus.MISSING_PROVENANCE:
        print("ℹ No provenance data available")
        print("This content was not signed at creation")
