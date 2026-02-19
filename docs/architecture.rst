Architecture
============

Asala uses a multi-layered approach to content verification that combines
cryptographic provenance with physics-based analysis.

.. mermaid::

   flowchart TB
       subgraph Layer1["Layer 1: Cryptographic Provenance"]
           A[Content Creation] --> B[Sign with Private Key]
           B --> C[Chain of Custody]
           C --> D[Verify with Public Key]
       end
       
       subgraph Layer2["Layer 2: Physics-Based Verification"]
           E[Image Analysis<br/>16 methods] --> F[Audio Analysis<br/>10 methods]
           F --> G[Video Analysis<br/>6 methods]
       end
       
       subgraph Layer3["Layer 3: Distributed Consensus"]
           H[Multi-party Verification] --> I[Reputation Scoring]
       end
       
       D --> J{All Passed?}
       G --> J
       I --> J
       
       J -->|Yes| K[Verified]
       J -->|No| L[Failed]

Core Philosophy
---------------

Instead of trying to detect fakes (which AI models will always improve at), we
**mathematically prove authenticity** using:

1. **Cryptographic signatures** at content creation
2. **Immutable chain of custody** for any edits
3. **Mathematical verification** that is inherently unbreakable

Why This Works
--------------

AI Detection (Current Approach)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Train models to recognize fake patterns
- Problem: Models always improve, detection must keep up
- Requires constant retraining and cloud resources
- Statistical approach, never 100% certain

Cryptographic Provenance (Our Approach)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Sign content with private keys at creation
- Verify with public keys mathematically
- Advantages:
  - **Unbreakable**: Based on prime factorization (math, not ML)
  - **No training**: Verification is instant and deterministic
  - **Local**: Can run entirely on user's device
  - **Certain**: Either authentic or not, no gray area

Layer 1: Cryptographic Provenance
---------------------------------

The foundation of Asala is cryptographic signatures using RSA-2048.

Content Signing
^^^^^^^^^^^^^^^

When content is created:

1. Content is hashed using SHA-256
2. Hash is signed with creator's private key
3. Signature and metadata embedded in C2PA manifest
4. Manifest travels with content

.. code-block:: python

    from asala import Asala
    
    asala = Asala()
    
    # Content creator signs at creation
    manifest = asala.sign_content(
        content=photo_bytes,
        private_key=creator_private_key,
        creator="John Doe"
    )

Chain of Custody
^^^^^^^^^^^^^^^^

Any edits or transformations must be signed:

1. Original content hash preserved
2. Each transformation signed by editor
3. Chain links reference previous state
4. Breaks if any link is tampered

.. code-block:: python

    from asala.manifest import ManifestBuilder
    
    # Add edit to chain
    manifest.add_chain_link(
        action="resize",
        actor="Editor Name",
        private_key=editor_private_key
    )

Verification Process
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # User verifies
    result = asala.verify(content, manifest)
    
    # Check layers
    for layer in result.layers:
        print(f"{layer.name}: {layer.score}%")

Layer 2: Physics-Based Verification
-----------------------------------

Secondary layer detects synthetic content through mathematical analysis of
physical properties. Fully implemented for images, audio, and video.

All analysis methods within each verifier are independent and can run in
parallel via the ``max_workers`` parameter (default ``1`` = sequential).
When ``max_workers > 1``, a ``ThreadPoolExecutor`` dispatches all methods
concurrently.

Image Analysis (16 Methods)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Phase 1 — Core Analysis (8 methods):**

- **Noise uniformity**: Laplacian and median-residual coefficient of variation across 4x4 grid
- **Noise frequency**: 2D FFT spectral band energy ratios (low/mid/high)
- **Frequency domain**: 2D DCT entropy and DC/AC separation
- **Geometric consistency**: Canny edge detection, line counting, angle histogram, corner density
- **Lighting analysis**: LAB L-channel grid brightness/contrast and gradient direction consistency
- **Texture patterns**: Sobel gradient magnitude distribution and regional variation
- **Color distribution**: HSV histogram entropy and regional color consistency
- **Compression artifacts**: Multi-quality ELA (95/85/75/65) with regional 8x8 grid analysis

**Phase 2 — Advanced Detection (4 methods):**

- **Noise consistency map**: Sliding-window noise estimation for splice detection
- **JPEG ghost detection**: Multi-quality recompression to find pasted regions
- **Spectral fingerprinting**: 2D power spectrum analysis for GAN artifacts
- **Cross-channel correlation**: R/G/B noise correlation for camera sensor vs GAN discrimination

**Phase 3 — Forensic Methods (4 methods):**

- **Benford's law on DCT**: First-digit distribution of DCT coefficients
- **Wavelet detail ratio**: High-frequency wavelet coefficient analysis
- **Blocking artifact grid**: 8x8 JPEG block boundary regularity detection
- **CFA demosaicing**: Bayer pattern periodicity in noise residuals

Audio Analysis (10 Methods)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Phase coherence**: Inter-frame phase stability in STFT
- **Voice quality**: Jitter, shimmer, and harmonics-to-noise ratio
- **ENF analysis**: Electrical network frequency (50/60 Hz hum) detection
- **Spectral tilt**: Power-law decay slope of the frequency spectrum
- **Noise consistency**: Background noise floor stability across segments
- **Mel regularity**: Mel-spectrogram temporal regularity
- **Formant bandwidth**: Vocal tract resonance bandwidth naturalness
- **Double compression**: Re-encoding artifact detection via DCT coefficient analysis
- **Spectral discontinuity**: Splice detection via spectral consistency at segment boundaries
- **Sub-band energy**: Energy distribution across frequency sub-bands

Video Analysis (6 Methods)
^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Per-frame analysis**: Aggregate image-level physics scores across sampled frames
- **Temporal noise**: Noise consistency between consecutive frames
- **Optical flow**: Block-matching motion estimation for natural motion patterns
- **Encoding analysis**: Codec artifact regularity and quantization patterns
- **Temporal lighting**: Brightness and contrast stability over time
- **Frame stability**: Normalized cross-correlation between consecutive frames

Layer 3: Distributed Consensus
-------------------------------

Optional third layer for high-value content:

Multi-Party Verification
^^^^^^^^^^^^^^^^^^^^^^^^

- Multiple independent verifiers check content
- Consensus required for verification
- Reduces single-point-of-failure

Reputation System
^^^^^^^^^^^^^^^^^

- Verifiers build reputation over time
- Stake-weighted voting
- Economic incentives for honesty

Data Flow
---------

.. mermaid::

   sequenceDiagram
       participant Creator
       participant Content
       participant Manifest
       participant User
       
       Creator->>Content: Create content
       Creator->>Manifest: Sign with private key
       Manifest->>Content: Embed manifest
       Content->>User: Share content
       User->>Manifest: Extract manifest
       Manifest->>User: Verify signatures
       User->>User: Display result

Security Properties
-------------------

Confidentiality
^^^^^^^^^^^^^^^

- Private keys never leave creator's device
- Verification uses only public keys
- No cloud processing required

Integrity
^^^^^^^^^

- SHA-256 hashing (collision-resistant)
- RSA-2048 signatures (computationally infeasible to forge)
- Chain of custody detects any tampering

Non-Repudiation
^^^^^^^^^^^^^^^

- Only private key holder can create valid signatures
- Cryptographic proof of authorship
- Immutable audit trail

Why AI Can't Beat This
----------------------

**The fundamental difference:**

- **AI Detection**: Pattern recognition problem
  - Can always train better models
  - Adversarial examples exist
  - Statistical confidence, not certainty

- **Cryptography**: Mathematical proof
  - Breaking RSA requires factoring large primes
  - No AI model can factor efficiently
  - Binary: valid or invalid

**Even a perfect deepfake cannot forge a valid signature.**

Performance
-----------

- **Signing**: < 10ms on modern hardware
- **Verification**: < 5ms per signature
- **Memory**: Minimal, scales with content size
- **No network required**: Can run offline
- **Parallel processing** (Python): All analysis methods within each verifier
  can run concurrently via ``ThreadPoolExecutor``. Set ``max_workers > 1`` to
  enable. Uses threads (not processes) because numpy, scipy, and OpenCV release
  the GIL during C-level computation, providing real parallelism without
  serialization overhead.

Standards Compliance
--------------------

Asala implements:

- **C2PA**: Coalition for Content Provenance and Authenticity
- **W3C Verifiable Credentials**: For identity assertions
- **ISO/IEC 13818**: For MPEG-based content

See Also
--------

- :doc:`api` - API reference
- :doc:`cli` - Command-line usage
- `C2PA Specification <https://c2pa.org/specifications/>`_
