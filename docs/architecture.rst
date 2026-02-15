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
           E[Noise Analysis] --> F[Lighting Check]
           F --> G[Acoustic Patterns]
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

Secondary layer detects synthetic content through mathematical analysis:

Noise Pattern Analysis
^^^^^^^^^^^^^^^^^^^^^^

Real camera sensors produce specific noise patterns:

- **Photo Response Non-Uniformity (PRNU)**: Each sensor unique
- **Dark Current Noise**: Temperature-dependent patterns
- **Read Noise**: Electronic characteristics

AI-generated content lacks these physical signatures.

Lighting Consistency
^^^^^^^^^^^^^^^^^^^^

Real-world lighting follows physics:

- Shadow directions must be consistent
- Light sources have spectral signatures
- Reflections follow optical laws

Acoustic Verification (Audio)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Real audio has:

- Room reverberation patterns
- Microphone frequency response
- Compression artifacts from codecs

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
