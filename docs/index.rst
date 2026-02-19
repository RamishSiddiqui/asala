Asala Documentation
===========================

**Cryptographic Content Authenticity Verification**

Asala is an open-source solution for verifying digital content authenticity 
using cryptographic provenance. Unlike AI-based detection systems, we use mathematical 
proofs that are inherently unbreakable.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   architecture
   api
   cli
   python
   contributing

Overview
--------

The internet is filled with fake images, videos, and documents. Traditional solutions
try to detect fakes using AI, but this is a losing battle as generative models improve.

Asala takes a different approach: **don't detect fakes, prove authenticity**.

Key Features
------------

- **Cryptographic Provenance**: Content signed at creation using RSA-2048 private keys
- **Chain of Custody**: Track all edits and transformations with signed chain links
- **Physics-Based Verification**: 16 image, 10 audio, and 6 video analysis methods with optional parallel processing
- **Multi-Language Support**: TypeScript/JavaScript and Python implementations with feature parity
- **Browser Extension**: Real-time verification on any website
- **Command-Line Tools**: Verify and sign content from terminal
- **Open Source**: Fully transparent and auditable

Quick Example
-------------

**Sign content (Python):**

.. code-block:: python

    from asala import Asala

    asala = Asala()
    
    # Generate keys
    public_key, private_key = asala.generate_key_pair()
    
    # Sign content
    with open('photo.jpg', 'rb') as f:
        content = f.read()
    
    manifest = asala.sign_content(content, private_key, "John Doe")

**Verify content:**

.. code-block:: python

    result = asala.verify(content, manifest)
    
    print(f"Status: {result.status.value}")
    print(f"Confidence: {result.confidence}%")

Installation
------------

**Python:**

.. code-block:: bash

    pip install asala

**Node.js:**

.. code-block:: bash

    npm install @asala/core

**CLI:**

.. code-block:: bash

    npm install -g @asala/cli
    # or
    pip install asala

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
