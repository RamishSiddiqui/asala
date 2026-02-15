Quick Start
===========

This guide will help you get started with Asala in minutes.

Installation
------------

Choose your preferred language:

Python
^^^^^^

.. code-block:: bash

    pip install asala

Node.js
^^^^^^^

.. code-block:: bash

    npm install @asala/core

Verify Content
--------------

Python
^^^^^^

.. code-block:: python

    from asala import Asala

    asala = Asala()
    
    # Load content
    with open('image.jpg', 'rb') as f:
        content = f.read()
    
    # Verify
    result = asala.verify(content)
    
    if result.status.value == 'verified':
        print(f"✓ Content is authentic! Confidence: {result.confidence}%")
    else:
        print(f"⚠ Could not verify content")

Node.js
^^^^^^^

.. code-block:: javascript

    const { Asala } = require('@asala/core');
    const fs = require('fs');

    const asala = new Asala();
    const content = fs.readFileSync('image.jpg');

    asala.verify(content).then(result => {
        if (result.status === 'verified') {
            console.log(`✓ Authentic! Confidence: ${result.confidence}%`);
        }
    });

Sign Content
------------

To sign content and create provenance:

Python
^^^^^^

.. code-block:: python

    from asala import Asala

    asala = Asala()
    
    # Generate keys (do this once and keep private key secure)
    public_key, private_key = asala.generate_key_pair()
    
    # Save keys
    with open('private.pem', 'w') as f:
        f.write(private_key)
    with open('public.pem', 'w') as f:
        f.write(public_key)
    
    # Sign content
    with open('photo.jpg', 'rb') as f:
        content = f.read()
    
    manifest = asala.sign_content(
        content, 
        private_key, 
        creator="Your Name"
    )
    
    # Save manifest
    import json
    with open('photo.jpg.manifest.json', 'w') as f:
        json.dump({
            'id': manifest.id,
            'content_hash': manifest.content_hash,
            'content_type': manifest.content_type.value,
            'created_at': manifest.created_at,
            'created_by': manifest.created_by,
            'signatures': [
                {
                    'algorithm': sig.algorithm,
                    'public_key': sig.public_key,
                    'signature': sig.signature,
                    'timestamp': sig.timestamp,
                    'signer': sig.signer
                }
                for sig in manifest.signatures
            ]
        }, f, indent=2)

CLI Usage
---------

Verify a file:

.. code-block:: bash

    asala verify ./photo.jpg

Sign a file:

.. code-block:: bash

    # Generate keys first
    asala keys --generate
    
    # Sign content
    asala sign ./photo.jpg --key ./keys/private.pem --creator "Your Name"

Browser Extension
-----------------

Install the browser extension to automatically verify content as you browse:

1. Download from Chrome Web Store or Firefox Add-ons
2. The extension will automatically scan images and videos
3. Green badges indicate verified content
4. Click badges for detailed provenance information

Next Steps
----------

- Learn about :doc:`architecture` to understand how it works
- Explore the :doc:`api` for detailed API documentation
- Read :doc:`contributing` to contribute to the project
