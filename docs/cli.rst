CLI Documentation
=================

Asala provides a command-line interface for content verification and signing.

Installation
------------

**Using npm:**

.. code-block:: bash

    npm install -g @asala/cli

**Using pip:**

.. code-block:: bash

    pip install asala

Commands
--------

verify
^^^^^^

Verify content authenticity.

.. code-block:: bash

    asala verify <file> [options]

**Options:**

- ``-m, --manifest <path>`` - Path to manifest file
- ``-t, --trust <keys...>`` - Trusted public keys
- ``-j, --json`` - Output as JSON
- ``-v, --verbose`` - Verbose output

**Examples:**

Basic verification:

.. code-block:: bash

    asala verify ./photo.jpg

With manifest:

.. code-block:: bash

    asala verify ./photo.jpg --manifest ./photo.jpg.manifest.json

Output as JSON:

.. code-block:: bash

    asala verify ./photo.jpg --json

With trust store:

.. code-block:: bash

    asala verify ./photo.jpg --trust ./trusted-keys/*.pem

sign
^^^^

Sign content with provenance data.

.. code-block:: bash

    asala sign <file> [options]

**Options:**

- ``-k, --key <path>`` - Path to private key (required)
- ``-o, --output <path>`` - Output file path
- ``-c, --creator <name>`` - Creator name
- ``-d, --device <device>`` - Device name

**Examples:**

Basic signing:

.. code-block:: bash

    asala sign ./photo.jpg --key ./private.pem

With creator info:

.. code-block:: bash

    asala sign ./photo.jpg \
        --key ./private.pem \
        --creator "John Doe" \
        --device "Canon EOS R5"

Custom output:

.. code-block:: bash

    asala sign ./photo.jpg \
        --key ./private.pem \
        --output ./signed-photo.manifest.json

keys
^^^^

Manage cryptographic keys.

.. code-block:: bash

    asala keys [options]

**Options:**

- ``-g, --generate`` - Generate new key pair
- ``-o, --output <dir>`` - Output directory (default: ./keys)

**Examples:**

Generate keys:

.. code-block:: bash

    asala keys --generate

Custom output directory:

.. code-block:: bash

    asala keys --generate --output ./my-keys

manifest
^^^^^^^^

View or extract manifest from content.

.. code-block:: bash

    asala manifest <file> [options]

**Options:**

- ``-e, --extract`` - Extract manifest to file
- ``-o, --output <path>`` - Output path for extracted manifest

**Examples:**

View manifest:

.. code-block:: bash

    asala manifest ./photo.jpg

Extract manifest:

.. code-block:: bash

    asala manifest ./photo.jpg --extract --output ./manifest.json

Output Formats
--------------

Standard Output
^^^^^^^^^^^^^^^

.. code-block::

    ============================================================
      Content Verification Report
    ============================================================

    File: /path/to/photo.jpg
    Status: VERIFIED
    Confidence: 95%

    Verification Layers:
      ✓ Signature Verification: 100%
      ✓ Chain Integrity: 100%

    Provenance Data:
      ID: urn:uuid:abc123...
      Creator: John Doe
      Created: 12/15/2023, 2:30:45 PM
      Signatures: 1

    ============================================================

JSON Output
^^^^^^^^^^^

.. code-block:: json

    {
      "status": "verified",
      "confidence": 95,
      "warnings": [],
      "errors": [],
      "layers": [
        {
          "name": "Signature Verification",
          "passed": true,
          "score": 100
        },
        {
          "name": "Chain Integrity",
          "passed": true,
          "score": 100
        }
      ]
    }

Exit Codes
----------

- ``0`` - Success / Verified
- ``1`` - Verification failed / Tampered / Error
- ``2`` - Invalid arguments

Shell Integration
-----------------

Bash/Zsh Completion
^^^^^^^^^^^^^^^^^^^

Add to your ``.bashrc`` or ``.zshrc``:

.. code-block:: bash

    eval "$(asala completion bash)"  # For bash
    eval "$(asala completion zsh)"   # For zsh

Alias
^^^^^

.. code-block:: bash

    alias uv='asala verify'
    alias us='asala sign'

Batch Processing
^^^^^^^^^^^^^^^^

Verify multiple files:

.. code-block:: bash

    for file in *.jpg; do
        echo "Verifying: $file"
        asala verify "$file" --json | jq '.status'
    done

CI/CD Integration
-----------------

GitHub Actions
^^^^^^^^^^^^^^

.. code-block:: yaml

    - name: Verify Content
      run: |
        npm install -g @asala/cli
        asala verify ./assets/photo.jpg

GitLab CI
^^^^^^^^^

.. code-block:: yaml

    verify:
      script:
        - pip install asala
        - asala verify ./assets/photo.jpg

Docker
------

Run in Docker:

.. code-block:: bash

    docker run -v $(pwd):/content asala/cli verify /content/photo.jpg

Build image:

.. code-block:: dockerfile

    FROM node:18-alpine
    RUN npm install -g @asala/cli
    ENTRYPOINT ["asala"]
