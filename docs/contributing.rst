Contributing
============

Thank you for your interest in contributing to Asala!

Development Setup
-----------------

Clone the Repository
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    git clone https://github.com/your-org/asala.git
    cd asala

Install Dependencies
^^^^^^^^^^^^^^^^^^^^

**Node.js packages:**

.. code-block:: bash

    npm install

**Python packages:**

.. code-block:: bash

    pip install -e ".[dev]"

Build
^^^^^

.. code-block:: bash

    npm run build

Testing
-------

Run all tests:

.. code-block:: bash

    npm test

Run Python tests:

.. code-block:: bash

    pytest python/tests

Run with coverage:

.. code-block:: bash

    pytest --cov=asala --cov-report=html

Code Style
----------

TypeScript/JavaScript
^^^^^^^^^^^^^^^^^^^^^

We use ESLint and Prettier:

.. code-block:: bash

    npm run lint
    npm run lint:fix
    npm run format

Python
^^^^^^

We use Black, isort, flake8, mypy, and bandit:

.. code-block:: bash

    black python/                 # Format code
    isort python/                 # Sort imports
    flake8 python/asala           # Lint
    mypy python/asala             # Type check
    bandit -r python/asala        # Security scan

Project Structure
-----------------

.. code-block::

    asala/
    ├── core/              # TypeScript core library
    │   ├── src/
    │   │   ├── __tests__/    # Jest tests
    │   │   ├── crypto/       # Cryptographic functions + ELA
    │   │   ├── imaging/      # Pure-JS image processing (FFT, DCT, convolution)
    │   │   ├── types/        # Type definitions
    │   │   └── verifiers/    # Physics, audio, video verification
    │   └── package.json
    ├── python/            # Python implementation
    │   ├── asala/            # Package source (verify, crypto, physics, audio, video)
    │   └── tests/            # pytest suite (108 tests)
    ├── cli/               # Node.js CLI
    ├── extension/         # Browser extension
    ├── web/               # Web interface
    ├── docs/              # Sphinx documentation
    └── examples/          # Usage examples

Making Changes
--------------

1. **Create a branch**

   .. code-block:: bash

       git checkout -b feature/your-feature-name

2. **Make your changes**

   - Write tests first (TDD)
   - Keep changes focused
   - Follow existing code style

3. **Test your changes**

   .. code-block:: bash

       npm test
       pytest python/tests

4. **Update documentation**

   - Update docstrings
   - Update relevant .rst files in docs/
   - Add examples if applicable

5. **Commit**

   .. code-block:: bash

       git add .
       git commit -m "feat: add new verification method"

   Follow conventional commits:
   
   - ``feat:`` - New feature
   - ``fix:`` - Bug fix
   - ``docs:`` - Documentation
   - ``test:`` - Tests
   - ``refactor:`` - Code refactoring

6. **Push and create PR**

   .. code-block:: bash

       git push origin feature/your-feature-name

Pull Request Guidelines
-----------------------

Before submitting:

- [ ] Tests pass
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
- [ ] PR description explains changes

Areas to Contribute
-------------------

High Priority
^^^^^^^^^^^^^

- C2PA manifest embedding in images
- Physics-based verification algorithms
- Browser extension improvements
- Mobile app development
- ~~Performance optimizations~~ (done: parallel processing via ``max_workers``)

Documentation
^^^^^^^^^^^^^

- API documentation improvements
- Tutorial videos
- Translation to other languages
- Integration guides

Testing
^^^^^^^

- Unit tests for edge cases
- Integration tests
- Browser compatibility testing
- Performance benchmarks

Reporting Issues
----------------

When reporting bugs:

1. Check if issue already exists
2. Provide minimal reproduction
3. Include environment details
4. Add error messages/logs

Security Issues
^^^^^^^^^^^^^^^

For security vulnerabilities:

- Email: security@asala.org
- Do not open public issues
- Include detailed description
- Allow time for fix before disclosure

Getting Help
------------

- **Discord**: [Link]
- **GitHub Discussions**: [Link]
- **Documentation**: Read the docs!

License
-------

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for making the internet more trustworthy!
