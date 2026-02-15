.PHONY: help install build test clean binary docs publish

help: ## Show this help message
	@echo "Asala - Build Commands"
	@echo "==============================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	npm install
	pip install -e ".[dev]"

build: ## Build all packages
	npm run build

test: ## Run all tests
	npm run test
	pytest python/tests -v

test-core: ## Run core tests only
	npm run test:core

test-python: ## Run Python tests only
	pytest python/tests -v

lint: ## Run all linters
	npm run lint
	flake8 python/asala
	black --check python/

lint-fix: ## Fix linting issues
	npm run lint:fix
	black python/

format: ## Format all code
	npm run format
	black python/

clean: ## Clean build artifacts
	npm run clean
	rm -rf bin/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .pytest_cache/

binary: ## Build all binaries (requires pkg and pyinstaller)
	@echo "Building Node.js binaries..."
	npm run binary:all
	@echo "Building Python binary..."
	pyinstaller --onefile --name asala-python --distpath bin python/asala/cli.py

docs: ## Build documentation
	cd docs && make html

docs-serve: ## Serve documentation locally
	cd docs && make livehtml

publish-npm: ## Publish to npm (requires auth)
	npm publish --workspaces --access public

publish-pypi: ## Publish to PyPI (requires auth)
	python -m build
	twine upload dist/*

publish: publish-npm publish-pypi ## Publish to all registries

all: clean install build test binary docs ## Run full build pipeline
