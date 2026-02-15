# Multi-stage Dockerfile for Asala

# Stage 1: Build Node.js packages
FROM node:20-alpine AS node-builder

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY core/package*.json ./core/
COPY cli/package*.json ./cli/
COPY extension/package*.json ./extension/
COPY web/package*.json ./web/

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build all packages
RUN npm run build

# Stage 2: Build Python package
FROM python:3.11-slim AS python-builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Python package
COPY pyproject.toml ./
COPY python/ ./python/

# Install Python package
RUN pip install --no-cache-dir build
RUN python -m build

# Stage 3: Final runtime image
FROM node:20-slim

WORKDIR /app

# Install Python for mixed usage
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy built Node.js packages
COPY --from=node-builder /app/core/dist ./core/dist
COPY --from=node-builder /app/cli/dist ./cli/dist
COPY --from=node-builder /app/extension/dist ./extension/dist
COPY --from=node-builder /app/web/dist ./web/dist
COPY --from=node-builder /app/node_modules ./node_modules
COPY --from=node-builder /app/package*.json ./

# Copy built Python package
COPY --from=python-builder /app/dist/*.whl /tmp/
RUN pip3 install /tmp/*.whl && rm /tmp/*.whl

# Create non-root user
RUN groupadd -r asala && useradd -r -g asala asala
RUN chown -R asala:asala /app
USER asala

# Expose port for web interface
EXPOSE 3000

# Default command
ENTRYPOINT ["asala"]
CMD ["--help"]
