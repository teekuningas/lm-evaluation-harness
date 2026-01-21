# Use a lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (to keep image small)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Set working directory
WORKDIR /app

# Copy the evaluation harness code
COPY . /app/lm-evaluation-harness

# Install the harness
WORKDIR /app/lm-evaluation-harness

# Install the package and dependencies
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir requests aiohttp tenacity tqdm tiktoken

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
