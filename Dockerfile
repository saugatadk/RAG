FROM python:3.12-slim

# Set environment variables for Python and Hugging Face
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface

# Install system dependencies (build-essential, libgomp for faiss, curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set up user with UID 1000 required by Hugging Face Spaces
RUN useradd -m -u 1000 user

WORKDIR $HOME/app

# Copy dependency requirements first to leverage Docker layer caching
COPY --chown=user:user requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache the sentence transformer model so cold-starts are instant
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code and starter data
COPY --chown=user:user . .

# Ensure upload and vectorstore directories exist and belong to user
RUN mkdir -p data/uploads data/faiss_store && \
    chown -R user:user $HOME/app

# Switch to non-root user
USER user

# Hugging Face Spaces expects traffic on port 7860
EXPOSE 7860

# Start FastAPI + Gradio server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
