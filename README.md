---
title: Document RAG Assistant
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# 📚 Document RAG Assistant

A complete Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **Sentence-Transformers**, **FAISS Vector Search**, **Gradio**, and **OpenRouter LLMs**.

Designed for zero-cost deployment on **Hugging Face Spaces** (Free Tier: 2 vCPU + 16 GB RAM).

---

## 🌟 Features

- **Multi-Format Ingestion:** Supports `.pdf`, `.txt`, `.csv`, `.xlsx`, `.docx`, and `.json`.
- **Dual Interface:**
  - **Interactive Web UI (`/`):** Test questions, view retrieved citations with page numbers, and upload documents with one click.
  - **REST API (`/docs`, `/query`, `/upload`, `/health`):** OpenAPI / Swagger documentation ready for integration.
- **Fast Similarity Search:** Cosine similarity via FAISS index flat inner-product on normalized `all-MiniLM-L6-v2` embeddings.
- **LLM Reasoning:** Connected to OpenRouter to query models such as `openai/gpt-oss-120b:free`.
- **Hugging Face Spaces Ready:** Dockerized with UID 1000 permissions and pre-cached models for instant cold boots.

---

## 🚀 Free Deployment to Hugging Face Spaces

### Step 1: Create a New Space
1. Log in to [Hugging Face](https://huggingface.co/).
2. Click **New Space** (or go to [huggingface.co/new-space](https://huggingface.co/new-space)).
3. Name your space (e.g., `document-rag-assistant`).
4. Select **Gradio** as the Space SDK (100% free, no credit card required).
5. Select the **Free** hardware tier (2 vCPU · 16 GB RAM).

### Step 2: Configure Secrets
1. Go to your Space's **Settings** tab.
2. Under **Variables and secrets**, click **New secret**.
3. Name: `OPENROUTER_API_KEY`
4. Value: *your OpenRouter API key*

### Step 3: Push Your Code
In your local repository terminal:

```bash
# Add Hugging Face Space as a remote
git remote add space https://huggingface.co/spaces/<YOUR_HF_USERNAME>/<YOUR_SPACE_NAME>

# Push the code
git push space main
```

Hugging Face will automatically build the Docker container and launch the web UI at your space's URL!

---

## 💻 Local Development

### Option A: Running with Python Virtualenv

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file with your key
echo "OPENROUTER_API_KEY=your_key_here" > .env

# Run FastAPI + Gradio server
uvicorn src.api:app --host 0.0.0.0 --port 7860 --reload
```

Open [http://localhost:7860](http://localhost:7860) for the Gradio UI, and [http://localhost:7860/docs](http://localhost:7860/docs) for the interactive Swagger API documentation.

### Option B: Running with Docker

```bash
# Build the image
docker build -t rag-app .

# Run the container
docker run -p 7860:7860 --env-file .env rag-app
```

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Interactive Gradio Web Chat & Ingestion UI |
| `GET` | `/docs` | OpenAPI / Swagger interactive documentation |
| `GET` | `/health` | Health check endpoint (`{"status": "ok"}`) |
| `POST` | `/query` | Query RAG pipeline (`{"query": "...", "top_k": 3}`) |
| `POST` | `/upload` | Upload new documents and rebuild FAISS index |
