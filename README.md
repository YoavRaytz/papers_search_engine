# Papers Search Engine - RAG Application

A production-ready Retrieval-Augmented Generation (RAG) system for querying academic papers using local LLM with streaming responses and AI image generation.

## ✨ Features

- ✅ **Memory-Efficient**: Streams datasets without loading everything to RAM
- ✅ **Local Processing**: FAISS vector indexing + TinyLlama LLM (fully on-premise)
- ✅ **Web Interface**: Interactive browser UI at http://127.0.0.1:8080
- ✅ **Smart Retrieval**: Automatically finds relevant papers and generates contextual answers
- ✅ **Streaming Responses**: Real-time word-by-word answer generation
- ✅ **AI Image Generation**: Automated visualization with Stable Diffusion XL
- ✅ **Docker Ready**: Single command deployment

## 🚀 Quick Start

### Option 1: Automated Script (Recommended)

The easiest way to run the application is using the provided automation script:

```bash
# Make the script executable (first time only)
chmod +x run.sh

# Run the script
./run.sh
```

**What the script does:**
- ✅ Automatically builds the Docker image if it doesn't exist
- ✅ Reuses existing container if already created (just starts it)
- ✅ Asks if you want to use GPU support
- ✅ Mounts the `data` folder automatically
- ✅ Shows helpful commands for managing the container


### Option 2: Manual Docker Commands

#### Docker (CPU)

```bash
# Build image
docker build -t genai-app:latest .

# Run with mounted data folder
docker run --rm -p 8080:8080 \
  -e DATA_PATH=/data/arxiv.jsonl \
  -v $(pwd)/data:/data:ro \
  genai-app:latest
```

#### Docker with GPU 

Requires: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
# Run with GPU support
docker run --rm -p 8080:8080 \
  --runtime=nvidia \
  --gpus all \
  -e DATA_PATH=/data/arxiv.jsonl \
  -v $(pwd)/data:/data:ro \
  genai-app:latest
```

The app will be accessible at **http://0.0.0.0:8080**

## 🔄 Continuous Running / Restart Container

If you want to keep the container running in the background (without `--rm` flag):

```bash
# Run container with a name (without --rm to persist it)
docker run -d --name papers-search \
  -p 8080:8080 \
  -e DATA_PATH=/data/arxiv.jsonl \
  -v $(pwd)/data:/data:ro \
  genai-app:latest
```

### Managing the Container

```bash
# Check running containers and get the container name
sudo docker ps

# Stop the container
sudo docker stop genai-app

# Start the container again
sudo docker start -ai genai-app

# View logs
sudo docker logs genai-app

# Remove the container (when done)
sudo docker rm genai-app
```

### Automatic Dataset Change Detection

The system automatically detects changes to the `arxiv.jsonl` file using SHA256 fingerprinting:

- **First run**: Builds the index from scratch
- **Subsequent runs**: Loads the existing index (fast startup)
- **When `arxiv.jsonl` changes**: Automatically detects the change and rebuilds the index

**Important**: Keep the filename as `arxiv.jsonl` in the `data` folder. The system will automatically detect any content changes and rebuild the index accordingly.

## 🎨 Image Generation Setup (Optional)

To enable AI-generated visualizations:

1. Create free account at https://huggingface.co/join
2. Generate API token at https://huggingface.co/settings/tokens (Read permissions)
3. Pass as environment variable when running Docker:
   ```bash
   docker run --rm -p 8080:8080 \
     -e DATA_PATH=/data/arxiv.jsonl \
     -e HF_API_TOKEN=hf_your_token_here \
     -v $(pwd)/data:/data:ro \
     genai-app:latest
   ```

Uses **Stable Diffusion XL Base 1.0** via HuggingFace Inference API.

## 🌐 Web Interface

Interactive UI provides:

- **Answer Box**: LLM-generated response with streaming option
- **AI Image**: Automated visualization from key concepts
- **Retrieved Context**: Citations with relevant paper excerpts
- **Source Documents**: Full list of papers used
- **Raw JSON**: Complete API response

**Example Queries:**
- "What are transformer models?"
- "How do attention mechanisms work?"
- "What does the Transformer architecture improve compared to RNNs?"




## 🏗️ How It Works

```
User Query
    ↓
Vector Search (FAISS)     → Find top-k similar papers
    ↓
Load Documents (Streaming) → Get full paper content
    ↓
Generate Answer (LLM)      → TinyLlama creates response
    ↓
Extract Concepts           → Analyze answer for key terms
    ↓
AI Image Generation        → Stable Diffusion XL (optional)
    ↓
Web Response               → Answer + Citations + Image
```

**RAG Pipeline:**
1. **Retrieval**: Vector search finds relevant papers
2. **Augmentation**: Build prompt with retrieved context
3. **Generation**: LLM generates grounded answer

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.10+ |
| **Vector DB** | FAISS IndexFlatL2 (L2 distance) |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 (384-dim) |
| **LLM** | TinyLlama-1.1B-Chat-v1.0 (CPU/GPU) |
| **Image Gen** | Stable Diffusion XL Base 1.0 (HF API) |
| **Web** | Flask + Server-Sent Events |
| **Dataset** | 2,900 ArXiv papers (~5.7MB JSONL) |


## � Features

- ✅ Loads JSONL dataset of academic abstracts
- ✅ Indexes locally with FAISS vector search
- ✅ Answers queries using RAG (retrieval + local LLM)
- ✅ Runs in browser at http://127.0.0.1:8080
- ✅ Docker deployment ready
- ✅ Fully on-premise (no external APIs required for core features)
- ✅ Memory-efficient (scalable to millions of records)
- ✅ Real-time SSE streaming with word-by-word generation
- ✅ Interactive GUI with streaming toggle
- ✅ AI image generation with key concept extraction
- ✅ Citation display with source attribution
- ✅ Automatic index rebuilding on dataset changes

## 📜 License

MIT License

