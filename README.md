# NanoSearch

NanoSearch is a semantic search engine for nanoscience papers. It compares various embedding models and segmentation strategies.

## Features

- **Embedding Support**: 
  - NanoBERT embeddings (huggingface: alan-yahya/NanoBERT-V2)
  - OpenAI embeddings (OpenAI: text-embedding-3-large)
- **Segmentation Strategies**:
  - Recursive segmentation
  - Direct segmentation
  - Variable context windows: 512 tokens (BERT), 8191 tokens (OpenAI)
- **FAISS Vector Search**: Fast similarity search using Facebook AI Similarity Search

## Installation

1. Clone the repository:   ```bash
   git clone https://github.com/yourusername/nanosearch.git
   cd nanosearch   ```

2. Set up environment variables:   ```bash
   cp .env.example .env
   # Edit .env with your API keys:
   # - HF_API_KEY (Hugging Face)
   # - OPENAI_API_KEY (OpenAI)
   # - FLASK_SECRET_KEY   ```

3. Using Docker:   ```bash
   docker build -t nanosearch .
   docker run -p 3000:3000 nanosearch   ```

   Or install locally:   ```bash
   pip install -r requirements.txt
   python app.py   ```

4. Access the application at `http://localhost:3000`