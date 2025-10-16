# PhD Thesis RAG Assistant 🎓

An AI-powered research assistant that uses **Multimodal Retrieval Augmented Generation (RAG)** to help continue doctoral research on "Analytical Modeling and Numerical Simulations of Time Delays in Attosecond Streaking of One- and Two-Photon Ionization".

## 🌟 Features

- **📄 PDF Processing**: Converts thesis pages to images for multimodal understanding
- **🔍 Semantic Search**: High-quality embeddings with intelligent chunk retrieval
- **📚 Table of Contents Detection**: Automatically finds and includes ToC for document navigation
- **👁️ Vision-Powered AI**: Claude "sees" your thesis pages - equations, figures, tables, everything
- **💬 Interactive Chat**: Natural conversation about your research
- **💾 Document Persistence**: Upload once, embeddings persist forever - no re-processing needed
- **🔄 Smart Re-processing**: Detects missing embeddings and offers one-click regeneration
- **📑 Source Citations**: See which pages were used to answer each question
- **⚡ Fast & Local**: ChromaDB for efficient local vector storage

## 🏗️ Architecture

### Backend (Python/FastAPI)
- **Document Processing**: Converts PDF → Images + Text chunks
- **Vector Store**: ChromaDB with sentence-transformers embeddings
- **Multimodal RAG**: Retrieves relevant pages and sends images to Claude
- **API**: RESTful endpoints for upload, query, and management

### Frontend (React/Vite)
- **Modern UI**: Tailwind CSS with beautiful gradient design
- **Real-time Chat**: Instant responses with source tracking
- **File Upload**: Drag-and-drop PDF processing
- **Responsive**: Works on desktop and mobile

### AI Integration
- **Claude 3.5 Sonnet**: Latest multimodal model
- **Vision Capabilities**: Reads equations, figures, tables directly from images
- **Local Proxy Support**: Configure custom API endpoint

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- Anthropic API key (or local proxy)
- `poppler-utils` for PDF processing:
  ```bash
  # macOS
  brew install poppler

  # Ubuntu/Debian
  sudo apt-get install poppler-utils

  # Windows
  # Download from: https://github.com/oschwartz10612/poppler-windows/releases/
  ```

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment (optional - only if you don't have ANTHROPIC_API_KEY in system env)
# The backend will use your system environment variable if available
# If not, you can create a .env file and add it there

# Start the server
python -m uvicorn app.main:app --reload
```

The backend will be available at `http://localhost:8000`

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

The frontend will be available at `http://localhost:5173`

## 📖 Usage

### 1. Upload Your Thesis

1. Click **"Upload Thesis PDF"** in the top right
2. Select your PhD thesis PDF
3. Wait for processing (converts pages to images and creates embeddings)
4. You'll see a confirmation with page count and chunks created

### 2. Ask Questions

Ask questions about your research:

- *"What were the main findings about Wigner time delays?"*
- *"Can you explain the continuum-continuum coupling mentioned in Chapter 3?"*
- *"What loose threads or future work did I identify?"*
- *"Show me the equation for the streaking momentum shift"*
- *"What figures demonstrate the two-photon ionization results?"*

### 3. View Sources

Each answer includes:
- **Page references**: Which pages were used
- **Token usage**: API cost tracking
- **Direct citations**: See the original thesis content

## 🔧 Configuration

### Backend Environment Variables

**Recommended (Most Secure):** Set `ANTHROPIC_API_KEY` in your system environment:
```bash
# Add to ~/.profile, ~/.zshrc, or ~/.bashrc
export ANTHROPIC_API_KEY=sk-ant-...
```

**Alternative:** Create `backend/.env` file:
```bash
# Required: Your Anthropic API key (only if not in system env)
ANTHROPIC_API_KEY=sk-ant-...

# Optional: Use a local proxy
# ANTHROPIC_BASE_URL=http://localhost:8080

# Storage paths
CHROMA_PERSIST_DIRECTORY=./chroma_db
PAGE_IMAGES_DIR=./page_images
UPLOADS_DIR=./uploads

# RAG settings
EMBEDDING_MODEL=all-mpnet-base-v2  # High-quality embeddings (default)
TOP_K_RESULTS=15                   # Number of chunks to retrieve
CHUNK_SIZE=600                     # Characters per chunk (smaller = more granular)
CHUNK_OVERLAP=150                  # Overlap between chunks

# Server
HOST=0.0.0.0
PORT=8000
```

### Using a Local Proxy

If you have a local proxy for the Anthropic API:

1. Set `ANTHROPIC_BASE_URL` in `.env`
2. The system will use your proxy instead of direct API calls

## 🧪 How It Works

### Multimodal RAG Pipeline

1. **Ingestion**:
   - PDF pages → High-quality images (150 DPI)
   - Extract text for semantic search
   - Chunk text with overlap for context

2. **Indexing**:
   - Create embeddings with sentence-transformers
   - Store in ChromaDB with page metadata
   - Keep page images on disk

3. **Retrieval**:
   - User asks a question
   - Embed question and search vector store
   - Find top-k most similar chunks
   - Retrieve corresponding page images

4. **Generation**:
   - Send page images + question to Claude
   - Claude "sees" the actual thesis pages
   - Reads equations, figures, tables directly
   - Generates answer grounded in visual context

### Why Multimodal?

Traditional PDF text extraction **fails for physics theses**:
- ❌ Math equations become garbled
- ❌ Figures are lost completely
- ❌ Table structure breaks
- ❌ Complex layouts scramble order

**Multimodal RAG solves this**:
- ✅ Claude sees the actual page images
- ✅ Perfect math rendering (LaTeX, equations)
- ✅ Figures and diagrams included
- ✅ Tables preserved exactly
- ✅ Layout doesn't matter

## 📊 API Endpoints

### `GET /`
Health check

### `GET /stats`
Get system statistics (chunk count, config)

### `POST /upload`
Upload and process a PDF
```json
{
  "file": "<PDF file>",
  "document_id": "optional-custom-id"
}
```

### `POST /ask`
Ask a question
```json
{
  "question": "What is the Wigner time delay?",
  "document_id": "optional-doc-id",
  "top_k": 3,
  "include_images": true,
  "stream": false
}
```

### `DELETE /document/{document_id}`
Delete a document and all its data

## 🎨 Customization

### Change Embedding Model

In `.env`, set `EMBEDDING_MODEL` to any sentence-transformers model:
- `all-mpnet-base-v2` (default, high quality, 768 dimensions)
- `all-MiniLM-L6-v2` (faster, good quality, 384 dimensions)
- `multi-qa-mpnet-base-dot-v1` (optimized for Q&A)

### Adjust Retrieval

- `TOP_K_RESULTS`: More pages = more context (but slower/more expensive)
- `CHUNK_SIZE`: Larger = more context per chunk
- `CHUNK_OVERLAP`: More overlap = better context continuity

### Change Claude Model

In `app/rag_system.py`, modify the `model` parameter in `ask()` method.

## 🐛 Troubleshooting

### "poppler not found"
Install poppler-utils (see Prerequisites)

### "ANTHROPIC_API_KEY not configured"
Set your API key in `backend/.env`

### Backend won't start
Check Python version: `python --version` (need 3.9+)

### Frontend won't connect
Ensure backend is running on port 8000

### Slow processing
- Reduce image DPI in `document_processor.py`
- Use smaller `TOP_K_RESULTS`
- Use faster embedding model

## 📝 Development

### Project Structure

```
CGAttoscientist/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app
│   │   ├── document_processor.py # PDF → Images + Text
│   │   ├── vector_store.py      # ChromaDB wrapper
│   │   └── rag_system.py        # Multimodal RAG logic
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main React component
│   │   ├── main.jsx
│   │   └── index.css
│   ├── package.json
│   └── vite.config.js
└── README.md
```

### Adding Features

**Streaming responses**: The backend supports streaming, just update the frontend to use EventSource or fetch with streaming.

**Multiple documents**: The system supports multiple documents via `document_id`.

**LaTeX source**: If you want to add LaTeX source files later, create a new processor in `app/` that parses .tex files.

## 🎯 Future Enhancements

- [ ] Streaming responses in UI
- [ ] Chat history persistence
- [ ] Export conversations to PDF
- [ ] Multi-document search
- [ ] LaTeX source integration
- [ ] Annotation system
- [ ] Research thread tracking
- [ ] Citation graph visualization

## 📄 License

MIT License - feel free to use this for your research!

## 🙏 Acknowledgments

Built with:
- [Anthropic Claude](https://www.anthropic.com/) - Multimodal AI
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [FastAPI](https://fastapi.tiangolo.com/) - Backend framework
- [React](https://react.dev/) - Frontend framework
- [Tailwind CSS](https://tailwindcss.com/) - Styling

---

**Happy researching! 🔬✨**
