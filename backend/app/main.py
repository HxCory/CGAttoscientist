"""
FastAPI Backend for PhD Thesis RAG Assistant
"""
import os
import shutil
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import json

from app.document_processor import DocumentProcessor
from app.vector_store import VectorStore
from app.rag_system import MultimodalRAG

# Load environment variables
load_dotenv()

# Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL")  # Optional
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
PAGE_IMAGES_DIR = os.getenv("PAGE_IMAGES_DIR", "./page_images")
UPLOADS_DIR = os.getenv("UPLOADS_DIR", "./uploads")
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Create directories
Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)
Path(PAGE_IMAGES_DIR).mkdir(parents=True, exist_ok=True)
Path(CHROMA_PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)

# Global instances (initialized in lifespan)
document_processor: Optional[DocumentProcessor] = None
vector_store: Optional[VectorStore] = None
rag_system: Optional[MultimodalRAG] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global document_processor, vector_store, rag_system
    
    print("=" * 60)
    print("🚀 Starting PhD Thesis RAG Assistant")
    print("=" * 60)
    
    # Initialize components
    print("\n1. Initializing Document Processor...")
    document_processor = DocumentProcessor(page_images_dir=PAGE_IMAGES_DIR)
    
    print("\n2. Initializing Vector Store...")
    vector_store = VectorStore(
        persist_directory=CHROMA_PERSIST_DIRECTORY,
        embedding_model=EMBEDDING_MODEL
    )
    
    print("\n3. Initializing RAG System...")
    if not ANTHROPIC_API_KEY:
        print("⚠️  WARNING: ANTHROPIC_API_KEY not set!")
    
    rag_system = MultimodalRAG(
        anthropic_api_key=ANTHROPIC_API_KEY or "dummy-key",
        vector_store=vector_store,
        document_processor=document_processor,
        base_url=ANTHROPIC_BASE_URL
    )
    
    stats = vector_store.get_collection_stats()
    print(f"\n✓ System ready! Vector store has {stats['total_chunks']} chunks")
    print("=" * 60)
    
    yield
    
    print("\n👋 Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="PhD Thesis RAG Assistant",
    description="Multimodal RAG system for continuing doctoral research",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class QuestionRequest(BaseModel):
    question: str
    document_id: Optional[str] = None
    top_k: Optional[int] = None
    include_images: Optional[bool] = True
    stream: Optional[bool] = False


class QuestionResponse(BaseModel):
    answer: str
    sources: dict
    metadata: dict


# API Routes

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "running",
        "service": "PhD Thesis RAG Assistant",
        "version": "1.0.0"
    }


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    stats = vector_store.get_collection_stats()
    return {
        "vector_store": stats,
        "config": {
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k_default": TOP_K_RESULTS
        }
    }


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    document_id: Optional[str] = Query(None, description="Optional document ID")
):
    """
    Upload and process a PDF document
    
    This will:
    1. Save the PDF
    2. Convert pages to images
    3. Extract text
    4. Create embeddings
    5. Store in vector database
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Generate document ID if not provided
    if not document_id:
        document_id = Path(file.filename).stem
    
    print(f"\n📄 Processing upload: {file.filename} (ID: {document_id})")
    
    try:
        # Save uploaded file
        pdf_path = Path(UPLOADS_DIR) / f"{document_id}.pdf"
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"✓ Saved to {pdf_path}")
        
        # Process PDF (extract text and create images)
        result = document_processor.process_pdf(
            pdf_path=str(pdf_path),
            document_id=document_id
        )
        
        # Create chunks from extracted text
        all_chunks = []
        for page_data in result['pages']:
            chunks = document_processor.chunk_text(
                text=page_data['text'],
                page_num=page_data['page_num'],
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            # Add image path to each chunk's metadata
            for chunk in chunks:
                chunk['image_path'] = page_data['image_path']
            all_chunks.extend(chunks)
        
        print(f"\n✓ Created {len(all_chunks)} text chunks")
        
        # Add to vector store
        vector_store.add_chunks(
            chunks=all_chunks,
            document_id=document_id
        )
        
        stats = vector_store.get_collection_stats()
        
        return {
            "status": "success",
            "message": f"Successfully processed {file.filename}",
            "document_id": document_id,
            "pages_processed": result['metadata']['num_pages'],
            "chunks_created": len(all_chunks),
            "total_chunks_in_db": stats['total_chunks']
        }
        
    except Exception as e:
        print(f"❌ Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Ask a question about the document
    
    Supports both regular and streaming responses
    """
    if not ANTHROPIC_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="ANTHROPIC_API_KEY not configured"
        )
    
    top_k = request.top_k or TOP_K_RESULTS
    
    try:
        if request.stream:
            # Streaming response
            async def stream_generator():
                async for chunk in rag_system.ask_streaming(
                    question=request.question,
                    document_id=request.document_id,
                    top_k=top_k,
                    include_images=request.include_images
                ):
                    # Send as Server-Sent Events format
                    yield f"data: {json.dumps(chunk)}\n\n"
            
            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream"
            )
        else:
            # Regular response
            result = rag_system.ask(
                question=request.question,
                document_id=request.document_id,
                top_k=top_k,
                include_images=request.include_images
            )
            return result
            
    except Exception as e:
        print(f"❌ Error answering question: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error answering question: {str(e)}"
        )


@app.delete("/document/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all its data"""
    try:
        # Remove from vector store
        vector_store.clear_document(document_id)
        
        # Remove images
        image_dir = Path(PAGE_IMAGES_DIR) / document_id
        if image_dir.exists():
            shutil.rmtree(image_dir)
        
        # Remove PDF
        pdf_path = Path(UPLOADS_DIR) / f"{document_id}.pdf"
        if pdf_path.exists():
            pdf_path.unlink()
        
        return {
            "status": "success",
            "message": f"Document {document_id} deleted"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True
    )

