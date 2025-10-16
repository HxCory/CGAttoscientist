#!/bin/bash

echo "=========================================="
echo "Clean Restart - Delete Old Embeddings"
echo "=========================================="

# Stop services
echo "🛑 Stopping services..."
pkill -9 -f "uvicorn" 2>/dev/null
pkill -9 -f "vite" 2>/dev/null
pkill -9 -f "start.sh" 2>/dev/null
sleep 2

# Delete ChromaDB collection
echo "🗑️  Deleting old ChromaDB collection..."
rm -rf chroma_db/
echo "✓ Old embeddings deleted"

# Keep document metadata and uploaded files
echo "✓ Keeping document metadata and uploaded PDFs"
echo ""
echo "=========================================="
echo "Starting fresh with new embedding model"
echo "=========================================="

# Start services
./start.sh
