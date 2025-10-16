"""
Vector Store for semantic search using ChromaDB
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorStore:
    """Manages document embeddings and retrieval using ChromaDB"""

    def __init__(
        self,
        persist_directory: str,
        embedding_model: str = "all-mpnet-base-v2",
        collection_name: str = "thesis_chunks",
    ):
        """
        Initialize vector store

        Args:
            persist_directory: Where to persist ChromaDB data
            embedding_model: Sentence transformer model name
                - all-mpnet-base-v2: Better quality, 768 dims (default)
                - all-MiniLM-L6-v2: Faster, 384 dims
            collection_name: Name for the ChromaDB collection
        """
        print(f"Initializing vector store with model: {embedding_model}")

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory, settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "PhD thesis document chunks with page references"},
        )

        print(f"✓ Vector store initialized. Collection has {self.collection.count()} items")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts"""
        embeddings = self.embedding_model.encode(
            texts, convert_to_tensor=False, show_progress_bar=True
        )
        return embeddings.tolist()

    def add_chunks(self, chunks: list[dict[str, any]], document_id: str, batch_size: int = 100):
        """
        Add document chunks to vector store

        Args:
            chunks: List of dicts with {text, page_num, chunk_index}
            document_id: Unique document identifier
            batch_size: Number of chunks to process at once
        """
        if not chunks:
            print("No chunks to add")
            return

        print(f"Adding {len(chunks)} chunks to vector store...")

        # Process in batches for efficiency
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]

            # Prepare data
            texts = [chunk["text"] for chunk in batch]
            embeddings = self.embed_texts(texts)

            # Create unique IDs
            ids = [
                f"{document_id}_page{chunk['page_num']}_chunk{chunk['chunk_index']}"
                for chunk in batch
            ]

            # Prepare metadata
            metadatas = [
                {
                    "document_id": document_id,
                    "page_num": chunk["page_num"],
                    "chunk_index": chunk["chunk_index"],
                    "text_preview": chunk["text"][:200],  # Store preview
                }
                for chunk in batch
            ]

            # Add to collection
            self.collection.add(
                ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
            )

            print(f"  Added batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")

        print(f"✓ Successfully added {len(chunks)} chunks")

    def search(
        self, query: str, top_k: int = 5, document_id: str | None = None
    ) -> list[dict[str, any]]:
        """
        Search for relevant chunks using semantic similarity

        Args:
            query: Search query
            top_k: Number of results to return
            document_id: Optional filter by document

        Returns:
            List of dicts with {text, page_num, score, metadata}
        """
        # Embed query
        query_embedding = self.embed_texts([query])[0]

        # Prepare filter
        where_filter = None
        if document_id:
            where_filter = {"document_id": document_id}

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k, where=where_filter
        )

        # Format results
        formatted_results = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                formatted_results.append(
                    {
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "page_num": results["metadatas"][0][i]["page_num"],
                        "score": results["distances"][0][i] if "distances" in results else None,
                        "metadata": results["metadatas"][0][i],
                    }
                )

        return formatted_results

    def search_by_page(self, page_num: int, document_id: str | None = None) -> list[dict[str, any]]:
        """
        Retrieve all chunks from a specific page

        Args:
            page_num: Page number to retrieve
            document_id: Optional filter by document

        Returns:
            List of dicts with {text, page_num, metadata}
        """
        # Build filter with proper ChromaDB operator syntax
        if document_id:
            where_filter = {"$and": [{"page_num": page_num}, {"document_id": document_id}]}
        else:
            where_filter = {"page_num": page_num}

        # Get all chunks from this page
        results = self.collection.get(where=where_filter, include=["documents", "metadatas"])

        # Format results
        formatted_results = []
        if results["ids"]:
            for i in range(len(results["ids"])):
                formatted_results.append(
                    {
                        "id": results["ids"][i],
                        "text": results["documents"][i],
                        "page_num": results["metadatas"][i]["page_num"],
                        "metadata": results["metadatas"][i],
                    }
                )

        return formatted_results

    def clear_document(self, document_id: str):
        """Remove all chunks for a specific document"""
        # Get all IDs for this document
        results = self.collection.get(where={"document_id": document_id})

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            print(f"✓ Removed {len(results['ids'])} chunks for document {document_id}")

    def reset_collection(self):
        """Delete and recreate the collection (use when changing embedding models)"""
        try:
            self.client.delete_collection(name=self.collection.name)
            print(f"✓ Deleted old collection: {self.collection.name}")
        except Exception as e:
            print(f"Note: Could not delete collection (may not exist): {e}")

        # Recreate collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"description": "PhD thesis document chunks with page references"},
        )
        print(f"✓ Created new collection with {self.embedding_dim} dimensions")

    def get_collection_dimension(self) -> int | None:
        """Get the actual dimension stored in the collection (not the model dimension)"""
        try:
            # Try to get one item to check its dimension
            results = self.collection.get(limit=1, include=["embeddings"])
            if results and results["embeddings"] and len(results["embeddings"]) > 0:
                return len(results["embeddings"][0])
        except Exception:
            pass
        return None

    def get_collection_stats(self) -> dict[str, any]:
        """Get statistics about the collection"""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": self.collection.name,
            "embedding_dimension": self.embedding_dim,
        }
