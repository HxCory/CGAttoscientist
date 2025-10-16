"""
Document Manager - Tracks processed documents and their metadata
"""

import json
from datetime import datetime
from pathlib import Path


class DocumentManager:
    """Manages metadata for processed documents"""

    def __init__(self, metadata_file: str = "document_metadata.json"):
        self.metadata_file = Path(metadata_file)
        self.metadata: dict[str, dict] = self._load_metadata()

    def _load_metadata(self) -> dict[str, dict]:
        """Load document metadata from disk"""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {}

    def _save_metadata(self):
        """Save document metadata to disk"""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def add_document(
        self,
        document_id: str,
        filename: str,
        num_pages: int,
        num_chunks: int,
        pdf_path: str,
        image_dir: str,
    ):
        """Register a processed document"""
        self.metadata[document_id] = {
            "filename": filename,
            "document_id": document_id,
            "num_pages": num_pages,
            "num_chunks": num_chunks,
            "pdf_path": pdf_path,
            "image_dir": image_dir,
            "processed_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
        }
        self._save_metadata()

    def get_document(self, document_id: str) -> dict | None:
        """Get metadata for a specific document"""
        doc = self.metadata.get(document_id)
        if doc:
            # Update last accessed time
            doc["last_accessed"] = datetime.now().isoformat()
            self._save_metadata()
        return doc

    def list_documents(self) -> list[dict]:
        """List all processed documents"""
        return list(self.metadata.values())

    def delete_document(self, document_id: str) -> bool:
        """Remove document from metadata"""
        if document_id in self.metadata:
            del self.metadata[document_id]
            self._save_metadata()
            return True
        return False

    def document_exists(self, document_id: str) -> bool:
        """Check if a document has been processed"""
        return document_id in self.metadata

    def get_document_chunk_count(self, document_id: str, vector_store) -> int:
        """Get actual chunk count from vector store for a document"""
        if not self.document_exists(document_id):
            return 0
        # Query vector store to count chunks for this document
        results = vector_store.collection.get(where={"document_id": document_id}, limit=1)
        if results and results["ids"]:
            # Get total count by querying
            all_results = vector_store.collection.get(where={"document_id": document_id})
            return len(all_results["ids"]) if all_results["ids"] else 0
        return 0
