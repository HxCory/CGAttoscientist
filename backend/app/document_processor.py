"""
Document Processor for Multimodal RAG
Converts PDF pages to images and extracts text for semantic search
"""

import base64
from pathlib import Path

from pdf2image import convert_from_path
from pypdf import PdfReader


class DocumentProcessor:
    """Processes PDF documents for multimodal RAG"""

    def __init__(self, page_images_dir: str):
        self.page_images_dir = Path(page_images_dir)
        self.page_images_dir.mkdir(parents=True, exist_ok=True)

    def process_pdf(
        self,
        pdf_path: str,
        document_id: str,
        dpi: int = 150,  # Good balance between quality and file size
    ) -> dict[str, any]:
        """
        Process a PDF: extract text and convert pages to images

        Args:
            pdf_path: Path to PDF file
            document_id: Unique identifier for this document
            dpi: Resolution for page images

        Returns:
            Dict with:
                - pages: List of dicts with {page_num, text, image_path}
                - metadata: Document metadata
        """
        print(f"Processing PDF: {pdf_path}")

        # Create directory for this document's images
        doc_image_dir = self.page_images_dir / document_id
        doc_image_dir.mkdir(parents=True, exist_ok=True)

        # Extract text from PDF
        print("Extracting text from PDF...")
        pdf_reader = PdfReader(pdf_path)
        num_pages = len(pdf_reader.pages)

        pages_data = []
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            text = page.extract_text()
            pages_data.append(
                {"page_num": page_num, "text": text, "image_path": None}  # Will be set below
            )

        # Convert PDF pages to images
        print(f"Converting {num_pages} pages to images at {dpi} DPI...")
        images = convert_from_path(pdf_path, dpi=dpi)

        # Save images and update paths
        for idx, image in enumerate(images):
            page_num = idx + 1
            image_filename = f"page_{page_num:04d}.png"
            image_path = doc_image_dir / image_filename

            # Save image (compressed to save space)
            image.save(image_path, "PNG", optimize=True)

            # Update the corresponding page data
            pages_data[idx]["image_path"] = str(image_path)

            if page_num % 10 == 0:
                print(f"  Processed {page_num}/{num_pages} pages")

        print(f"✓ Successfully processed {num_pages} pages")

        return {
            "pages": pages_data,
            "metadata": {"num_pages": num_pages, "document_id": document_id, "pdf_path": pdf_path},
        }

    def chunk_text(
        self, text: str, page_num: int, chunk_size: int = 1000, chunk_overlap: int = 200
    ) -> list[dict[str, any]]:
        """
        Split text into overlapping chunks for better retrieval

        Args:
            text: Text to chunk
            page_num: Page number this text came from
            chunk_size: Target size of each chunk (characters)
            chunk_overlap: Overlap between chunks

        Returns:
            List of dicts with {text, page_num, chunk_index}
        """
        if not text.strip():
            return []

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Extract chunk
            end = start + chunk_size
            chunk_text = text[start:end]

            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings in the last 100 chars
                last_period = chunk_text.rfind(". ")
                last_newline = chunk_text.rfind("\n")
                break_point = max(last_period, last_newline)

                if break_point > chunk_size - 200:  # Only break if it's not too far back
                    chunk_text = chunk_text[: break_point + 1]
                    end = start + break_point + 1

            chunks.append(
                {"text": chunk_text.strip(), "page_num": page_num, "chunk_index": chunk_index}
            )

            # Move to next chunk with overlap
            start = end - chunk_overlap
            chunk_index += 1

        return chunks

    def image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 for API transmission"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_page_image(self, image_path: str) -> bytes:
        """Load page image as bytes"""
        with open(image_path, "rb") as f:
            return f.read()
