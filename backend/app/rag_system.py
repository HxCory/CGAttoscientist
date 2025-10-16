"""
Multimodal RAG System - Retrieval Augmented Generation with Vision
"""

from collections.abc import AsyncIterator

import anthropic

from app.document_processor import DocumentProcessor
from app.query_parser import QueryParser
from app.vector_store import VectorStore


class MultimodalRAG:
    """RAG system that uses both text retrieval and vision for answering questions"""

    # System prompt for the AI assistant
    SYSTEM_PROMPT = """You are an AI research assistant helping a physicist continue their doctoral research on attosecond streaking and time delays in photoionization.

You have access to pages from their PhD thesis. The pages contain equations, figures, tables, and detailed scientific content.

**IMPORTANT - Page Numbering:**
The thesis uses its own page numbering (starting at 1 for Chapter 1). However, the PDF pages you see have different numbers due to front matter (title page, abstract, acknowledgements, etc.).
- When you see a Table of Contents, the page numbers it lists are THESIS page numbers
- The actual PDF page numbers are shown as [Page X] in the content I provide
- There is an offset between these numbering systems (typically +22)
- When referencing pages, always use the PDF page numbers I provide, but understand the thesis numbering from the ToC

Your role is to:
1. Answer questions accurately based on the thesis content
2. Reference specific equations, figures, or sections when relevant
3. Use the Table of Contents to navigate the document structure
4. Help connect concepts and identify loose threads for future research
5. Be precise with mathematical and physical concepts

When you see equations or figures in the images, describe them clearly and relate them to the question."""

    def __init__(
        self,
        anthropic_api_key: str,
        vector_store: VectorStore,
        document_processor: DocumentProcessor,
        base_url: str | None = None,
    ):
        """
        Initialize RAG system

        Args:
            anthropic_api_key: Anthropic API key
            vector_store: Vector store instance
            document_processor: Document processor instance
            base_url: Optional base URL for API (for local proxy)
        """
        # Initialize Anthropic client
        client_kwargs = {"api_key": anthropic_api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
            print(f"Using custom base URL: {base_url}")

        self.client = anthropic.Anthropic(**client_kwargs)
        self.vector_store = vector_store
        self.doc_processor = document_processor
        self.query_parser = QueryParser()

    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
        toc_pages: list[int] | None = None,
        page_offset: int = 0,
    ) -> dict[str, any]:
        """
        Retrieve relevant context for a query using intelligent page-aware retrieval

        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            document_id: Optional document ID to search within
            toc_pages: Optional list of ToC page numbers to always include
            page_offset: Offset between thesis page numbers and PDF page numbers

        Returns:
            Dict with:
                - chunks: Retrieved text chunks
                - pages: Unique page numbers (includes ToC + retrieved pages)
                - page_images: Dict mapping page_num -> image_path
                - toc_pages: List of ToC page numbers for special handling
                - query_info: Parsed query information
        """
        # Parse query for explicit page/chapter references
        query_info = self.query_parser.parse_query(query)
        pages = set()
        page_to_image = {}
        results = []

        # Strategy 1: Direct page retrieval if specific pages are mentioned
        if query_info["has_page_reference"]:
            print(f"📍 Query mentions specific pages: {query_info}")

            # Get PDF pages to retrieve
            pdf_pages_to_fetch = set(query_info["pdf_pages"])

            # Convert thesis pages to PDF pages
            if query_info["thesis_pages"] and page_offset:
                for thesis_page in query_info["thesis_pages"]:
                    pdf_page = thesis_page + page_offset
                    pdf_pages_to_fetch.add(pdf_page)
                    print(f"   Thesis page {thesis_page} → PDF page {pdf_page}")

            # Handle page ranges
            if query_info["page_range"]:
                start, end = query_info["page_range"]
                for page in range(start, end + 1):
                    pdf_pages_to_fetch.add(page)

            # Retrieve all mentioned pages
            pages.update(pdf_pages_to_fetch)
            print(f"   Retrieving pages: {sorted(pdf_pages_to_fetch)}")

            # Get chunks from these specific pages for context
            if document_id and pdf_pages_to_fetch:
                for page_num in pdf_pages_to_fetch:
                    # Get chunks from this page
                    page_results = self.vector_store.search_by_page(
                        page_num=page_num, document_id=document_id
                    )
                    results.extend(page_results)

        # Strategy 2: Semantic search (either as fallback or if no page refs found)
        if not query_info["has_page_reference"] or len(pages) == 0:
            print(f"🔍 Using semantic search for: {query[:60]}...")
            results = self.vector_store.search(query=query, top_k=top_k, document_id=document_id)

            # Collect unique pages from search results
            for result in results:
                pages.add(result["page_num"])

        # Get image paths for all pages
        if document_id:
            for page_num in pages:
                image_path = (
                    self.doc_processor.page_images_dir / document_id / f"page_{page_num:04d}.png"
                )
                if image_path.exists():
                    page_to_image[page_num] = str(image_path)

        # Always include ToC pages if provided
        toc_pages_list = toc_pages or []
        if toc_pages_list and document_id:
            for toc_page in toc_pages_list:
                pages.add(toc_page)
                image_path = (
                    self.doc_processor.page_images_dir / document_id / f"page_{toc_page:04d}.png"
                )
                if image_path.exists():
                    page_to_image[toc_page] = str(image_path)

        return {
            "chunks": results,
            "pages": sorted(pages),
            "page_images": page_to_image,
            "toc_pages": toc_pages_list,
            "query_info": query_info,
        }

    def _prepare_image_content(self, image_path: str) -> dict[str, any]:
        """Prepare image for Claude API"""
        # Read and encode image
        image_data = self.doc_processor.image_to_base64(image_path)

        # Determine media type
        if image_path.endswith(".png"):
            media_type = "image/png"
        elif image_path.endswith(".jpg") or image_path.endswith(".jpeg"):
            media_type = "image/jpeg"
        else:
            media_type = "image/png"

        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": image_data},
        }

    def _build_message_content(
        self,
        context: dict[str, any],
        question: str,
        include_images: bool = True,
        page_offset: int = 0,
    ) -> list[dict[str, any]]:
        """
        Build message content for Claude API

        Args:
            context: Retrieved context with pages and images
            question: User's question
            include_images: Whether to include page images (multimodal)
            page_offset: Offset between thesis page numbers and PDF page numbers

        Returns:
            List of content blocks for the message
        """
        content = []
        toc_pages = context.get("toc_pages", [])
        retrieved_pages = [p for p in context["pages"] if p not in toc_pages]

        # Add page images if multimodal
        if include_images and context["page_images"]:
            # First, add ToC pages if available
            if toc_pages:
                offset_note = ""
                if page_offset > 0:
                    offset_note = f" (NOTE: Thesis page X = PDF page X+{page_offset}. E.g., thesis page 1 = PDF page {1+page_offset})"

                content.append(
                    {
                        "type": "text",
                        "text": f"📚 TABLE OF CONTENTS{offset_note}:",
                    }
                )
                for page_num in toc_pages:
                    if page_num in context["page_images"]:
                        image_path = context["page_images"][page_num]
                        content.append(self._prepare_image_content(image_path))
                        content.append({"type": "text", "text": f"[ToC - PDF Page {page_num}]"})

            # Then add retrieved pages
            if retrieved_pages:
                content.append(
                    {
                        "type": "text",
                        "text": f"\n\n📄 RELEVANT PAGES (pages {', '.join(map(str, retrieved_pages))}):",
                    }
                )
                for page_num in retrieved_pages:
                    if page_num in context["page_images"]:
                        image_path = context["page_images"][page_num]
                        content.append(self._prepare_image_content(image_path))
                        content.append({"type": "text", "text": f"[Page {page_num}]"})
        else:
            # Text-only fallback
            text_context = "\n\n".join(
                [f"[Page {chunk['page_num']}]\n{chunk['text']}" for chunk in context["chunks"]]
            )
            content.append(
                {"type": "text", "text": f"Relevant excerpts from the thesis:\n\n{text_context}"}
            )

        # Add the question with instructions
        instruction = "\n\nQuestion: {}\n\nPlease answer based on the thesis content above.".format(
            question
        )
        if toc_pages:
            instruction += " Use the table of contents to understand the document structure and locate specific chapters/sections."
            instruction += " Note: Page numbers in the ToC are thesis page numbers; add the offset to get actual PDF page numbers shown in brackets."

        content.append({"type": "text", "text": instruction})

        return content

    def ask(
        self,
        question: str,
        document_id: str | None = None,
        top_k: int = 15,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 4096,
        include_images: bool = True,
        toc_pages: list[int] | None = None,
        page_offset: int = 0,
    ) -> dict[str, any]:
        """
        Ask a question about the document

        Args:
            question: User's question
            document_id: Optional document to search in
            top_k: Number of chunks to retrieve (default 15 for better coverage)
            model: Claude model to use
            max_tokens: Max tokens in response
            include_images: Whether to include page images (multimodal)
            toc_pages: Optional list of table of contents pages to always include
            page_offset: Offset between thesis page numbers and PDF page numbers

        Returns:
            Dict with answer, sources, and metadata
        """
        # Retrieve relevant context
        print(f"Retrieving context for: {question}")
        context = self.retrieve_context(
            query=question,
            top_k=top_k,
            document_id=document_id,
            toc_pages=toc_pages,
            page_offset=page_offset,
        )

        print(f"Found {len(context['pages'])} relevant pages: {context['pages']}")

        # Build message content
        content = self._build_message_content(context, question, include_images, page_offset)

        # Call Claude API
        print(f"Calling Claude API ({model})...")
        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )

        answer = response.content[0].text

        return {
            "answer": answer,
            "sources": {"pages": context["pages"], "chunks": context["chunks"]},
            "metadata": {
                "model": model,
                "tokens_used": {
                    "input": response.usage.input_tokens,
                    "output": response.usage.output_tokens,
                },
            },
        }

    async def ask_streaming(
        self,
        question: str,
        document_id: str | None = None,
        top_k: int = 15,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 4096,
        include_images: bool = True,
        toc_pages: list[int] | None = None,
        page_offset: int = 0,
    ) -> AsyncIterator[dict[str, any]]:
        """
        Ask a question with streaming response

        Yields:
            Dicts with streaming data
        """
        # Retrieve relevant context
        print(f"Retrieving context for: {question}")
        context = self.retrieve_context(
            query=question,
            top_k=top_k,
            document_id=document_id,
            toc_pages=toc_pages,
            page_offset=page_offset,
        )

        print(f"Found {len(context['pages'])} relevant pages: {context['pages']}")

        # Yield sources first
        yield {"type": "sources", "data": {"pages": context["pages"], "chunks": context["chunks"]}}

        # Build message content
        content = self._build_message_content(context, question, include_images, page_offset)

        # Stream response
        print(f"Streaming from Claude API ({model})...")

        async with self.client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        ) as stream:
            async for event in stream:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        yield {"type": "content", "data": event.delta.text}
                elif event.type == "message_stop":
                    yield {"type": "done", "data": {}}
