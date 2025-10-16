"""
Multimodal RAG System - Retrieval Augmented Generation with Vision
"""

from collections.abc import AsyncIterator

import anthropic

from app.document_processor import DocumentProcessor
from app.vector_store import VectorStore


class MultimodalRAG:
    """RAG system that uses both text retrieval and vision for answering questions"""

    # System prompt for the AI assistant
    SYSTEM_PROMPT = """You are an AI research assistant helping a physicist continue their doctoral research on attosecond streaking and time delays in photoionization.

You have access to pages from their PhD thesis. The pages contain equations, figures, tables, and detailed scientific content.

Your role is to:
1. Answer questions accurately based on the thesis content
2. Reference specific equations, figures, or sections when relevant
3. Help connect concepts and identify loose threads for future research
4. Be precise with mathematical and physical concepts

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

    def retrieve_context(
        self, query: str, top_k: int = 5, document_id: str | None = None
    ) -> dict[str, any]:
        """
        Retrieve relevant context for a query

        Returns:
            Dict with:
                - chunks: Retrieved text chunks
                - pages: Unique page numbers
                - page_images: Dict mapping page_num -> image_path
        """
        # Search for relevant chunks
        results = self.vector_store.search(query=query, top_k=top_k, document_id=document_id)

        # Collect unique pages
        pages = set()
        page_to_image = {}

        for result in results:
            page_num = result["page_num"]
            pages.add(page_num)

            # Get image path from metadata (we'll need to store this)
            # For now, we'll reconstruct it from the document_id
            doc_id = result["metadata"]["document_id"]
            image_path = self.doc_processor.page_images_dir / doc_id / f"page_{page_num:04d}.png"

            if image_path.exists():
                page_to_image[page_num] = str(image_path)

        return {"chunks": results, "pages": sorted(pages), "page_images": page_to_image}

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
        self, context: dict[str, any], question: str, include_images: bool = True
    ) -> list[dict[str, any]]:
        """
        Build message content for Claude API

        Args:
            context: Retrieved context with pages and images
            question: User's question
            include_images: Whether to include page images (multimodal)

        Returns:
            List of content blocks for the message
        """
        content = []

        # Add page images if multimodal
        if include_images and context["page_images"]:
            content.append(
                {
                    "type": "text",
                    "text": f"Here are the relevant pages from the thesis (pages {', '.join(map(str, context['pages']))}):",
                }
            )

            for page_num in context["pages"]:
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

        # Add the question
        content.append(
            {
                "type": "text",
                "text": f"\n\nQuestion: {question}\n\nPlease answer based on the thesis content above.",
            }
        )

        return content

    def ask(
        self,
        question: str,
        document_id: str | None = None,
        top_k: int = 3,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 4096,
        include_images: bool = True,
    ) -> dict[str, any]:
        """
        Ask a question about the document

        Args:
            question: User's question
            document_id: Optional document to search in
            top_k: Number of pages to retrieve
            model: Claude model to use
            max_tokens: Max tokens in response
            include_images: Whether to include page images (multimodal)

        Returns:
            Dict with answer, sources, and metadata
        """
        # Retrieve relevant context
        print(f"Retrieving context for: {question}")
        context = self.retrieve_context(query=question, top_k=top_k, document_id=document_id)

        print(f"Found {len(context['pages'])} relevant pages: {context['pages']}")

        # Build message content
        content = self._build_message_content(context, question, include_images)

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
        top_k: int = 3,
        model: str = "claude-3-5-sonnet-20241022",
        max_tokens: int = 4096,
        include_images: bool = True,
    ) -> AsyncIterator[dict[str, any]]:
        """
        Ask a question with streaming response

        Yields:
            Dicts with streaming data
        """
        # Retrieve relevant context
        print(f"Retrieving context for: {question}")
        context = self.retrieve_context(query=question, top_k=top_k, document_id=document_id)

        print(f"Found {len(context['pages'])} relevant pages: {context['pages']}")

        # Yield sources first
        yield {"type": "sources", "data": {"pages": context["pages"], "chunks": context["chunks"]}}

        # Build message content
        content = self._build_message_content(context, question, include_images)

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
