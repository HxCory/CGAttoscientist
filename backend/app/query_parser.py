"""
Query parser for extracting page numbers and chapter references from user queries.
"""

import re


class QueryParser:
    """Parse user queries to extract explicit page references and chapter mentions"""

    def __init__(self):
        # Patterns for page number extraction
        self.page_patterns = [
            # "PDF page 135" or "pdf page 135"
            r"pdf\s+page\s+(\d+)",
            # "page 135" (generic)
            r"(?:^|\s)page\s+(\d+)(?:\s|$|[,.])",
            # "thesis page 113"
            r"thesis\s+page\s+(\d+)",
            # "on page 135" or "at page 135"
            r"(?:on|at)\s+page\s+(\d+)",
            # "p. 135" or "p.135"
            r"p\.\s*(\d+)",
            # "pages 10-15" or "pages 10 to 15" (range)
            r"pages?\s+(\d+)(?:\s*-\s*|\s+to\s+)(\d+)",
        ]

        # Patterns for chapter references
        self.chapter_patterns = [
            # "Chapter 1" or "chapter 1"
            r"chapter\s+(\d+)",
            # "Ch. 1" or "ch 1"
            r"ch\.?\s+(\d+)",
            # "introduction chapter" or "intro chapter"
            r"(introduction|intro)\s+chapter",
            # Section references like "1.1" or "section 1.1"
            r"(?:section\s+)?(\d+)\.(\d+)",
        ]

    def extract_page_numbers(self, query: str) -> dict:
        """
        Extract page number references from query

        Args:
            query: User's question

        Returns:
            Dict with:
                - pdf_pages: List of explicit PDF page numbers mentioned
                - thesis_pages: List of explicit thesis page numbers mentioned
                - page_range: Tuple of (start, end) if a range is specified
                - has_page_reference: Boolean indicating if query mentions specific pages
        """
        query_lower = query.lower()
        result = {
            "pdf_pages": [],
            "thesis_pages": [],
            "page_range": None,
            "has_page_reference": False,
        }

        # Check for page ranges first (e.g., "pages 10-15")
        range_match = re.search(r"pages?\s+(\d+)(?:\s*-\s*|\s+to\s+)(\d+)", query_lower)
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            result["page_range"] = (start, end)
            result["has_page_reference"] = True
            return result

        # Check for PDF page references
        for pattern in [r"pdf\s+page\s+(\d+)", r"pdf\s+p\.\s*(\d+)"]:
            matches = re.findall(pattern, query_lower)
            if matches:
                result["pdf_pages"].extend([int(m) for m in matches])
                result["has_page_reference"] = True

        # Check for thesis page references
        for pattern in [r"thesis\s+page\s+(\d+)", r"thesis\s+p\.\s*(\d+)"]:
            matches = re.findall(pattern, query_lower)
            if matches:
                result["thesis_pages"].extend([int(m) for m in matches])
                result["has_page_reference"] = True

        # Check for generic "page X" references (assume PDF page if not specified)
        if not result["pdf_pages"] and not result["thesis_pages"]:
            generic_pattern = r"(?:on|at|see)?\s*page\s+(\d+)"
            matches = re.findall(generic_pattern, query_lower)
            if matches:
                result["pdf_pages"].extend([int(m) for m in matches])
                result["has_page_reference"] = True

        return result

    def extract_chapter_info(self, query: str) -> dict:
        """
        Extract chapter references from query

        Args:
            query: User's question

        Returns:
            Dict with:
                - chapter_number: Integer chapter number if found
                - section_number: Tuple of (chapter, section) for section refs like "1.1"
                - has_chapter_reference: Boolean
        """
        query_lower = query.lower()
        result = {
            "chapter_number": None,
            "section_number": None,
            "has_chapter_reference": False,
        }

        # Check for "Chapter X"
        chapter_match = re.search(r"chapter\s+(\d+)", query_lower)
        if chapter_match:
            result["chapter_number"] = int(chapter_match.group(1))
            result["has_chapter_reference"] = True

        # Check for "introduction chapter"
        if re.search(r"(introduction|intro)\s+chapter", query_lower):
            result["chapter_number"] = 1  # Usually Chapter 1
            result["has_chapter_reference"] = True

        # Check for section references like "section 1.1" or "1.1"
        section_match = re.search(r"(?:section\s+)?(\d+)\.(\d+)", query_lower)
        if section_match:
            chapter = int(section_match.group(1))
            section = int(section_match.group(2))
            result["section_number"] = (chapter, section)
            result["has_chapter_reference"] = True
            if not result["chapter_number"]:
                result["chapter_number"] = chapter

        return result

    def parse_query(self, query: str) -> dict:
        """
        Parse user query for all references

        Args:
            query: User's question

        Returns:
            Dict combining page and chapter information
        """
        page_info = self.extract_page_numbers(query)
        chapter_info = self.extract_chapter_info(query)

        return {
            **page_info,
            **chapter_info,
            "original_query": query,
        }
