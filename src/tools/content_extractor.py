import re
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

from crewai_tools import BaseTool
from pydantic import BaseModel, Field


class ContentExtractionResult(BaseModel):
    """Result model for content extraction"""

    url: str
    title: str
    content: str
    summary: str
    key_topics: List[str]
    metadata: Dict[str, Any]
    credibility_score: float
    extraction_timestamp: datetime


class ContentExtractorTool(BaseTool):
    """Tool for extracting and processing web content"""

    name: str = "content_extractor"
    description: str = "Extracts, cleans, and analyzes content from web pages"

    def __init__(self):
        super().__init__()
        self.browser_tool = None  # Will be injected

    async def _run(self, url: str, **kwargs) -> ContentExtractionResult:
        """Extract content from a URL"""
        if not self.browser_tool:
            raise ValueError("Browser tool not initialized")

        try:
            # Extract content using browser tool
            content_data = await self.browser_tool.extract_page_content(url)

            if not content_data.get("success", False):
                raise Exception(
                    f"Failed to extract content: {content_data.get('error', 'Unknown error')}"
                )

            # Create result object
            result = ContentExtractionResult(
                url=url,
                title=content_data.get("title", ""),
                content=content_data.get("content", ""),
                summary=self._generate_summary(content_data.get("content", "")),
                key_topics=self._extract_topics(content_data.get("content", "")),
                metadata={
                    "word_count": content_data.get("word_count", 0),
                    "source_type": content_data.get("source_type", "unknown"),
                    "meta_description": content_data.get("meta_description", ""),
                    "meta_keywords": content_data.get("meta_keywords", ""),
                },
                credibility_score=content_data.get("credibility_score", 0.5),
                extraction_timestamp=datetime.now(),
            )

            return result

        except Exception as e:
            # Return empty result on error
            return ContentExtractionResult(
                url=url,
                title="",
                content="",
                summary=f"Extraction failed: {str(e)}",
                key_topics=[],
                metadata={},
                credibility_score=0.0,
                extraction_timestamp=datetime.now(),
            )

    def _generate_summary(self, content: str, max_sentences: int = 3) -> str:
        """Generate a summary from content"""
        if not content:
            return ""

        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        # Take first few sentences
        summary_sentences = sentences[:max_sentences]
        return ". ".join(summary_sentences) + "." if summary_sentences else content[:200] + "..."

    def _extract_topics(self, content: str, max_topics: int = 10) -> List[str]:
        """Extract key topics from content"""
        if not content:
            return []

        # Simple keyword extraction
        words = re.findall(r"\b[a-zA-Z]{4,}\b", content.lower())

        # Common stop words to filter out
        stop_words = {
            "this",
            "that",
            "with",
            "have",
            "will",
            "from",
            "they",
            "know",
            "want",
            "been",
            "good",
            "much",
            "some",
            "time",
            "very",
            "when",
            "come",
            "here",
            "just",
            "like",
            "long",
            "make",
            "many",
            "over",
            "such",
            "take",
            "than",
            "them",
            "well",
            "were",
            "more",
            "said",
            "each",
            "which",
            "their",
            "would",
            "there",
            "could",
            "other",
        }

        # Count word frequency
        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and return top topics
        sorted_topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [topic[0] for topic in sorted_topics[:max_topics]]


class SearchTool(BaseTool):
    """Tool for searching and extracting content from multiple sources"""

    name: str = "search_tool"
    description: str = "Searches for information across multiple sources and extracts content"

    def __init__(self):
        super().__init__()
        self.browser_tool = None  # Will be injected

    async def _run(
        self, query: str, max_results: int = 10, **kwargs
    ) -> List[ContentExtractionResult]:
        """Search for content and extract from multiple sources"""
        if not self.browser_tool:
            raise ValueError("Browser tool not initialized")

        try:
            # Search using browser tool
            search_results = await self.browser_tool.search_and_extract([query], max_results)

            # Convert to ContentExtractionResult objects
            results = []
            for result_data in search_results:
                if result_data.get("success", False):
                    result = ContentExtractionResult(
                        url=result_data.get("url", ""),
                        title=result_data.get("title", ""),
                        content=result_data.get("content", ""),
                        summary=self._generate_summary(result_data.get("content", "")),
                        key_topics=self._extract_topics(result_data.get("content", "")),
                        metadata={
                            "word_count": result_data.get("word_count", 0),
                            "source_type": result_data.get("source_type", "unknown"),
                            "meta_description": result_data.get("meta_description", ""),
                            "meta_keywords": result_data.get("meta_keywords", ""),
                        },
                        credibility_score=result_data.get("credibility_score", 0.5),
                        extraction_timestamp=result_data.get(
                            "extraction_timestamp", datetime.now()
                        ),
                    )
                    results.append(result)

            return results

        except Exception as e:
            print(f"Search error: {str(e)}")
            return []

    def _generate_summary(self, content: str, max_sentences: int = 2) -> str:
        """Generate a summary from content"""
        if not content:
            return ""

        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        summary_sentences = sentences[:max_sentences]
        return ". ".join(summary_sentences) + "." if summary_sentences else content[:150] + "..."

    def _extract_topics(self, content: str, max_topics: int = 5) -> List[str]:
        """Extract key topics from content"""
        if not content:
            return []

        # Simple keyword extraction
        words = re.findall(r"\b[a-zA-Z]{4,}\b", content.lower())

        stop_words = {
            "this",
            "that",
            "with",
            "have",
            "will",
            "from",
            "they",
            "know",
            "want",
            "been",
            "good",
            "much",
            "some",
            "time",
            "very",
            "when",
        }

        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1

        sorted_topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [topic[0] for topic in sorted_topics[:max_topics]]


class DeepLinkTool(BaseTool):
    """Tool for following links to deeper levels of research"""

    name: str = "deep_link_tool"
    description: str = "Follows links from sources to find additional relevant information"

    def __init__(self):
        super().__init__()
        self.browser_tool = None  # Will be injected

    async def _run(
        self, start_url: str, max_depth: int = 2, max_links: int = 5, **kwargs
    ) -> List[ContentExtractionResult]:
        """Follow links from a starting URL to specified depth"""
        if not self.browser_tool:
            raise ValueError("Browser tool not initialized")

        try:
            # Use browser tool to follow links
            deep_results = await self.browser_tool.follow_links_depth(
                start_url, max_depth, max_links
            )

            # Convert to ContentExtractionResult objects
            results = []
            for result_data in deep_results:
                if result_data.get("success", False):
                    result = ContentExtractionResult(
                        url=result_data.get("url", ""),
                        title=result_data.get("title", ""),
                        content=result_data.get("content", ""),
                        summary=self._generate_summary(result_data.get("content", "")),
                        key_topics=self._extract_topics(result_data.get("content", "")),
                        metadata={
                            "word_count": result_data.get("word_count", 0),
                            "source_type": result_data.get("source_type", "unknown"),
                            "meta_description": result_data.get("meta_description", ""),
                            "meta_keywords": result_data.get("meta_keywords", ""),
                        },
                        credibility_score=result_data.get("credibility_score", 0.5),
                        extraction_timestamp=result_data.get(
                            "extraction_timestamp", datetime.now()
                        ),
                    )
                    results.append(result)

            return results

        except Exception as e:
            print(f"Deep link error: {str(e)}")
            return []

    def _generate_summary(self, content: str, max_sentences: int = 2) -> str:
        """Generate a summary from content"""
        if not content:
            return ""

        sentences = re.split(r"[.!?]+", content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        summary_sentences = sentences[:max_sentences]
        return ". ".join(summary_sentences) + "." if summary_sentences else content[:150] + "..."

    def _extract_topics(self, content: str, max_topics: int = 5) -> List[str]:
        """Extract key topics from content"""
        if not content:
            return []

        words = re.findall(r"\b[a-zA-Z]{4,}\b", content.lower())

        stop_words = {
            "this",
            "that",
            "with",
            "have",
            "will",
            "from",
            "they",
            "know",
            "want",
            "been",
            "good",
            "much",
            "some",
            "time",
            "very",
            "when",
        }

        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1

        sorted_topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [topic[0] for topic in sorted_topics[:max_topics]]
