"""Utility helper functions for research operations."""

import re
import urllib.parse
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse


def extract_text_content(content: str) -> str:
    """Extract and clean text content."""
    if not content:
        return ""

    # Remove extra whitespace and normalize
    cleaned = re.sub(r"\s+", " ", content.strip())
    return cleaned


def detect_source_type(url: str) -> str:
    """Detect the type of source from URL."""
    if not url:
        return "unknown"

    domain = urlparse(url).netloc.lower()

    # Academic sources
    if any(domain.endswith(suffix) for suffix in [".edu", ".ac.uk", ".ac.jp"]):
        return "academic"

    # Government sources
    if any(domain.endswith(suffix) for suffix in [".gov", ".gov.uk", ".gov.au"]):
        return "government"

    # News sources
    news_domains = ["bbc.com", "cnn.com", "reuters.com", "ap.org", "npr.org"]
    if any(news in domain for news in news_domains):
        return "news"

    # Wikipedia
    if "wikipedia.org" in domain:
        return "encyclopedia"

    return "general"


def calculate_credibility_score(url: str, content: str) -> float:
    """Calculate credibility score for a source."""
    score = 0.5  # Base score

    source_type = detect_source_type(url)
    type_scores = {
        "academic": 0.9,
        "government": 0.85,
        "news": 0.8,
        "encyclopedia": 0.75,
        "general": 0.5,
    }

    score = type_scores.get(source_type, 0.5)

    # Adjust based on content quality indicators
    if content:
        content_lower = content.lower()

        # Positive indicators
        if any(word in content_lower for word in ["study", "research", "analysis"]):
            score += 0.1

        if any(word in content_lower for word in ["doi:", "pmid:", "arxiv:"]):
            score += 0.15

        # Negative indicators
        if any(word in content_lower for word in ["opinion", "blog", "personal"]):
            score -= 0.1

    return min(max(score, 0.0), 1.0)


def extract_key_topics(text: str) -> List[str]:
    """Extract key topics from text."""
    if not text:
        return []

    # Simple keyword extraction (can be enhanced with NLP libraries)
    words = re.findall(r"\b[A-Z][a-z]+\b", text)

    # Filter common words
    common_words = {"The", "This", "That", "With", "From", "They", "When", "Where"}
    topics = [word for word in words if word not in common_words]

    # Return unique topics, limited to top 10
    return list(dict.fromkeys(topics))[:10]


def generate_summary(text: str, max_length: int = 200) -> str:
    """Generate a summary of the text."""
    if not text or len(text) <= max_length:
        return text

    # Simple extractive summarization - take first sentences up to max_length
    sentences = re.split(r"[.!?]+", text)
    summary = ""

    for sentence in sentences:
        if len(summary + sentence) <= max_length:
            summary += sentence.strip() + ". "
        else:
            break

    return summary.strip()


def validate_url(url: str) -> bool:
    """Validate if URL is properly formatted."""
    if not url:
        return False

    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def clean_url(url: str) -> str:
    """Clean and normalize URL."""
    if not url:
        return ""

    # Remove common tracking parameters
    tracking_params = [
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_content",
        "utm_term",
        "fbclid",
        "gclid",
        "ref",
        "source",
    ]

    parsed = urlparse(url)
    query_params = urllib.parse.parse_qs(parsed.query)

    # Remove tracking parameters
    clean_params = {k: v for k, v in query_params.items() if k not in tracking_params}

    # Rebuild URL
    clean_query = urllib.parse.urlencode(clean_params, doseq=True)
    clean_parsed = parsed._replace(query=clean_query)

    return urllib.parse.urlunparse(clean_parsed)
