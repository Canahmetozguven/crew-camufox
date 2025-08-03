"""
Query Optimizer Module

Provides ML-enhanced query optimization, expansion, and refinement
for improved research results using semantic similarity and learning
from successful search patterns.
"""

import re
import json
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class QueryType(Enum):
    """Types of research queries"""

    ACADEMIC = "academic"
    NEWS = "news"
    TECHNICAL = "technical"
    MARKET = "market"
    GENERAL = "general"


class ExpansionMethod(Enum):
    """Methods for query expansion"""

    SEMANTIC = "semantic"
    SYNONYM = "synonym"
    DOMAIN_SPECIFIC = "domain_specific"
    TEMPORAL = "temporal"
    CONTEXTUAL = "contextual"


@dataclass
class QueryExpansion:
    """Represents a query expansion result"""

    original_query: str
    expanded_terms: List[str]
    synonyms: List[str]
    related_concepts: List[str]
    domain_terms: List[str]
    temporal_modifiers: List[str]
    confidence_score: float
    expansion_method: ExpansionMethod
    estimated_improvement: float


@dataclass
class QueryMetrics:
    """Metrics for query performance"""

    query: str
    query_type: QueryType
    search_engines_used: List[str]
    total_results: int
    relevant_results: int
    quality_score: float
    execution_time: float
    success_rate: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QueryPattern:
    """Represents a successful query pattern"""

    pattern_id: str
    query_structure: str
    domain: str
    success_rate: float
    avg_quality_score: float
    usage_count: int
    last_used: datetime
    keywords: List[str]
    effectiveness_score: float


class QueryOptimizer:
    """
    ML-enhanced query optimizer that learns from successful search patterns
    and provides intelligent query expansion and refinement.
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the query optimizer"""
        self.model_path = model_path or "data/ml_models/query_optimizer"
        self.query_history: List[QueryMetrics] = []
        self.successful_patterns: Dict[str, QueryPattern] = {}
        self.domain_vocabulary: Dict[str, Set[str]] = defaultdict(set)
        self.synonym_map: Dict[str, List[str]] = {}
        self.temporal_patterns: Dict[str, List[str]] = {}

        # Initialize ML components if available
        self.vectorizer = None
        self.pattern_model = None
        self._initialize_ml_components()

        # Load existing data
        self._load_optimization_data()

    def _initialize_ml_components(self):
        """Initialize ML components if sklearn is available"""
        if HAS_SKLEARN:
            self.vectorizer = TfidfVectorizer(
                max_features=1000, stop_words="english", ngram_range=(1, 2)
            )
            self.pattern_model = KMeans(n_clusters=5, random_state=42)

    def _load_optimization_data(self):
        """Load historical optimization data"""
        try:
            data_path = Path(self.model_path)
            if data_path.exists():
                with open(data_path / "query_history.json", "r") as f:
                    history_data = json.load(f)
                    self.query_history = [QueryMetrics(**item) for item in history_data]

                with open(data_path / "patterns.json", "r") as f:
                    pattern_data = json.load(f)
                    self.successful_patterns = {
                        k: QueryPattern(**v) for k, v in pattern_data.items()
                    }

                with open(data_path / "domain_vocab.json", "r") as f:
                    vocab_data = json.load(f)
                    self.domain_vocabulary = {k: set(v) for k, v in vocab_data.items()}
        except Exception as e:
            print(f"Warning: Could not load optimization data: {e}")

    def _save_optimization_data(self):
        """Save optimization data to disk"""
        try:
            data_path = Path(self.model_path)
            data_path.mkdir(parents=True, exist_ok=True)

            # Save query history
            with open(data_path / "query_history.json", "w") as f:
                history_data = [
                    {**metrics.__dict__, "timestamp": metrics.timestamp.isoformat()}
                    for metrics in self.query_history[-1000:]  # Keep last 1000
                ]
                json.dump(history_data, f, indent=2)

            # Save patterns
            with open(data_path / "patterns.json", "w") as f:
                pattern_data = {
                    k: {**v.__dict__, "last_used": v.last_used.isoformat()}
                    for k, v in self.successful_patterns.items()
                }
                json.dump(pattern_data, f, indent=2)

            # Save domain vocabulary
            with open(data_path / "domain_vocab.json", "w") as f:
                vocab_data = {k: list(v) for k, v in self.domain_vocabulary.items()}
                json.dump(vocab_data, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not save optimization data: {e}")

    def classify_query(self, query: str) -> QueryType:
        """Classify the type of research query"""
        query_lower = query.lower()

        # Academic indicators
        academic_terms = {
            "research",
            "study",
            "analysis",
            "paper",
            "journal",
            "academic",
            "peer-reviewed",
            "literature",
            "methodology",
            "hypothesis",
            "empirical",
            "systematic review",
            "meta-analysis",
        }

        # Technical indicators
        technical_terms = {
            "implementation",
            "algorithm",
            "framework",
            "api",
            "code",
            "programming",
            "software",
            "system",
            "architecture",
            "design",
            "specification",
            "documentation",
            "tutorial",
        }

        # News indicators
        news_terms = {
            "news",
            "breaking",
            "latest",
            "current",
            "recent",
            "today",
            "yesterday",
            "update",
            "announcement",
            "event",
            "happening",
        }

        # Market indicators
        market_terms = {
            "market",
            "industry",
            "competition",
            "business",
            "company",
            "financial",
            "revenue",
            "profit",
            "economic",
            "trend",
            "analysis",
            "forecast",
            "growth",
        }

        # Count matches
        academic_score = sum(1 for term in academic_terms if term in query_lower)
        technical_score = sum(1 for term in technical_terms if term in query_lower)
        news_score = sum(1 for term in news_terms if term in query_lower)
        market_score = sum(1 for term in market_terms if term in query_lower)

        # Determine category
        scores = {
            QueryType.ACADEMIC: academic_score,
            QueryType.TECHNICAL: technical_score,
            QueryType.NEWS: news_score,
            QueryType.MARKET: market_score,
        }

        max_score = max(scores.values())
        if max_score > 0:
            return max(scores, key=scores.get)
        else:
            return QueryType.GENERAL

    def extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Simple keyword extraction
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
        }

        # Extract words and phrases
        words = re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())
        keywords = [word for word in words if word not in stop_words]

        # Extract quoted phrases
        phrases = re.findall(r'"([^"]*)"', query)
        keywords.extend([phrase.lower() for phrase in phrases])

        return list(set(keywords))

    def generate_synonyms(self, term: str) -> List[str]:
        """Generate synonyms for a term"""
        # Basic synonym mapping (in production, use WordNet or similar)
        synonym_db = {
            "artificial intelligence": ["AI", "machine learning", "neural networks"],
            "machine learning": ["ML", "AI", "deep learning", "neural networks"],
            "research": ["study", "investigation", "analysis", "examination"],
            "analysis": ["examination", "evaluation", "assessment", "review"],
            "development": ["creation", "implementation", "building", "construction"],
            "technology": ["tech", "innovation", "advancement", "system"],
            "industry": ["sector", "market", "business", "field"],
            "company": ["corporation", "business", "organization", "firm"],
        }

        synonyms = []
        term_lower = term.lower()

        # Direct lookup
        if term_lower in synonym_db:
            synonyms.extend(synonym_db[term_lower])

        # Partial matches
        for key, values in synonym_db.items():
            if term_lower in key or any(term_lower in v for v in values):
                synonyms.extend(values)
                if term_lower not in synonyms:
                    synonyms.append(key)

        return list(set(synonyms))

    def expand_query(
        self, query: str, method: ExpansionMethod = ExpansionMethod.SEMANTIC
    ) -> QueryExpansion:
        """Expand query with additional relevant terms"""
        query_type = self.classify_query(query)
        keywords = self.extract_keywords(query)

        expanded_terms = []
        synonyms = []
        related_concepts = []
        domain_terms = []
        temporal_modifiers = []

        # Generate synonyms for keywords
        for keyword in keywords:
            keyword_synonyms = self.generate_synonyms(keyword)
            synonyms.extend(keyword_synonyms)

        # Add domain-specific terms
        domain_key = query_type.value
        if domain_key in self.domain_vocabulary:
            domain_terms = list(self.domain_vocabulary[domain_key])[:5]

        # Add temporal modifiers based on query type
        if query_type == QueryType.NEWS:
            temporal_modifiers = ["recent", "latest", "2024", "current"]
        elif query_type == QueryType.ACADEMIC:
            temporal_modifiers = ["2023", "2024", "recent studies"]

        # Add related concepts based on successful patterns
        for pattern in self.successful_patterns.values():
            if any(keyword in pattern.keywords for keyword in keywords):
                related_concepts.extend(pattern.keywords[:3])

        # Combine all expansions
        expanded_terms = list(set(synonyms[:3] + domain_terms[:3] + related_concepts[:3]))

        # Calculate confidence based on available data
        confidence_score = 0.7  # Base confidence
        if domain_terms:
            confidence_score += 0.1
        if related_concepts:
            confidence_score += 0.1
        if len(synonyms) > 2:
            confidence_score += 0.1

        confidence_score = min(confidence_score, 1.0)

        return QueryExpansion(
            original_query=query,
            expanded_terms=expanded_terms,
            synonyms=list(set(synonyms)),
            related_concepts=list(set(related_concepts)),
            domain_terms=domain_terms,
            temporal_modifiers=temporal_modifiers,
            confidence_score=confidence_score,
            expansion_method=method,
            estimated_improvement=0.2 + (confidence_score * 0.3),
        )

    def optimize_query(self, query: str) -> Tuple[str, QueryExpansion]:
        """Optimize a query for better search results"""
        expansion = self.expand_query(query)

        # Build optimized query
        optimized_parts = [query]

        # Add high-confidence expanded terms
        if expansion.confidence_score > 0.8:
            optimized_parts.extend(expansion.expanded_terms[:2])
        elif expansion.confidence_score > 0.6:
            optimized_parts.extend(expansion.expanded_terms[:1])

        # Add domain terms for specific query types
        if expansion.domain_terms and len(expansion.domain_terms) > 0:
            optimized_parts.append(expansion.domain_terms[0])

        # Add temporal modifiers for time-sensitive queries
        if expansion.temporal_modifiers:
            optimized_parts.append(expansion.temporal_modifiers[0])

        optimized_query = " ".join(optimized_parts)

        return optimized_query, expansion

    def learn_from_results(self, query: str, metrics: QueryMetrics):
        """Learn from query results to improve future optimization"""
        self.query_history.append(metrics)

        # Update domain vocabulary
        keywords = self.extract_keywords(query)
        domain = metrics.query_type.value
        self.domain_vocabulary[domain].update(keywords)

        # Create or update pattern if successful
        if metrics.success_rate > 0.7 and metrics.quality_score > 0.7:
            pattern_key = f"{domain}_{len(keywords)}"

            if pattern_key in self.successful_patterns:
                pattern = self.successful_patterns[pattern_key]
                pattern.usage_count += 1
                pattern.last_used = datetime.now()
                pattern.success_rate = pattern.success_rate * 0.9 + metrics.success_rate * 0.1
                pattern.avg_quality_score = (
                    pattern.avg_quality_score * 0.9 + metrics.quality_score * 0.1
                )
            else:
                self.successful_patterns[pattern_key] = QueryPattern(
                    pattern_id=pattern_key,
                    query_structure=self._extract_query_structure(query),
                    domain=domain,
                    success_rate=metrics.success_rate,
                    avg_quality_score=metrics.quality_score,
                    usage_count=1,
                    last_used=datetime.now(),
                    keywords=keywords,
                    effectiveness_score=metrics.success_rate * metrics.quality_score,
                )

        # Periodic cleanup and save
        if len(self.query_history) % 10 == 0:
            self._cleanup_old_data()
            self._save_optimization_data()

    def _extract_query_structure(self, query: str) -> str:
        """Extract structural pattern from query"""
        # Simple pattern extraction
        words = query.lower().split()
        if len(words) <= 3:
            return "short_query"
        elif len(words) <= 7:
            return "medium_query"
        else:
            return "long_query"

    def _cleanup_old_data(self):
        """Clean up old data to prevent memory bloat"""
        # Keep only recent history
        cutoff_date = datetime.now() - timedelta(days=30)
        self.query_history = [
            metrics for metrics in self.query_history if metrics.timestamp > cutoff_date
        ]

        # Remove unused patterns
        for pattern_id in list(self.successful_patterns.keys()):
            pattern = self.successful_patterns[pattern_id]
            if pattern.last_used < cutoff_date and pattern.usage_count < 3:
                del self.successful_patterns[pattern_id]

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        recent_metrics = [
            m for m in self.query_history if m.timestamp > datetime.now() - timedelta(days=7)
        ]

        if not recent_metrics:
            return {"status": "no_recent_data"}

        avg_quality = np.mean([m.quality_score for m in recent_metrics])
        avg_success = np.mean([m.success_rate for m in recent_metrics])

        return {
            "total_queries_optimized": len(self.query_history),
            "recent_queries": len(recent_metrics),
            "average_quality_score": float(avg_quality),
            "average_success_rate": float(avg_success),
            "successful_patterns": len(self.successful_patterns),
            "domain_vocabularies": len(self.domain_vocabulary),
            "improvement_trend": self._calculate_improvement_trend(),
        }

    def _calculate_improvement_trend(self) -> float:
        """Calculate improvement trend over time"""
        if len(self.query_history) < 10:
            return 0.0

        recent_scores = [m.quality_score for m in self.query_history[-10:]]
        older_scores = [m.quality_score for m in self.query_history[-20:-10]]

        if not older_scores:
            return 0.0

        recent_avg = np.mean(recent_scores)
        older_avg = np.mean(older_scores)

        return float(recent_avg - older_avg)


# Fallback implementations for when sklearn is not available
if not HAS_SKLEARN:

    class MockTfidfVectorizer:
        def __init__(self, **kwargs):
            pass

        def fit_transform(self, documents):
            return np.random.rand(len(documents), 100)

    class MockKMeans:
        def __init__(self, **kwargs):
            pass

        def fit(self, X):
            return self

        def predict(self, X):
            return np.random.randint(0, 5, X.shape[0])
