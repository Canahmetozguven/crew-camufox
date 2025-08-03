"""
Source Quality Predictor Module

Provides ML-enhanced source quality assessment, credibility prediction,
and content quality scoring to improve research result filtering.
"""

import json
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any, Union
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import re
from pathlib import Path

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

    # Create mock numpy for basic functionality
    class MockNumpy:
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0

        @staticmethod
        def std(data):
            if not data:
                return 0
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            return variance**0.5

        @staticmethod
        def array(data):
            return data

    np = MockNumpy()


class QualityDimension(Enum):
    """Dimensions of source quality assessment"""

    AUTHORITY = "authority"
    ACCURACY = "accuracy"
    OBJECTIVITY = "objectivity"
    CURRENCY = "currency"
    COVERAGE = "coverage"
    RELEVANCE = "relevance"


class ContentType(Enum):
    """Types of content for quality assessment"""

    ACADEMIC_PAPER = "academic_paper"
    NEWS_ARTICLE = "news_article"
    BLOG_POST = "blog_post"
    GOVERNMENT_DOC = "government_doc"
    CORPORATE_INFO = "corporate_info"
    SOCIAL_MEDIA = "social_media"
    TECHNICAL_DOC = "technical_doc"


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for a source"""

    source_url: str
    content_type: ContentType
    authority_score: float
    accuracy_score: float
    objectivity_score: float
    currency_score: float
    coverage_score: float
    relevance_score: float
    overall_quality: float
    confidence_interval: Tuple[float, float]
    quality_factors: Dict[str, float]
    assessment_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QualityFeatures:
    """Features extracted for quality prediction"""

    domain_authority: float
    content_length: int
    citation_count: int
    publication_date: Optional[datetime]
    author_credentials: int  # 0-5 scale
    peer_reviewed: bool
    factual_claims: int
    opinion_indicators: int
    grammatical_errors: int
    external_links: int
    multimedia_elements: int
    update_frequency: float
    social_engagement: int


@dataclass
class TrainingExample:
    """Training example for quality prediction model"""

    features: QualityFeatures
    quality_metrics: QualityMetrics
    human_rating: Optional[float] = None
    feedback_score: Optional[float] = None


class SourceQualityPredictor:
    """
    ML-enhanced source quality predictor that learns from user feedback
    and research outcomes to improve quality assessment accuracy.
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the source quality predictor"""
        self.model_path = model_path or "data/ml_models/quality_predictor"
        self.training_data: List[TrainingExample] = []
        self.quality_baselines: Dict[ContentType, Dict[str, float]] = {}
        self.domain_reputation: Dict[str, float] = {}
        self.feature_weights: Dict[str, float] = {}

        # Initialize default weights
        self._initialize_feature_weights()

        # Load existing model data
        self._load_model_data()

    def _initialize_feature_weights(self):
        """Initialize feature weights for quality prediction"""
        self.feature_weights = {
            "domain_authority": 0.15,
            "content_length": 0.08,
            "citation_count": 0.12,
            "author_credentials": 0.10,
            "peer_reviewed": 0.15,
            "factual_claims": 0.08,
            "opinion_indicators": -0.05,  # Negative weight
            "grammatical_errors": -0.07,  # Negative weight
            "external_links": 0.06,
            "multimedia_elements": 0.03,
            "update_frequency": 0.05,
            "social_engagement": 0.04,
            "currency_factor": 0.10,
            "coverage_depth": 0.06,
        }

    def _load_model_data(self):
        """Load existing model data and baselines"""
        try:
            data_path = Path(self.model_path)
            if data_path.exists():
                # Load training data
                if (data_path / "training_data.json").exists():
                    with open(data_path / "training_data.json", "r") as f:
                        training_data = json.load(f)
                        # Convert to TrainingExample objects (simplified)
                        self.training_data = training_data[-1000:]  # Keep recent data

                # Load quality baselines
                if (data_path / "baselines.json").exists():
                    with open(data_path / "baselines.json", "r") as f:
                        baselines_data = json.load(f)
                        self.quality_baselines = {
                            ContentType(k): v for k, v in baselines_data.items()
                        }

                # Load domain reputation
                if (data_path / "domain_reputation.json").exists():
                    with open(data_path / "domain_reputation.json", "r") as f:
                        self.domain_reputation = json.load(f)

                # Load feature weights
                if (data_path / "feature_weights.json").exists():
                    with open(data_path / "feature_weights.json", "r") as f:
                        self.feature_weights.update(json.load(f))

        except Exception as e:
            print(f"Warning: Could not load model data: {e}")

    def _save_model_data(self):
        """Save model data and learned parameters"""
        try:
            data_path = Path(self.model_path)
            data_path.mkdir(parents=True, exist_ok=True)

            # Save feature weights
            with open(data_path / "feature_weights.json", "w") as f:
                json.dump(self.feature_weights, f, indent=2)

            # Save quality baselines
            with open(data_path / "baselines.json", "w") as f:
                baselines_data = {k.value: v for k, v in self.quality_baselines.items()}
                json.dump(baselines_data, f, indent=2)

            # Save domain reputation
            with open(data_path / "domain_reputation.json", "w") as f:
                json.dump(self.domain_reputation, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not save model data: {e}")

    def extract_features(
        self, url: str, content: str, metadata: Optional[Dict] = None
    ) -> QualityFeatures:
        """Extract quality features from source content"""
        metadata = metadata or {}

        # Domain analysis
        domain = self._extract_domain(url)
        domain_authority = self.domain_reputation.get(domain, 0.5)

        # Content analysis
        content_length = len(content)
        citation_count = self._count_citations(content)
        factual_claims = self._count_factual_claims(content)
        opinion_indicators = self._count_opinion_indicators(content)
        grammatical_errors = self._estimate_grammar_errors(content)
        external_links = self._count_external_links(content)
        multimedia_elements = self._count_multimedia(content)

        # Authority indicators
        author_credentials = self._assess_author_credentials(content, metadata)
        peer_reviewed = self._is_peer_reviewed(content, metadata)

        # Temporal factors
        publication_date = self._extract_publication_date(content, metadata)
        update_frequency = self._estimate_update_frequency(metadata)

        # Social indicators
        social_engagement = metadata.get("social_shares", 0)

        return QualityFeatures(
            domain_authority=domain_authority,
            content_length=content_length,
            citation_count=citation_count,
            publication_date=publication_date,
            author_credentials=author_credentials,
            peer_reviewed=peer_reviewed,
            factual_claims=factual_claims,
            opinion_indicators=opinion_indicators,
            grammatical_errors=grammatical_errors,
            external_links=external_links,
            multimedia_elements=multimedia_elements,
            update_frequency=update_frequency,
            social_engagement=social_engagement,
        )

    def predict_quality(
        self, features: QualityFeatures, content_type: ContentType
    ) -> QualityMetrics:
        """Predict quality metrics based on extracted features"""

        # Calculate individual dimension scores
        authority_score = self._calculate_authority_score(features, content_type)
        accuracy_score = self._calculate_accuracy_score(features, content_type)
        objectivity_score = self._calculate_objectivity_score(features, content_type)
        currency_score = self._calculate_currency_score(features, content_type)
        coverage_score = self._calculate_coverage_score(features, content_type)
        relevance_score = 0.7  # Default relevance (requires context)

        # Calculate overall quality using weighted average
        dimension_scores = {
            QualityDimension.AUTHORITY: authority_score,
            QualityDimension.ACCURACY: accuracy_score,
            QualityDimension.OBJECTIVITY: objectivity_score,
            QualityDimension.CURRENCY: currency_score,
            QualityDimension.COVERAGE: coverage_score,
            QualityDimension.RELEVANCE: relevance_score,
        }

        # Weight dimensions based on content type
        dimension_weights = self._get_dimension_weights(content_type)
        overall_quality = sum(
            dimension_scores[dim] * weight for dim, weight in dimension_weights.items()
        )

        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            overall_quality, features, content_type
        )

        # Extract quality factors
        quality_factors = {
            "domain_reputation": features.domain_authority,
            "content_depth": min(features.content_length / 1000, 1.0),
            "citation_density": min(features.citation_count / 10, 1.0),
            "peer_review_status": 1.0 if features.peer_reviewed else 0.0,
            "author_authority": features.author_credentials / 5.0,
            "objectivity_indicators": max(0, 1.0 - features.opinion_indicators / 10),
            "content_freshness": currency_score,
        }

        return QualityMetrics(
            source_url="",  # Will be set by caller
            content_type=content_type,
            authority_score=authority_score,
            accuracy_score=accuracy_score,
            objectivity_score=objectivity_score,
            currency_score=currency_score,
            coverage_score=coverage_score,
            relevance_score=relevance_score,
            overall_quality=overall_quality,
            confidence_interval=confidence_interval,
            quality_factors=quality_factors,
        )

    def _calculate_authority_score(
        self, features: QualityFeatures, content_type: ContentType
    ) -> float:
        """Calculate authority score based on features"""
        score = 0.0

        # Domain authority contribution
        score += features.domain_authority * 0.4

        # Author credentials contribution
        score += (features.author_credentials / 5.0) * 0.3

        # Peer review status
        if features.peer_reviewed:
            score += 0.2

        # Citation count (normalized)
        citation_factor = min(features.citation_count / 20, 1.0)
        score += citation_factor * 0.1

        return min(score, 1.0)

    def _calculate_accuracy_score(
        self, features: QualityFeatures, content_type: ContentType
    ) -> float:
        """Calculate accuracy score based on features"""
        score = 0.7  # Base score

        # Positive factors
        if features.peer_reviewed:
            score += 0.15

        citation_factor = min(features.citation_count / 10, 0.1)
        score += citation_factor

        factual_density = min(features.factual_claims / features.content_length * 1000, 0.1)
        score += factual_density

        # Negative factors
        grammar_penalty = min(features.grammatical_errors / 100, 0.1)
        score -= grammar_penalty

        return max(min(score, 1.0), 0.0)

    def _calculate_objectivity_score(
        self, features: QualityFeatures, content_type: ContentType
    ) -> float:
        """Calculate objectivity score based on features"""
        score = 0.8  # Base score

        # Opinion indicators reduce objectivity
        opinion_penalty = min(features.opinion_indicators / 20, 0.3)
        score -= opinion_penalty

        # Peer review improves objectivity
        if features.peer_reviewed:
            score += 0.1

        # Academic content is generally more objective
        if content_type == ContentType.ACADEMIC_PAPER:
            score += 0.1
        elif content_type == ContentType.BLOG_POST:
            score -= 0.1
        elif content_type == ContentType.SOCIAL_MEDIA:
            score -= 0.2

        return max(min(score, 1.0), 0.0)

    def _calculate_currency_score(
        self, features: QualityFeatures, content_type: ContentType
    ) -> float:
        """Calculate currency (freshness) score based on features"""
        if not features.publication_date:
            return 0.5  # Unknown date

        days_old = (datetime.now() - features.publication_date).days

        # Different decay rates for different content types
        if content_type == ContentType.NEWS_ARTICLE:
            # News becomes stale quickly
            currency = max(0, 1.0 - days_old / 30)
        elif content_type == ContentType.ACADEMIC_PAPER:
            # Academic papers have longer shelf life
            currency = max(0, 1.0 - days_old / 365)
        elif content_type == ContentType.TECHNICAL_DOC:
            # Technical docs moderate shelf life
            currency = max(0, 1.0 - days_old / 180)
        else:
            # General content
            currency = max(0, 1.0 - days_old / 90)

        # Update frequency bonus
        if features.update_frequency > 0:
            currency += min(features.update_frequency * 0.1, 0.2)

        return min(currency, 1.0)

    def _calculate_coverage_score(
        self, features: QualityFeatures, content_type: ContentType
    ) -> float:
        """Calculate coverage (depth) score based on features"""
        score = 0.0

        # Content length factor
        length_factor = min(features.content_length / 2000, 1.0)
        score += length_factor * 0.4

        # Citation depth
        citation_factor = min(features.citation_count / 15, 1.0)
        score += citation_factor * 0.3

        # External links (show research breadth)
        link_factor = min(features.external_links / 10, 1.0)
        score += link_factor * 0.2

        # Multimedia elements
        media_factor = min(features.multimedia_elements / 5, 1.0)
        score += media_factor * 0.1

        return min(score, 1.0)

    def _get_dimension_weights(self, content_type: ContentType) -> Dict[QualityDimension, float]:
        """Get dimension weights based on content type"""
        if content_type == ContentType.ACADEMIC_PAPER:
            return {
                QualityDimension.AUTHORITY: 0.25,
                QualityDimension.ACCURACY: 0.25,
                QualityDimension.OBJECTIVITY: 0.20,
                QualityDimension.CURRENCY: 0.10,
                QualityDimension.COVERAGE: 0.15,
                QualityDimension.RELEVANCE: 0.05,
            }
        elif content_type == ContentType.NEWS_ARTICLE:
            return {
                QualityDimension.AUTHORITY: 0.20,
                QualityDimension.ACCURACY: 0.30,
                QualityDimension.OBJECTIVITY: 0.25,
                QualityDimension.CURRENCY: 0.20,
                QualityDimension.COVERAGE: 0.05,
                QualityDimension.RELEVANCE: 0.00,
            }
        else:
            # Default weights
            return {
                QualityDimension.AUTHORITY: 0.20,
                QualityDimension.ACCURACY: 0.20,
                QualityDimension.OBJECTIVITY: 0.15,
                QualityDimension.CURRENCY: 0.15,
                QualityDimension.COVERAGE: 0.15,
                QualityDimension.RELEVANCE: 0.15,
            }

    def _calculate_confidence_interval(
        self, quality_score: float, features: QualityFeatures, content_type: ContentType
    ) -> Tuple[float, float]:
        """Calculate confidence interval for quality prediction"""
        # Base confidence based on available features
        confidence = 0.8

        # Reduce confidence for missing information
        if not features.publication_date:
            confidence -= 0.1
        if features.author_credentials == 0:
            confidence -= 0.1
        if features.domain_authority == 0.5:  # Default value
            confidence -= 0.1

        # Confidence interval width
        interval_width = (1.0 - confidence) * 0.3

        lower_bound = max(0.0, quality_score - interval_width)
        upper_bound = min(1.0, quality_score + interval_width)

        return (lower_bound, upper_bound)

    def learn_from_feedback(
        self,
        features: QualityFeatures,
        predicted_quality: QualityMetrics,
        actual_quality: float,
        feedback_type: str = "implicit",
    ):
        """Learn from user feedback or research outcomes"""
        # Calculate prediction error
        error = abs(predicted_quality.overall_quality - actual_quality)

        # Update feature weights based on error
        if error > 0.2:  # Significant error
            learning_rate = 0.01

            # Adjust weights for features that may have caused the error
            if actual_quality > predicted_quality.overall_quality:
                # We underestimated - increase weights for positive features
                if features.peer_reviewed:
                    self.feature_weights["peer_reviewed"] += learning_rate
                if features.citation_count > 5:
                    self.feature_weights["citation_count"] += learning_rate
            else:
                # We overestimated - decrease weights
                if features.opinion_indicators > 3:
                    self.feature_weights["opinion_indicators"] -= learning_rate

        # Update domain reputation
        domain = self._extract_domain(predicted_quality.source_url)
        if domain:
            current_rep = self.domain_reputation.get(domain, 0.5)
            self.domain_reputation[domain] = current_rep * 0.9 + actual_quality * 0.1

        # Update baselines
        content_type = predicted_quality.content_type
        if content_type not in self.quality_baselines:
            self.quality_baselines[content_type] = {}

        baseline = self.quality_baselines[content_type]
        baseline["overall_quality"] = (
            baseline.get("overall_quality", 0.5) * 0.95 + actual_quality * 0.05
        )

        # Periodic save
        if len(self.training_data) % 20 == 0:
            self._save_model_data()

    # Helper methods for feature extraction
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse

            return urlparse(url).netloc.lower()
        except:
            return ""

    def _count_citations(self, content: str) -> int:
        """Count citation indicators in content"""
        citation_patterns = [
            r"\[\d+\]",  # [1], [2], etc.
            r"\(\d{4}\)",  # (2024)
            r"et al\.",  # et al.
            r"doi:",  # DOI references
            r"http[s]?://.*\.edu",  # Academic URLs
        ]

        count = 0
        for pattern in citation_patterns:
            count += len(re.findall(pattern, content, re.IGNORECASE))

        return count

    def _count_factual_claims(self, content: str) -> int:
        """Count factual claim indicators"""
        factual_indicators = [
            r"\d+%",  # Percentages
            r"\$\d+",  # Dollar amounts
            r"\d+\s*(million|billion|thousand)",  # Large numbers
            r"according to",  # Attribution
            r"study shows",  # Research references
            r"data indicates",  # Data references
        ]

        count = 0
        for pattern in factual_indicators:
            count += len(re.findall(pattern, content, re.IGNORECASE))

        return count

    def _count_opinion_indicators(self, content: str) -> int:
        """Count opinion/bias indicators"""
        opinion_indicators = [
            r"\bI think\b",
            r"\bI believe\b",
            r"\bIn my opinion\b",
            r"\bobviously\b",
            r"\bclearly\b",
            r"\bundoubtedly\b",
            r"\!{2,}",  # Multiple exclamation marks
        ]

        count = 0
        for pattern in opinion_indicators:
            count += len(re.findall(pattern, content, re.IGNORECASE))

        return count

    def _estimate_grammar_errors(self, content: str) -> int:
        """Estimate grammatical errors (simple heuristic)"""
        errors = 0

        # Simple checks
        sentences = content.split(".")
        for sentence in sentences:
            if sentence.strip():
                # Check for basic grammar issues
                if not sentence.strip()[0].isupper():  # Sentence doesn't start with capital
                    errors += 1
                if "  " in sentence:  # Double spaces
                    errors += 1

        return errors

    def _count_external_links(self, content: str) -> int:
        """Count external links in content"""
        return len(re.findall(r"http[s]?://[^\s]+", content))

    def _count_multimedia(self, content: str) -> int:
        """Count multimedia elements"""
        multimedia_patterns = [
            r"<img",
            r"<video",
            r"<audio",
            r"\[image\]",
            r"\[video\]",
            r"!\[.*\]\(",  # Markdown images
        ]

        count = 0
        for pattern in multimedia_patterns:
            count += len(re.findall(pattern, content, re.IGNORECASE))

        return count

    def _assess_author_credentials(self, content: str, metadata: Dict) -> int:
        """Assess author credentials (0-5 scale)"""
        score = 0

        # Check metadata first
        if "author_title" in metadata:
            title = metadata["author_title"].lower()
            if any(title_indicator in title for title_indicator in ["dr.", "prof.", "phd"]):
                score += 3
            elif any(title_indicator in title for title_indicator in ["mr.", "ms.", "mrs."]):
                score += 1

        # Check content for credentials
        credential_patterns = [
            r"\bPh\.?D\b",
            r"\bDr\.\s",
            r"\bProfessor\b",
            r"\bResearcher\b",
            r"\bScientist\b",
        ]

        for pattern in credential_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                score += 1

        return min(score, 5)

    def _is_peer_reviewed(self, content: str, metadata: Dict) -> bool:
        """Check if content is peer-reviewed"""
        if metadata.get("peer_reviewed"):
            return True

        peer_review_indicators = [
            "peer-reviewed",
            "peer reviewed",
            "refereed journal",
            "manuscript received",
            "accepted for publication",
        ]

        content_lower = content.lower()
        return any(indicator in content_lower for indicator in peer_review_indicators)

    def _extract_publication_date(self, content: str, metadata: Dict) -> Optional[datetime]:
        """Extract publication date from content or metadata"""
        if "publication_date" in metadata:
            try:
                return datetime.fromisoformat(metadata["publication_date"])
            except:
                pass

        # Try to extract from content
        date_patterns = [
            r"(\d{4}-\d{2}-\d{2})",  # YYYY-MM-DD
            r"(\d{1,2}/\d{1,2}/\d{4})",  # MM/DD/YYYY
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}",
        ]

        for pattern in date_patterns:
            match = re.search(pattern, content)
            if match:
                try:
                    # Simple date parsing (could be improved)
                    return datetime.now()  # Placeholder
                except:
                    continue

        return None

    def _estimate_update_frequency(self, metadata: Dict) -> float:
        """Estimate how frequently content is updated"""
        if "last_modified" in metadata and "publication_date" in metadata:
            try:
                pub_date = datetime.fromisoformat(metadata["publication_date"])
                mod_date = datetime.fromisoformat(metadata["last_modified"])
                days_diff = (mod_date - pub_date).days
                if days_diff > 0:
                    return 1.0 / days_diff  # Updates per day
            except:
                pass

        return 0.0  # No update information

    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction statistics and model performance"""
        if not self.training_data:
            return {"status": "no_training_data"}

        # Calculate model performance metrics
        recent_predictions = len([t for t in self.training_data if t.human_rating is not None])

        return {
            "total_predictions": len(self.training_data),
            "rated_predictions": recent_predictions,
            "domain_reputations": len(self.domain_reputation),
            "content_type_baselines": len(self.quality_baselines),
            "feature_weights": self.feature_weights,
            "model_confidence": 0.8,  # Placeholder
        }
