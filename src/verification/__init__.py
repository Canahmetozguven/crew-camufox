"""
Source Verification System

Advanced source verification and credibility assessment system with multiple
verification algorithms, bias detection, and reliability scoring.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import re
import hashlib
import asyncio
from urllib.parse import urlparse
import statistics


class SourceType(Enum):
    """Types of sources that can be verified"""

    ACADEMIC = "academic"
    NEWS = "news"
    GOVERNMENT = "government"
    CORPORATE = "corporate"
    SOCIAL_MEDIA = "social_media"
    BLOG = "blog"
    WIKI = "wiki"
    FORUM = "forum"
    UNKNOWN = "unknown"


class CredibilityLevel(Enum):
    """Credibility levels for sources"""

    VERY_HIGH = "very_high"  # 90-100%
    HIGH = "high"  # 75-89%
    MEDIUM = "medium"  # 50-74%
    LOW = "low"  # 25-49%
    VERY_LOW = "very_low"  # 0-24%


class BiasType(Enum):
    """Types of bias that can be detected"""

    POLITICAL_LEFT = "political_left"
    POLITICAL_RIGHT = "political_right"
    COMMERCIAL = "commercial"
    CONFIRMATION = "confirmation"
    SELECTION = "selection"
    SURVIVORSHIP = "survivorship"
    RECENCY = "recency"
    AUTHORITY = "authority"
    NONE_DETECTED = "none_detected"


@dataclass
class SourceMetadata:
    """Metadata for a source"""

    url: str
    domain: str
    title: str
    author: Optional[str] = None
    publication_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    content_type: str = "text"
    language: str = "en"
    word_count: int = 0


@dataclass
class VerificationResult:
    """Result of source verification"""

    source_metadata: SourceMetadata
    credibility_score: float  # 0.0 to 1.0
    credibility_level: CredibilityLevel
    source_type: SourceType

    # Verification components
    authority_score: float = 0.0
    accuracy_score: float = 0.0
    objectivity_score: float = 0.0
    currency_score: float = 0.0
    coverage_score: float = 0.0

    # Bias analysis
    detected_biases: Optional[List[BiasType]] = None
    bias_score: float = 0.0  # 0.0 = no bias, 1.0 = extreme bias

    # Quality indicators
    has_citations: bool = False
    citation_count: int = 0
    external_validation: bool = False
    peer_reviewed: bool = False

    # Risk factors
    risk_factors: Optional[List[str]] = None
    red_flags: Optional[List[str]] = None

    # Cross-reference data
    similar_sources: Optional[List[str]] = None
    contradicting_sources: Optional[List[str]] = None

    verification_timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.detected_biases is None:
            self.detected_biases = []
        if self.risk_factors is None:
            self.risk_factors = []
        if self.red_flags is None:
            self.red_flags = []
        if self.similar_sources is None:
            self.similar_sources = []
        if self.contradicting_sources is None:
            self.contradicting_sources = []
        if self.verification_timestamp is None:
            self.verification_timestamp = datetime.now()


class SourceClassifier:
    """Classifies sources by type and authority level"""

    def __init__(self):
        self.academic_domains = {
            "edu",
            "ac.uk",
            "ac.jp",
            "edu.au",
            "scholar.google.com",
            "pubmed.ncbi.nlm.nih.gov",
            "arxiv.org",
            "jstor.org",
            "ieee.org",
            "acm.org",
            "springer.com",
            "elsevier.com",
        }

        self.government_domains = {
            "gov",
            "mil",
            "gov.uk",
            "gov.au",
            "gov.ca",
            "europa.eu",
            "un.org",
            "who.int",
            "worldbank.org",
            "imf.org",
        }

        self.news_domains = {
            "reuters.com",
            "ap.org",
            "bbc.com",
            "cnn.com",
            "nytimes.com",
            "washingtonpost.com",
            "theguardian.com",
            "wsj.com",
            "npr.org",
            "economist.com",
            "bloomberg.com",
            "ft.com",
        }

        self.high_authority_indicators = {
            "institutional_affiliation",
            "peer_review",
            "editorial_board",
            "established_publication",
            "expert_author",
            "citations",
        }

    def classify_source_type(self, url: str, content: str = "") -> SourceType:
        """Classify source type based on URL and content"""
        domain = urlparse(url).netloc.lower()

        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]

        # Check against known domain categories
        domain_parts = domain.split(".")

        # Academic sources
        if any(edu_domain in domain for edu_domain in self.academic_domains):
            return SourceType.ACADEMIC

        # Government sources
        if any(gov_domain in domain for gov_domain in self.government_domains):
            return SourceType.GOVERNMENT

        # News sources
        if any(news_domain in domain for news_domain in self.news_domains):
            return SourceType.NEWS

        # Social media
        if any(
            social in domain
            for social in ["facebook", "twitter", "instagram", "linkedin", "youtube", "tiktok"]
        ):
            return SourceType.SOCIAL_MEDIA

        # Wikis
        if "wiki" in domain:
            return SourceType.WIKI

        # Forums
        if any(forum in domain for forum in ["reddit", "stackoverflow", "quora", "forum"]):
            return SourceType.FORUM

        # Content-based classification
        if content:
            content_lower = content.lower()

            # Academic indicators
            if any(
                indicator in content_lower
                for indicator in ["abstract", "methodology", "references", "peer-reviewed", "doi:"]
            ):
                return SourceType.ACADEMIC

            # Blog indicators
            if any(
                indicator in content_lower
                for indicator in ["posted by", "blog", "personal opinion", "my thoughts"]
            ):
                return SourceType.BLOG

        # Default classification
        if any(tld in domain_parts for tld in ["com", "org", "net"]):
            return SourceType.CORPORATE

        return SourceType.UNKNOWN

    def assess_authority_level(
        self, source_type: SourceType, metadata: SourceMetadata, content: str = ""
    ) -> float:
        """Assess authority level of source (0.0 to 1.0)"""
        base_scores = {
            SourceType.ACADEMIC: 0.9,
            SourceType.GOVERNMENT: 0.85,
            SourceType.NEWS: 0.7,
            SourceType.CORPORATE: 0.5,
            SourceType.WIKI: 0.6,
            SourceType.BLOG: 0.3,
            SourceType.SOCIAL_MEDIA: 0.2,
            SourceType.FORUM: 0.25,
            SourceType.UNKNOWN: 0.1,
        }

        score = base_scores.get(source_type, 0.1)

        # Adjust based on additional factors
        if metadata.author:
            score += 0.1  # Has identified author

        if metadata.publication_date:
            # Recency bonus (within last 2 years)
            days_old = (datetime.now() - metadata.publication_date).days
            if days_old < 730:  # 2 years
                score += 0.05

        # Content analysis adjustments
        if content:
            content_lower = content.lower()

            # Positive indicators
            if "references" in content_lower or "bibliography" in content_lower:
                score += 0.1
            if "peer-reviewed" in content_lower or "peer reviewed" in content_lower:
                score += 0.15
            if content_lower.count("citation") > 2:
                score += 0.05

            # Negative indicators
            if any(
                negative in content_lower
                for negative in ["unverified", "rumor", "allegedly", "unconfirmed"]
            ):
                score -= 0.1

        return min(max(score, 0.0), 1.0)


class BiasDetector:
    """Detects various types of bias in content"""

    def __init__(self):
        self.political_left_keywords = {
            "progressive",
            "liberal",
            "democrat",
            "socialism",
            "climate change",
            "social justice",
            "inequality",
            "diversity",
            "inclusion",
        }

        self.political_right_keywords = {
            "conservative",
            "republican",
            "traditional",
            "free market",
            "individual responsibility",
            "law and order",
            "patriot",
        }

        self.commercial_indicators = {
            "sponsored",
            "advertisement",
            "affiliate",
            "product placement",
            "buy now",
            "limited time",
            "special offer",
            "discount",
        }

        self.bias_phrases = {
            "obviously",
            "clearly",
            "everyone knows",
            "it is undeniable",
            "without a doubt",
            "any reasonable person",
            "common sense",
        }

    def detect_bias(self, content: str, metadata: SourceMetadata) -> Tuple[List[BiasType], float]:
        """Detect bias types and calculate bias score"""
        detected_biases = []
        bias_indicators = 0
        total_checks = 0

        content_lower = content.lower()

        # Political bias detection
        left_count = sum(1 for keyword in self.political_left_keywords if keyword in content_lower)
        right_count = sum(
            1 for keyword in self.political_right_keywords if keyword in content_lower
        )

        if left_count > right_count + 2:
            detected_biases.append(BiasType.POLITICAL_LEFT)
            bias_indicators += 1
        elif right_count > left_count + 2:
            detected_biases.append(BiasType.POLITICAL_RIGHT)
            bias_indicators += 1

        total_checks += 1

        # Commercial bias detection
        commercial_count = sum(
            1 for indicator in self.commercial_indicators if indicator in content_lower
        )
        if commercial_count > 3:
            detected_biases.append(BiasType.COMMERCIAL)
            bias_indicators += 1

        total_checks += 1

        # Confirmation bias indicators
        bias_phrase_count = sum(1 for phrase in self.bias_phrases if phrase in content_lower)
        if bias_phrase_count > 2:
            detected_biases.append(BiasType.CONFIRMATION)
            bias_indicators += 1

        total_checks += 1

        # Authority bias (excessive appeal to authority)
        authority_phrases = ["expert says", "scientists agree", "studies show"]
        authority_count = sum(1 for phrase in authority_phrases if phrase in content_lower)
        if authority_count > 5:
            detected_biases.append(BiasType.AUTHORITY)
            bias_indicators += 1

        total_checks += 1

        # Recency bias (overemphasis on recent events)
        recent_phrases = ["breaking", "just released", "latest news", "trending"]
        recent_count = sum(1 for phrase in recent_phrases if phrase in content_lower)
        if recent_count > 3:
            detected_biases.append(BiasType.RECENCY)
            bias_indicators += 1

        total_checks += 1

        # Calculate bias score
        bias_score = bias_indicators / total_checks if total_checks > 0 else 0.0

        if not detected_biases:
            detected_biases.append(BiasType.NONE_DETECTED)

        return detected_biases, bias_score


class CrossReferenceValidator:
    """Validates information through cross-referencing"""

    def __init__(self):
        self.fact_check_domains = {
            "snopes.com",
            "factcheck.org",
            "politifact.com",
            "reuters.com/fact-check",
            "ap.org/ap-fact-check",
        }

    async def cross_reference_claims(self, content: str, source_url: str) -> Dict[str, Any]:
        """Cross-reference claims with other sources"""
        # Extract key claims from content
        claims = self._extract_claims(content)

        validation_results = {
            "verified_claims": [],
            "contradicted_claims": [],
            "unverified_claims": [],
            "supporting_sources": [],
            "contradicting_sources": [],
        }

        # This would integrate with actual search engines and fact-checking APIs
        # For now, return a mock validation
        for claim in claims:
            # Mock validation logic
            if len(claim) > 50:  # Longer claims more likely to be factual
                validation_results["verified_claims"].append(claim)
            else:
                validation_results["unverified_claims"].append(claim)

        return validation_results

    def _extract_claims(self, content: str) -> List[str]:
        """Extract factual claims from content"""
        # Simple claim extraction using sentence patterns
        sentences = re.split(r"[.!?]+", content)

        claims = []
        claim_indicators = [
            "according to",
            "research shows",
            "studies indicate",
            "data reveals",
            "statistics show",
            "evidence suggests",
        ]

        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in claim_indicators):
                if len(sentence) > 20:  # Filter out very short sentences
                    claims.append(sentence)

        return claims[:10]  # Limit to top 10 claims


class SourceVerifier:
    """Main source verification system"""

    def __init__(self):
        self.classifier = SourceClassifier()
        self.bias_detector = BiasDetector()
        self.cross_validator = CrossReferenceValidator()
        self.verification_cache: Dict[str, VerificationResult] = {}

    async def verify_source(
        self, url: str, content: str = "", metadata: Optional[SourceMetadata] = None
    ) -> VerificationResult:
        """Comprehensive source verification"""

        # Check cache first
        cache_key = self._generate_cache_key(url, content)
        if cache_key in self.verification_cache:
            cached_result = self.verification_cache[cache_key]
            # Return cached result if less than 24 hours old
            if cached_result.verification_timestamp and (
                datetime.now() - cached_result.verification_timestamp
            ) < timedelta(hours=24):
                return cached_result

        # Create metadata if not provided
        if metadata is None:
            metadata = self._extract_metadata(url, content)

        # Classify source type
        source_type = self.classifier.classify_source_type(url, content)

        # Assess authority
        authority_score = self.classifier.assess_authority_level(source_type, metadata, content)

        # Detect bias
        detected_biases, bias_score = self.bias_detector.detect_bias(content, metadata)

        # Analyze content quality
        accuracy_score = self._assess_accuracy(content, metadata)
        objectivity_score = self._assess_objectivity(content, bias_score)
        currency_score = self._assess_currency(metadata)
        coverage_score = self._assess_coverage(content)

        # Cross-reference validation
        cross_ref_result = await self.cross_validator.cross_reference_claims(content, url)

        # Calculate overall credibility score
        credibility_score = self._calculate_credibility_score(
            authority_score,
            accuracy_score,
            objectivity_score,
            currency_score,
            coverage_score,
            bias_score,
        )

        # Determine credibility level
        credibility_level = self._determine_credibility_level(credibility_score)

        # Identify risk factors and red flags
        risk_factors = self._identify_risk_factors(content, metadata, source_type)
        red_flags = self._identify_red_flags(content, metadata)

        # Create verification result
        result = VerificationResult(
            source_metadata=metadata,
            credibility_score=credibility_score,
            credibility_level=credibility_level,
            source_type=source_type,
            authority_score=authority_score,
            accuracy_score=accuracy_score,
            objectivity_score=objectivity_score,
            currency_score=currency_score,
            coverage_score=coverage_score,
            detected_biases=detected_biases,
            bias_score=bias_score,
            has_citations=self._has_citations(content),
            citation_count=self._count_citations(content),
            external_validation=len(cross_ref_result["verified_claims"]) > 0,
            risk_factors=risk_factors,
            red_flags=red_flags,
            similar_sources=cross_ref_result["supporting_sources"],
            contradicting_sources=cross_ref_result["contradicting_sources"],
        )

        # Cache result
        self.verification_cache[cache_key] = result

        return result

    def _extract_metadata(self, url: str, content: str) -> SourceMetadata:
        """Extract metadata from URL and content"""
        domain = urlparse(url).netloc

        # Extract title (simple heuristic)
        title_match = re.search(r"<title[^>]*>([^<]+)</title>", content, re.IGNORECASE)
        title = title_match.group(1) if title_match else "Unknown Title"

        # Extract author (simple heuristic)
        author_patterns = [
            r"by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
            r"author[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)",
            r"written\s+by\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
        ]
        author = None
        for pattern in author_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                author = match.group(1)
                break

        word_count = len(content.split()) if content else 0

        return SourceMetadata(
            url=url, domain=domain, title=title, author=author, word_count=word_count
        )

    def _assess_accuracy(self, content: str, metadata: SourceMetadata) -> float:
        """Assess content accuracy indicators"""
        score = 0.5  # Base score

        content_lower = content.lower()

        # Positive indicators
        if any(
            indicator in content_lower
            for indicator in ["source:", "according to", "study found", "research shows"]
        ):
            score += 0.2

        if any(
            indicator in content_lower for indicator in ["peer-reviewed", "published in", "journal"]
        ):
            score += 0.2

        if self._has_citations(content):
            score += 0.1

        # Negative indicators
        if any(
            indicator in content_lower
            for indicator in ["unverified", "rumor", "allegedly", "unconfirmed", "breaking"]
        ):
            score -= 0.2

        if metadata.word_count < 200:  # Very short content
            score -= 0.1

        return min(max(score, 0.0), 1.0)

    def _assess_objectivity(self, content: str, bias_score: float) -> float:
        """Assess content objectivity"""
        # Start with inverse of bias score
        score = 1.0 - bias_score

        content_lower = content.lower()

        # Subjective language indicators
        subjective_words = ["amazing", "terrible", "best", "worst", "incredible", "shocking"]
        subjective_count = sum(1 for word in subjective_words if word in content_lower)

        if subjective_count > 5:
            score -= 0.2

        # Emotional language
        emotional_words = ["outrageous", "devastating", "thrilling", "horrific"]
        emotional_count = sum(1 for word in emotional_words if word in content_lower)

        if emotional_count > 3:
            score -= 0.1

        return min(max(score, 0.0), 1.0)

    def _assess_currency(self, metadata: SourceMetadata) -> float:
        """Assess content currency (how up-to-date it is)"""
        if not metadata.publication_date:
            return 0.5  # Unknown date gets middle score

        days_old = (datetime.now() - metadata.publication_date).days

        # Currency scoring based on age
        if days_old < 30:  # Less than 1 month
            return 1.0
        elif days_old < 365:  # Less than 1 year
            return 0.8
        elif days_old < 1095:  # Less than 3 years
            return 0.6
        elif days_old < 1825:  # Less than 5 years
            return 0.4
        else:  # Older than 5 years
            return 0.2

    def _assess_coverage(self, content: str) -> float:
        """Assess comprehensiveness of coverage"""
        word_count = len(content.split())

        # Coverage based on content length
        if word_count > 2000:
            return 1.0
        elif word_count > 1000:
            return 0.8
        elif word_count > 500:
            return 0.6
        elif word_count > 200:
            return 0.4
        else:
            return 0.2

    def _calculate_credibility_score(
        self,
        authority: float,
        accuracy: float,
        objectivity: float,
        currency: float,
        coverage: float,
        bias: float,
    ) -> float:
        """Calculate overall credibility score"""
        # Weighted average of components
        weights = {
            "authority": 0.25,
            "accuracy": 0.25,
            "objectivity": 0.20,
            "currency": 0.15,
            "coverage": 0.10,
            "bias_penalty": 0.05,
        }

        score = (
            authority * weights["authority"]
            + accuracy * weights["accuracy"]
            + objectivity * weights["objectivity"]
            + currency * weights["currency"]
            + coverage * weights["coverage"]
            - bias * weights["bias_penalty"]
        )

        return min(max(score, 0.0), 1.0)

    def _determine_credibility_level(self, score: float) -> CredibilityLevel:
        """Determine credibility level from score"""
        if score >= 0.9:
            return CredibilityLevel.VERY_HIGH
        elif score >= 0.75:
            return CredibilityLevel.HIGH
        elif score >= 0.5:
            return CredibilityLevel.MEDIUM
        elif score >= 0.25:
            return CredibilityLevel.LOW
        else:
            return CredibilityLevel.VERY_LOW

    def _identify_risk_factors(
        self, content: str, metadata: SourceMetadata, source_type: SourceType
    ) -> List[str]:
        """Identify potential risk factors"""
        risk_factors = []
        content_lower = content.lower()

        # Source type risks
        if source_type in [SourceType.SOCIAL_MEDIA, SourceType.BLOG, SourceType.FORUM]:
            risk_factors.append("User-generated content")

        # Content risks
        if any(word in content_lower for word in ["unverified", "unconfirmed", "alleged"]):
            risk_factors.append("Contains unverified information")

        if not metadata.author:
            risk_factors.append("No identified author")

        if not metadata.publication_date:
            risk_factors.append("No publication date")

        if metadata.word_count < 200:
            risk_factors.append("Very brief content")

        # Bias risks
        if any(word in content_lower for word in ["obviously", "clearly", "everyone knows"]):
            risk_factors.append("Contains bias indicators")

        return risk_factors

    def _identify_red_flags(self, content: str, metadata: SourceMetadata) -> List[str]:
        """Identify serious red flags"""
        red_flags = []
        content_lower = content.lower()

        # Serious credibility issues
        if any(word in content_lower for word in ["fake news", "hoax", "conspiracy"]):
            red_flags.append("Contains conspiracy or fake news indicators")

        if any(word in content_lower for word in ["clickbait", "you won't believe"]):
            red_flags.append("Clickbait indicators")

        # Commercial red flags
        if content_lower.count("buy") > 5 or content_lower.count("sale") > 3:
            red_flags.append("Heavy commercial content")

        # Authority red flags
        if "self-proclaimed expert" in content_lower:
            red_flags.append("Self-proclaimed expertise")

        return red_flags

    def _has_citations(self, content: str) -> bool:
        """Check if content has citations"""
        citation_patterns = [
            r"\[\d+\]",  # [1], [2], etc.
            r"\(\d{4}\)",  # (2023), etc.
            r"doi:",  # DOI references
            r"http[s]?://[^\s]+",  # URLs
            r"et al\.",  # Academic citations
        ]

        for pattern in citation_patterns:
            if re.search(pattern, content):
                return True

        return False

    def _count_citations(self, content: str) -> int:
        """Count number of citations in content"""
        citation_patterns = [r"\[\d+\]", r"\(\d{4}\)", r"doi:"]

        count = 0
        for pattern in citation_patterns:
            count += len(re.findall(pattern, content))

        return count

    def _generate_cache_key(self, url: str, content: str) -> str:
        """Generate cache key for verification result"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:10]
        return f"{url}_{content_hash}"

    async def batch_verify_sources(
        self, sources: List[Tuple[str, str]]
    ) -> List[VerificationResult]:
        """Verify multiple sources in batch"""
        tasks = []
        for url, content in sources:
            task = self.verify_source(url, content)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and return valid results
        valid_results = []
        for result in results:
            if isinstance(result, VerificationResult):
                valid_results.append(result)

        return valid_results

    def get_verification_summary(self, results: List[VerificationResult]) -> Dict[str, Any]:
        """Generate summary statistics for verification results"""
        if not results:
            return {"error": "No verification results provided"}

        credibility_scores = [result.credibility_score for result in results]
        bias_scores = [result.bias_score for result in results]

        # Source type distribution
        source_types = {}
        for result in results:
            source_type = result.source_type.value
            source_types[source_type] = source_types.get(source_type, 0) + 1

        # Credibility level distribution
        credibility_levels = {}
        for result in results:
            level = result.credibility_level.value
            credibility_levels[level] = credibility_levels.get(level, 0) + 1

        # Bias analysis
        all_biases = []
        for result in results:
            if result.detected_biases:
                all_biases.extend([bias.value for bias in result.detected_biases])

        bias_distribution = {}
        for bias in all_biases:
            bias_distribution[bias] = bias_distribution.get(bias, 0) + 1

        return {
            "total_sources": len(results),
            "average_credibility": statistics.mean(credibility_scores),
            "median_credibility": statistics.median(credibility_scores),
            "average_bias_score": statistics.mean(bias_scores),
            "source_type_distribution": source_types,
            "credibility_level_distribution": credibility_levels,
            "bias_distribution": bias_distribution,
            "high_credibility_sources": len(
                [
                    r
                    for r in results
                    if r.credibility_level in [CredibilityLevel.HIGH, CredibilityLevel.VERY_HIGH]
                ]
            ),
            "low_credibility_sources": len(
                [
                    r
                    for r in results
                    if r.credibility_level in [CredibilityLevel.LOW, CredibilityLevel.VERY_LOW]
                ]
            ),
            "sources_with_red_flags": len([r for r in results if r.red_flags]),
        }
