#!/usr/bin/env python3
"""
Content-Specific Validation for CrewAI
Advanced content validation capabilities for research and agent outputs
"""

import re
import string
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter

try:
    from rich.console import Console
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    
    class MockConsole:
        def print(self, *args, **kwargs):
            print(*args)
    
    console = MockConsole()

class ContentType(Enum):
    """Content type classifications"""
    RESEARCH_REPORT = "research_report"
    SUMMARY = "summary"
    DATA_ANALYSIS = "data_analysis"
    CITATION = "citation"
    GENERAL_TEXT = "general_text"
    STRUCTURED_DATA = "structured_data"

@dataclass
class QualityMetrics:
    """Content quality metrics"""
    readability_score: float = 0.0
    coherence_score: float = 0.0
    completeness_score: float = 0.0
    accuracy_score: float = 0.0
    relevance_score: float = 0.0
    overall_quality: float = 0.0
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SourceCredibility:
    """Source credibility assessment"""
    domain_reputation: float = 0.0
    source_type: str = "unknown"
    publication_date: Optional[datetime] = None
    author_credibility: float = 0.0
    peer_reviewed: bool = False
    overall_credibility: float = 0.0

class ContentValidator:
    """
    Advanced content validation system for research outputs
    """
    
    def __init__(
        self,
        enable_deep_analysis: bool = True,
        quality_threshold: float = 0.7,
        credibility_threshold: float = 0.6
    ):
        self.enable_deep_analysis = enable_deep_analysis
        self.quality_threshold = quality_threshold
        self.credibility_threshold = credibility_threshold
        
        # Content analysis patterns
        self.citation_patterns = [
            r'\[[0-9]+\]',  # [1], [2], etc.
            r'\([^)]*[0-9]{4}[^)]*\)',  # (Author, 2023)
            r'https?://[^\s<>"{}|\\^`\[\]]+',  # URLs
        ]
        
        self.academic_indicators = [
            'abstract', 'methodology', 'conclusion', 'references',
            'hypothesis', 'experiment', 'analysis', 'findings',
            'research', 'study', 'investigation', 'survey'
        ]
        
        self.credible_domains = {
            'edu': 0.9,
            'gov': 0.9,
            'org': 0.7,
            'nature.com': 0.95,
            'science.org': 0.95,
            'ieee.org': 0.9,
            'arxiv.org': 0.8,
            'pubmed.ncbi.nlm.nih.gov': 0.95,
            'scholar.google.com': 0.8
        }
        
        console.print(f"[green]📝 Content Validator initialized[/green]")
        console.print(f"[cyan]   • Deep analysis: {enable_deep_analysis}[/cyan]")
        console.print(f"[cyan]   • Quality threshold: {quality_threshold}[/cyan]")
    
    async def validate_content_quality(
        self,
        content: str,
        content_type: ContentType = ContentType.GENERAL_TEXT,
        context: Optional[Dict[str, Any]] = None
    ) -> QualityMetrics:
        """Validate content quality with comprehensive metrics"""
        
        console.print(f"[blue]📊 Analyzing content quality for {content_type.value}...[/blue]")
        
        context = context or {}
        
        # Initialize metrics
        metrics = QualityMetrics()
        
        if not content or not content.strip():
            console.print("[red]❌ Empty content provided[/red]")
            return metrics
        
        # Calculate individual quality metrics
        metrics.readability_score = self._calculate_readability(content)
        metrics.coherence_score = self._calculate_coherence(content)
        metrics.completeness_score = self._calculate_completeness(content, content_type)
        metrics.accuracy_score = self._calculate_accuracy(content, context)
        metrics.relevance_score = self._calculate_relevance(content, context)
        
        # Calculate overall quality score
        weights = {
            'readability': 0.2,
            'coherence': 0.25,
            'completeness': 0.25,
            'accuracy': 0.2,
            'relevance': 0.1
        }
        
        metrics.overall_quality = (
            metrics.readability_score * weights['readability'] +
            metrics.coherence_score * weights['coherence'] +
            metrics.completeness_score * weights['completeness'] +
            metrics.accuracy_score * weights['accuracy'] +
            metrics.relevance_score * weights['relevance']
        )
        
        # Store detailed metrics
        metrics.detailed_metrics = self._get_detailed_metrics(content, content_type)
        
        console.print(f"[green]✅ Content quality analysis completed[/green]")
        console.print(f"[cyan]   • Overall quality: {metrics.overall_quality:.2f}[/cyan]")
        console.print(f"[cyan]   • Readability: {metrics.readability_score:.2f}[/cyan]")
        console.print(f"[cyan]   • Coherence: {metrics.coherence_score:.2f}[/cyan]")
        
        return metrics
    
    def _calculate_readability(self, content: str) -> float:
        """Calculate readability score using simplified metrics"""
        
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        words = content.split()
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Syllable estimation (simplified)
        syllable_count = sum(self._estimate_syllables(word) for word in words)
        avg_syllables_per_word = syllable_count / len(words) if words else 0
        
        # Simplified Flesch Reading Ease calculation
        if avg_sentence_length > 0 and avg_syllables_per_word > 0:
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            # Normalize to 0-1 scale
            readability = max(0, min(1, flesch_score / 100))
        else:
            readability = 0.5
        
        return readability
    
    def _estimate_syllables(self, word: str) -> int:
        """Estimate syllable count in a word"""
        
        word = word.lower().strip('.,!?;:"')
        if not word:
            return 0
        
        # Count vowel groups
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Adjust for silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _calculate_coherence(self, content: str) -> float:
        """Calculate content coherence based on structure and flow"""
        
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        if len(paragraphs) < 2:
            return 0.7  # Single paragraph gets neutral score
        
        coherence_factors = []
        
        # Paragraph length consistency
        paragraph_lengths = [len(p.split()) for p in paragraphs]
        if paragraph_lengths:
            avg_length = sum(paragraph_lengths) / len(paragraph_lengths)
            length_variance = sum((length - avg_length) ** 2 for length in paragraph_lengths) / len(paragraph_lengths)
            length_consistency = 1.0 / (1.0 + length_variance / 100)  # Normalize variance
            coherence_factors.append(length_consistency)
        
        # Transition word usage
        transition_words = {
            'however', 'therefore', 'furthermore', 'moreover', 'additionally',
            'consequently', 'nevertheless', 'subsequently', 'meanwhile', 'finally',
            'first', 'second', 'third', 'next', 'then', 'also', 'similarly',
            'in contrast', 'on the other hand', 'as a result', 'in conclusion'
        }
        
        content_lower = content.lower()
        transition_count = sum(1 for word in transition_words if word in content_lower)
        transition_score = min(1.0, transition_count / len(paragraphs))
        coherence_factors.append(transition_score)
        
        # Topic consistency (simplified keyword analysis)
        all_words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
        word_freq = Counter(all_words)
        top_words = [word for word, count in word_freq.most_common(10)]
        
        topic_consistency = 0.0
        for paragraph in paragraphs:
            paragraph_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', paragraph.lower()))
            overlap = len(paragraph_words.intersection(set(top_words)))
            topic_consistency += overlap / len(top_words) if top_words else 0
        
        topic_consistency /= len(paragraphs)
        coherence_factors.append(topic_consistency)
        
        return sum(coherence_factors) / len(coherence_factors) if coherence_factors else 0.5
    
    def _calculate_completeness(self, content: str, content_type: ContentType) -> float:
        """Calculate content completeness based on type expectations"""
        
        completeness_factors = []
        
        # Basic completeness metrics
        word_count = len(content.split())
        
        # Type-specific completeness checks
        if content_type == ContentType.RESEARCH_REPORT:
            # Research reports should have certain sections
            required_sections = ['introduction', 'method', 'result', 'conclusion', 'reference']
            section_indicators = sum(1 for section in required_sections 
                                   if section in content.lower())
            section_completeness = section_indicators / len(required_sections)
            completeness_factors.append(section_completeness)
            
            # Expected length for research reports
            length_completeness = min(1.0, word_count / 1000)  # Expect at least 1000 words
            completeness_factors.append(length_completeness)
            
        elif content_type == ContentType.SUMMARY:
            # Summaries should be concise but comprehensive
            length_completeness = 1.0 if 50 <= word_count <= 500 else 0.5
            completeness_factors.append(length_completeness)
            
            # Should contain key information
            key_elements = ['main', 'key', 'important', 'significant', 'conclusion']
            key_completeness = sum(1 for element in key_elements 
                                 if element in content.lower()) / len(key_elements)
            completeness_factors.append(key_completeness)
            
        elif content_type == ContentType.DATA_ANALYSIS:
            # Data analysis should contain numbers, statistics
            number_pattern = r'\d+\.?\d*'
            numbers = re.findall(number_pattern, content)
            number_completeness = min(1.0, len(numbers) / 10)  # Expect some numbers
            completeness_factors.append(number_completeness)
            
            # Should contain analytical terms
            analytical_terms = ['analysis', 'trend', 'pattern', 'correlation', 'significant']
            term_completeness = sum(1 for term in analytical_terms 
                                  if term in content.lower()) / len(analytical_terms)
            completeness_factors.append(term_completeness)
        
        else:
            # General completeness based on structure
            sentences = re.split(r'[.!?]+', content)
            sentence_completeness = min(1.0, len(sentences) / 5)  # Expect at least 5 sentences
            completeness_factors.append(sentence_completeness)
        
        return sum(completeness_factors) / len(completeness_factors) if completeness_factors else 0.5
    
    def _calculate_accuracy(self, content: str, context: Dict[str, Any]) -> float:
        """Calculate content accuracy using available context"""
        
        accuracy_factors = []
        
        # Citation presence (indicates effort to verify)
        citations = 0
        for pattern in self.citation_patterns:
            citations += len(re.findall(pattern, content))
        
        citation_score = min(1.0, citations / 5)  # Expect some citations
        accuracy_factors.append(citation_score)
        
        # Academic language indicators
        academic_count = sum(1 for indicator in self.academic_indicators 
                           if indicator in content.lower())
        academic_score = min(1.0, academic_count / 3)
        accuracy_factors.append(academic_score)
        
        # Fact consistency (simplified check)
        # Look for contradictory statements
        contradiction_indicators = ['however', 'but', 'although', 'despite', 'nevertheless']
        contradiction_count = sum(1 for indicator in contradiction_indicators 
                                if indicator in content.lower())
        
        # Some contradictions are normal, too many might indicate confusion
        contradiction_score = 1.0 if contradiction_count <= 3 else max(0.3, 1.0 - (contradiction_count - 3) * 0.1)
        accuracy_factors.append(contradiction_score)
        
        # Context-based accuracy
        if 'expected_facts' in context:
            expected_facts = context['expected_facts']
            fact_matches = sum(1 for fact in expected_facts if fact.lower() in content.lower())
            fact_accuracy = fact_matches / len(expected_facts) if expected_facts else 1.0
            accuracy_factors.append(fact_accuracy)
        
        return sum(accuracy_factors) / len(accuracy_factors) if accuracy_factors else 0.7
    
    def _calculate_relevance(self, content: str, context: Dict[str, Any]) -> float:
        """Calculate content relevance to the given context"""
        
        if not context:
            return 0.8  # Neutral score if no context provided
        
        relevance_factors = []
        content_lower = content.lower()
        
        # Keyword relevance
        if 'keywords' in context:
            keywords = context['keywords']
            if isinstance(keywords, str):
                keywords = [keywords]
            
            keyword_matches = sum(1 for keyword in keywords 
                                if keyword.lower() in content_lower)
            keyword_relevance = keyword_matches / len(keywords) if keywords else 0.5
            relevance_factors.append(keyword_relevance)
        
        # Topic relevance
        if 'topic' in context:
            topic = context['topic'].lower()
            topic_words = topic.split()
            topic_matches = sum(1 for word in topic_words if word in content_lower)
            topic_relevance = topic_matches / len(topic_words) if topic_words else 0.5
            relevance_factors.append(topic_relevance)
        
        # Query relevance
        if 'query' in context:
            query = context['query'].lower()
            query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query))
            content_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', content_lower))
            
            common_words = query_words.intersection(content_words)
            query_relevance = len(common_words) / len(query_words) if query_words else 0.5
            relevance_factors.append(query_relevance)
        
        return sum(relevance_factors) / len(relevance_factors) if relevance_factors else 0.8
    
    def _get_detailed_metrics(self, content: str, content_type: ContentType) -> Dict[str, Any]:
        """Get detailed content metrics"""
        
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0,
            'avg_sentences_per_paragraph': len(sentences) / len(paragraphs) if paragraphs else 0,
            'character_count': len(content),
            'unique_words': len(set(word.lower().strip(string.punctuation) for word in words)),
            'lexical_diversity': len(set(word.lower().strip(string.punctuation) for word in words)) / len(words) if words else 0,
            'citation_count': sum(len(re.findall(pattern, content)) for pattern in self.citation_patterns),
            'content_type': content_type.value
        }
    
    async def validate_source_credibility(
        self, 
        sources: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> List[SourceCredibility]:
        """Validate credibility of sources"""
        
        console.print(f"[blue]🔍 Analyzing credibility of {len(sources)} sources...[/blue]")
        
        credibility_results = []
        
        for source in sources:
            credibility = SourceCredibility()
            
            # Domain-based credibility
            domain = self._extract_domain(source)
            credibility.domain_reputation = self._get_domain_credibility(domain)
            
            # Source type detection
            credibility.source_type = self._detect_source_type(source, domain)
            
            # Overall credibility calculation
            credibility.overall_credibility = (
                credibility.domain_reputation * 0.6 +
                (0.9 if credibility.source_type in ['academic', 'government'] else 0.5) * 0.4
            )
            
            credibility_results.append(credibility)
        
        avg_credibility = sum(c.overall_credibility for c in credibility_results) / len(credibility_results) if credibility_results else 0
        console.print(f"[green]✅ Source credibility analysis completed[/green]")
        console.print(f"[cyan]   • Average credibility: {avg_credibility:.2f}[/cyan]")
        
        return credibility_results
    
    def _extract_domain(self, source: str) -> str:
        """Extract domain from source URL or reference"""
        
        # Simple domain extraction
        if 'http' in source:
            import re
            match = re.search(r'https?://([^/]+)', source)
            if match:
                return match.group(1).lower()
        
        # Check for common domain patterns in citations
        domain_patterns = [
            r'([a-zA-Z0-9-]+\.(?:edu|gov|org|com|net))',
            r'(arxiv\.org)',
            r'(nature\.com)',
            r'(science\.org)'
        ]
        
        for pattern in domain_patterns:
            match = re.search(pattern, source.lower())
            if match:
                return match.group(1)
        
        return 'unknown'
    
    def _get_domain_credibility(self, domain: str) -> float:
        """Get credibility score for domain"""
        
        # Check exact matches
        if domain in self.credible_domains:
            return self.credible_domains[domain]
        
        # Check domain extensions
        if domain.endswith('.edu'):
            return 0.9
        elif domain.endswith('.gov'):
            return 0.9
        elif domain.endswith('.org'):
            return 0.7
        elif domain.endswith('.com'):
            return 0.5
        else:
            return 0.3
    
    def _detect_source_type(self, source: str, domain: str) -> str:
        """Detect the type of source"""
        
        source_lower = source.lower()
        
        if any(indicator in domain for indicator in ['.edu', 'arxiv', 'pubmed', 'ieee']):
            return 'academic'
        elif '.gov' in domain:
            return 'government'
        elif any(indicator in source_lower for indicator in ['journal', 'peer-reviewed', 'published']):
            return 'academic'
        elif any(indicator in source_lower for indicator in ['news', 'report', 'article']):
            return 'news'
        elif any(indicator in source_lower for indicator in ['blog', 'opinion', 'personal']):
            return 'blog'
        else:
            return 'unknown'
    
    def get_content_recommendations(self, metrics: QualityMetrics) -> List[str]:
        """Generate content improvement recommendations"""
        
        recommendations = []
        
        if metrics.readability_score < 0.6:
            recommendations.append("Improve readability by using shorter sentences and simpler vocabulary")
        
        if metrics.coherence_score < 0.6:
            recommendations.append("Enhance coherence by adding transition words and improving paragraph flow")
        
        if metrics.completeness_score < 0.7:
            recommendations.append("Add more comprehensive information and ensure all required sections are included")
        
        if metrics.accuracy_score < 0.7:
            recommendations.append("Include more citations and verify factual claims")
        
        if metrics.relevance_score < 0.6:
            recommendations.append("Focus more on the core topic and ensure content relevance")
        
        if metrics.detailed_metrics.get('citation_count', 0) < 3:
            recommendations.append("Add more credible sources and citations to support claims")
        
        word_count = metrics.detailed_metrics.get('word_count', 0)
        if word_count < 100:
            recommendations.append("Expand content with more detailed information and examples")
        
        return recommendations