#!/usr/bin/env python3
"""
Research-Specific Validation for CrewAI
Advanced validation capabilities for research outputs and academic content
"""

import re
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

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

class ResearchQuality(Enum):
    """Research quality levels"""
    POOR = "poor"
    FAIR = "fair" 
    GOOD = "good"
    EXCELLENT = "excellent"

@dataclass
class CitationInfo:
    """Citation information structure"""
    citation_text: str
    url: Optional[str] = None
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    publication_date: Optional[str] = None
    source_type: str = "unknown"
    credibility_score: float = 0.0
    is_accessible: bool = False

class CitationValidator:
    """
    Validator for citations and references in research content
    """
    
    def __init__(self):
        self.citation_patterns = {
            'apa': r'\([^)]*\d{4}[^)]*\)',
            'mla': r'[A-Z][a-z]+ \d{4}',
            'chicago': r'\d+\.',
            'numbered': r'\[\d+\]',
            'url': r'https?://[^\s<>"{}|\\^`\[\]]+'
        }
        
        self.academic_domains = {
            'pubmed.ncbi.nlm.nih.gov': 0.95,
            'scholar.google.com': 0.85,
            'arxiv.org': 0.9,
            'ieee.org': 0.9,
            'acm.org': 0.9,
            'nature.com': 0.95,
            'science.org': 0.95,
            'springer.com': 0.85,
            'elsevier.com': 0.85,
            'wiley.com': 0.85
        }
    
    async def validate_citations(self, content: str) -> Dict[str, Any]:
        """Validate citations in research content"""
        
        console.print("[blue]📚 Validating citations and references...[/blue]")
        
        citations = self._extract_citations(content)
        citation_analysis = {
            'total_citations': len(citations),
            'citation_types': {},
            'credibility_scores': [],
            'accessible_count': 0,
            'academic_count': 0,
            'citations_details': []
        }
        
        for citation in citations:
            citation_info = await self._analyze_citation(citation)
            citation_analysis['citations_details'].append(citation_info.__dict__)
            citation_analysis['credibility_scores'].append(citation_info.credibility_score)
            
            if citation_info.is_accessible:
                citation_analysis['accessible_count'] += 1
            
            if citation_info.source_type == 'academic':
                citation_analysis['academic_count'] += 1
            
            # Count citation types
            citation_type = self._detect_citation_type(citation)
            citation_analysis['citation_types'][citation_type] = citation_analysis['citation_types'].get(citation_type, 0) + 1
        
        # Calculate overall citation quality
        if citation_analysis['credibility_scores']:
            citation_analysis['avg_credibility'] = sum(citation_analysis['credibility_scores']) / len(citation_analysis['credibility_scores'])
            citation_analysis['academic_ratio'] = citation_analysis['academic_count'] / citation_analysis['total_citations']
            citation_analysis['accessibility_ratio'] = citation_analysis['accessible_count'] / citation_analysis['total_citations']
        else:
            citation_analysis['avg_credibility'] = 0.0
            citation_analysis['academic_ratio'] = 0.0
            citation_analysis['accessibility_ratio'] = 0.0
        
        console.print(f"[green]✅ Citation validation completed[/green]")
        console.print(f"[cyan]   • Total citations: {citation_analysis['total_citations']}[/cyan]")
        console.print(f"[cyan]   • Average credibility: {citation_analysis['avg_credibility']:.2f}[/cyan]")
        console.print(f"[cyan]   • Academic ratio: {citation_analysis['academic_ratio']:.1%}[/cyan]")
        
        return citation_analysis
    
    def _extract_citations(self, content: str) -> List[str]:
        """Extract citations from content"""
        
        citations = set()
        
        for pattern_name, pattern in self.citation_patterns.items():
            matches = re.findall(pattern, content)
            citations.update(matches)
        
        return list(citations)
    
    async def _analyze_citation(self, citation: str) -> CitationInfo:
        """Analyze individual citation"""
        
        citation_info = CitationInfo(citation_text=citation)
        
        # Extract URL if present
        url_match = re.search(r'https?://[^\s<>"{}|\\^`\[\]]+', citation)
        if url_match:
            citation_info.url = url_match.group(0)
            citation_info.is_accessible = True  # Assume accessible for now
            
            # Analyze domain credibility
            domain = self._extract_domain(citation_info.url)
            citation_info.credibility_score = self._get_domain_credibility(domain)
            citation_info.source_type = self._detect_source_type(domain)
        
        # Extract potential authors
        author_patterns = [
            r'([A-Z][a-z]+ [A-Z][a-z]+)',  # First Last
            r'([A-Z]\. [A-Z][a-z]+)',      # F. Last
        ]
        
        for pattern in author_patterns:
            authors = re.findall(pattern, citation)
            citation_info.authors.extend(authors)
        
        # Extract publication date
        date_match = re.search(r'\b(19|20)\d{2}\b', citation)
        if date_match:
            citation_info.publication_date = date_match.group(0)
        
        return citation_info
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except:
            return 'unknown'
    
    def _get_domain_credibility(self, domain: str) -> float:
        """Get credibility score for domain"""
        
        # Check exact matches
        for academic_domain, score in self.academic_domains.items():
            if academic_domain in domain:
                return score
        
        # Check domain extensions
        if domain.endswith('.edu'):
            return 0.9
        elif domain.endswith('.gov'):
            return 0.9
        elif domain.endswith('.org'):
            return 0.7
        elif any(indicator in domain for indicator in ['wikipedia', 'wiki']):
            return 0.6
        elif domain.endswith('.com'):
            return 0.5
        else:
            return 0.3
    
    def _detect_source_type(self, domain: str) -> str:
        """Detect source type from domain"""
        
        if any(academic in domain for academic in self.academic_domains.keys()):
            return 'academic'
        elif domain.endswith('.edu'):
            return 'academic'
        elif domain.endswith('.gov'):
            return 'government'
        elif 'wiki' in domain:
            return 'wiki'
        elif any(news in domain for news in ['news', 'times', 'post', 'guardian', 'bbc']):
            return 'news'
        else:
            return 'unknown'
    
    def _detect_citation_type(self, citation: str) -> str:
        """Detect citation format type"""
        
        if re.match(r'\[\d+\]', citation):
            return 'numbered'
        elif re.match(r'\([^)]*\d{4}[^)]*\)', citation):
            return 'apa'
        elif 'http' in citation:
            return 'url'
        else:
            return 'unknown'

class FactChecker:
    """
    Basic fact checking capabilities for research content
    """
    
    def __init__(self):
        self.fact_patterns = {
            'dates': r'\b(19|20)\d{2}\b',
            'percentages': r'\d+\.?\d*%',
            'numbers': r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b',
            'currencies': r'\$\d+(?:,\d{3})*(?:\.\d{2})?',
            'measurements': r'\d+\.?\d*\s*(km|m|cm|mm|kg|g|lb|ft|in|miles?)\b'
        }
        
        self.fact_indicators = [
            'according to', 'studies show', 'research indicates',
            'data reveals', 'statistics show', 'evidence suggests',
            'reported that', 'found that', 'concluded that'
        ]
    
    async def check_facts(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Basic fact checking of content"""
        
        console.print("[blue]🔍 Performing fact checking analysis...[/blue]")
        
        fact_analysis = {
            'factual_claims': 0,
            'fact_types': {},
            'fact_indicators_count': 0,
            'confidence_score': 0.0,
            'suspicious_patterns': [],
            'recommendations': []
        }
        
        # Count factual claims by type
        for fact_type, pattern in self.fact_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            fact_analysis['fact_types'][fact_type] = len(matches)
            fact_analysis['factual_claims'] += len(matches)
        
        # Count fact indicators
        content_lower = content.lower()
        for indicator in self.fact_indicators:
            if indicator in content_lower:
                fact_analysis['fact_indicators_count'] += 1
        
        # Look for suspicious patterns
        suspicious_patterns = [
            (r'\b100%\b', 'Absolute percentage claims'),
            (r'\bproven\b|\bproof\b', 'Absolute proof claims'),
            (r'\balways\b|\bnever\b', 'Absolute statements'),
            (r'\beveryone\b|\bno one\b', 'Universal statements')
        ]
        
        for pattern, description in suspicious_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                fact_analysis['suspicious_patterns'].append(description)
        
        # Calculate confidence score
        if fact_analysis['factual_claims'] > 0:
            indicator_ratio = fact_analysis['fact_indicators_count'] / fact_analysis['factual_claims']
            base_confidence = min(1.0, indicator_ratio)
            
            # Reduce confidence for suspicious patterns
            suspicion_penalty = len(fact_analysis['suspicious_patterns']) * 0.1
            fact_analysis['confidence_score'] = max(0.0, base_confidence - suspicion_penalty)
        else:
            fact_analysis['confidence_score'] = 0.8  # Neutral if no specific claims
        
        # Generate recommendations
        if fact_analysis['confidence_score'] < 0.7:
            fact_analysis['recommendations'].append("Verify factual claims with additional sources")
        
        if len(fact_analysis['suspicious_patterns']) > 0:
            fact_analysis['recommendations'].append("Review absolute statements for accuracy")
        
        if fact_analysis['fact_indicators_count'] < fact_analysis['factual_claims'] * 0.3:
            fact_analysis['recommendations'].append("Add more source attributions for factual claims")
        
        console.print(f"[green]✅ Fact checking completed[/green]")
        console.print(f"[cyan]   • Factual claims: {fact_analysis['factual_claims']}[/cyan]")
        console.print(f"[cyan]   • Confidence score: {fact_analysis['confidence_score']:.2f}[/cyan]")
        
        return fact_analysis

class ResearchValidator:
    """
    Comprehensive research validation system
    """
    
    def __init__(
        self,
        enable_citation_validation: bool = True,
        enable_fact_checking: bool = True,
        research_quality_threshold: float = 0.7
    ):
        self.enable_citation_validation = enable_citation_validation
        self.enable_fact_checking = enable_fact_checking
        self.research_quality_threshold = research_quality_threshold
        
        self.citation_validator = CitationValidator()
        self.fact_checker = FactChecker()
        
        console.print("[green]📊 Research Validator initialized[/green]")
    
    async def validate_research_content(
        self,
        content: str,
        research_type: str = "general",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Comprehensive research content validation"""
        
        console.print(f"[blue]🎓 Validating research content ({research_type})...[/blue]")
        
        validation_result = {
            'research_type': research_type,
            'validation_timestamp': datetime.now().isoformat(),
            'overall_quality': ResearchQuality.FAIR,
            'quality_score': 0.0,
            'validation_components': {}
        }
        
        # Citation validation
        if self.enable_citation_validation:
            citation_analysis = await self.citation_validator.validate_citations(content)
            validation_result['validation_components']['citations'] = citation_analysis
        
        # Fact checking
        if self.enable_fact_checking:
            fact_analysis = await self.fact_checker.check_facts(content, context)
            validation_result['validation_components']['facts'] = fact_analysis
        
        # Research structure validation
        structure_analysis = self._validate_research_structure(content, research_type)
        validation_result['validation_components']['structure'] = structure_analysis
        
        # Calculate overall quality score
        quality_score = self._calculate_research_quality(validation_result['validation_components'])
        validation_result['quality_score'] = quality_score
        validation_result['overall_quality'] = self._get_quality_level(quality_score)
        
        # Generate recommendations
        validation_result['recommendations'] = self._generate_research_recommendations(validation_result)
        
        console.print(f"[green]✅ Research validation completed[/green]")
        console.print(f"[cyan]   • Overall quality: {validation_result['overall_quality'].value}[/cyan]")
        console.print(f"[cyan]   • Quality score: {quality_score:.2f}[/cyan]")
        
        return validation_result
    
    def _validate_research_structure(self, content: str, research_type: str) -> Dict[str, Any]:
        """Validate research structure and organization"""
        
        structure_analysis = {
            'has_introduction': False,
            'has_methodology': False,
            'has_results': False,
            'has_conclusion': False,
            'has_references': False,
            'section_count': 0,
            'structure_score': 0.0
        }
        
        content_lower = content.lower()
        
        # Check for common research sections
        section_indicators = {
            'has_introduction': ['introduction', 'background', 'overview'],
            'has_methodology': ['methodology', 'method', 'approach', 'procedure'],
            'has_results': ['results', 'findings', 'analysis', 'outcomes'],
            'has_conclusion': ['conclusion', 'summary', 'discussion', 'implications'],
            'has_references': ['references', 'bibliography', 'citations', 'sources']
        }
        
        for section, indicators in section_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                structure_analysis[section] = True
                structure_analysis['section_count'] += 1
        
        # Calculate structure score
        if research_type == 'academic':
            # Academic papers should have all sections
            required_sections = 5
            structure_analysis['structure_score'] = structure_analysis['section_count'] / required_sections
        else:
            # General research can be more flexible
            required_sections = 3
            structure_analysis['structure_score'] = min(1.0, structure_analysis['section_count'] / required_sections)
        
        return structure_analysis
    
    def _calculate_research_quality(self, components: Dict[str, Any]) -> float:
        """Calculate overall research quality score"""
        
        quality_factors = []
        weights = {}
        
        # Citation quality
        if 'citations' in components:
            citation_data = components['citations']
            if citation_data['total_citations'] > 0:
                citation_quality = (
                    citation_data['avg_credibility'] * 0.4 +
                    citation_data['academic_ratio'] * 0.3 +
                    citation_data['accessibility_ratio'] * 0.3
                )
            else:
                citation_quality = 0.3  # Penalty for no citations
            
            quality_factors.append(citation_quality)
            weights['citations'] = 0.3
        
        # Fact checking quality
        if 'facts' in components:
            fact_quality = components['facts']['confidence_score']
            quality_factors.append(fact_quality)
            weights['facts'] = 0.25
        
        # Structure quality
        if 'structure' in components:
            structure_quality = components['structure']['structure_score']
            quality_factors.append(structure_quality)
            weights['structure'] = 0.45
        
        # Calculate weighted average
        if quality_factors:
            total_weight = sum(weights.values())
            weighted_sum = sum(factor * weight for factor, weight in zip(quality_factors, weights.values()))
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return 0.5
    
    def _get_quality_level(self, score: float) -> ResearchQuality:
        """Convert quality score to quality level"""
        
        if score >= 0.9:
            return ResearchQuality.EXCELLENT
        elif score >= 0.7:
            return ResearchQuality.GOOD
        elif score >= 0.5:
            return ResearchQuality.FAIR
        else:
            return ResearchQuality.POOR
    
    def _generate_research_recommendations(self, validation_result: Dict[str, Any]) -> List[str]:
        """Generate research improvement recommendations"""
        
        recommendations = []
        components = validation_result['validation_components']
        
        # Citation recommendations
        if 'citations' in components:
            citation_data = components['citations']
            if citation_data['total_citations'] < 5:
                recommendations.append("Add more citations to support research claims")
            if citation_data['academic_ratio'] < 0.5:
                recommendations.append("Include more academic and peer-reviewed sources")
            if citation_data['avg_credibility'] < 0.7:
                recommendations.append("Use more credible and authoritative sources")
        
        # Fact checking recommendations
        if 'facts' in components:
            fact_data = components['facts']
            recommendations.extend(fact_data.get('recommendations', []))
        
        # Structure recommendations
        if 'structure' in components:
            structure_data = components['structure']
            if not structure_data['has_introduction']:
                recommendations.append("Add a clear introduction section")
            if not structure_data['has_methodology']:
                recommendations.append("Include methodology or approach section")
            if not structure_data['has_conclusion']:
                recommendations.append("Add a conclusion or summary section")
            if not structure_data['has_references']:
                recommendations.append("Include a references or bibliography section")
        
        # Overall quality recommendations
        if validation_result['quality_score'] < 0.7:
            recommendations.append("Improve overall research quality and rigor")
        
        return recommendations