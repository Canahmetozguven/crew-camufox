"""
Machine Learning Integration Module

This module provides ML-enhanced capabilities for the crew-camufox research system:
- Query optimization and expansion
- Source quality prediction
- Research pattern recognition
- Personalized recommendations
- Automated workflow optimization
"""

from .query_optimizer import QueryOptimizer, QueryExpansion
from .source_predictor import SourceQualityPredictor, QualityMetrics
from .pattern_recognition import PatternRecognizer, ResearchPattern
from .recommendation_engine import RecommendationEngine, ResearchRecommendation
from .ml_coordinator import MLCoordinator

__all__ = [
    "QueryOptimizer",
    "QueryExpansion",
    "SourceQualityPredictor",
    "QualityMetrics",
    "PatternRecognizer",
    "ResearchPattern",
    "RecommendationEngine",
    "ResearchRecommendation",
    "MLCoordinator",
]
