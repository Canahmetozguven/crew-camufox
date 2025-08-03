"""
Research Templates Module

This module provides pre-defined research templates, workflow patterns,
and customizable research strategies for various research scenarios.
"""

from .research_templates import (
    ResearchTemplate,
    ResearchTemplateManager,
    TemplateType,
    AcademicResearchTemplate,
    MarketResearchTemplate,
    CompetitiveAnalysisTemplate,
    NewsResearchTemplate,
    TechnicalResearchTemplate,
)

from .workflow_patterns import (
    WorkflowPattern,
    WorkflowManager,
    PatternType,
    LinearResearchPattern,
    DeepDivePattern,
    ComparativeAnalysisPattern,
    TrendAnalysisPattern,
)

__all__ = [
    "ResearchTemplate",
    "ResearchTemplateManager",
    "TemplateType",
    "AcademicResearchTemplate",
    "MarketResearchTemplate",
    "CompetitiveAnalysisTemplate",
    "NewsResearchTemplate",
    "TechnicalResearchTemplate",
    "WorkflowPattern",
    "WorkflowManager",
    "PatternType",
    "LinearResearchPattern",
    "DeepDivePattern",
    "ComparativeAnalysisPattern",
    "TrendAnalysisPattern",
]
