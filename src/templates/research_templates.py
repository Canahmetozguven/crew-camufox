"""
Research Templates System

Provides pre-defined research templates and workflow patterns for different
research scenarios including academic research, market analysis, competitive
intelligence, news research, and technical documentation.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
from datetime import datetime
from pathlib import Path

if TYPE_CHECKING:
    from src.models.research_models import ResearchQuery


class TemplateType(Enum):
    """Types of research templates available"""

    ACADEMIC = "academic"
    MARKET_RESEARCH = "market_research"
    COMPETITIVE_ANALYSIS = "competitive_analysis"
    NEWS_RESEARCH = "news_research"
    TECHNICAL = "technical"
    CUSTOM = "custom"


@dataclass
class ResearchStep:
    """Individual step in a research workflow"""

    name: str
    description: str
    agent_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    expected_output: str = ""
    estimated_duration: int = 300  # seconds
    priority: int = 1  # 1 = highest, 5 = lowest


@dataclass
class TemplateMetadata:
    """Metadata for research templates"""

    name: str
    description: str
    template_type: TemplateType
    version: str = "1.0"
    author: str = "crew-camufox"
    created_date: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard
    estimated_completion: int = 1800  # seconds


class ResearchTemplate(ABC):
    """Abstract base class for research templates"""

    def __init__(self, metadata: TemplateMetadata):
        self.metadata = metadata
        self.steps: List[ResearchStep] = []
        self.configuration: Dict[str, Any] = {}

    @abstractmethod
    def define_workflow(self) -> List[ResearchStep]:
        """Define the research workflow steps"""
        raise NotImplementedError("Subclasses must implement define_workflow")

    @abstractmethod
    def configure_agents(self) -> Dict[str, Any]:
        """Configure agents for this template"""
        raise NotImplementedError("Subclasses must implement configure_agents")

    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate required inputs for this template"""
        required_fields = self.get_required_fields()
        return all(field in inputs for field in required_fields)

    def get_required_fields(self) -> List[str]:
        """Get required input fields for this template"""
        return ["query", "scope"]

    def customize_for_query(self, query: "ResearchQuery") -> None:
        """Customize template based on specific query requirements"""
        # Default implementation - can be overridden by subclasses
        _ = query  # Avoid unused parameter warning
        return

    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary representation"""
        return {
            "metadata": {
                "name": self.metadata.name,
                "description": self.metadata.description,
                "template_type": self.metadata.template_type.value,
                "version": self.metadata.version,
                "author": self.metadata.author,
                "created_date": self.metadata.created_date.isoformat(),
                "tags": self.metadata.tags,
                "difficulty": self.metadata.difficulty,
                "estimated_completion": self.metadata.estimated_completion,
            },
            "steps": [
                {
                    "name": step.name,
                    "description": step.description,
                    "agent_type": step.agent_type,
                    "parameters": step.parameters,
                    "dependencies": step.dependencies,
                    "expected_output": step.expected_output,
                    "estimated_duration": step.estimated_duration,
                    "priority": step.priority,
                }
                for step in self.steps
            ],
            "configuration": self.configuration,
        }


class AcademicResearchTemplate(ResearchTemplate):
    """Template for academic research with peer-reviewed sources"""

    def __init__(self):
        metadata = TemplateMetadata(
            name="Academic Research",
            description="Comprehensive academic research with peer-reviewed sources, citations, and scholarly analysis",
            template_type=TemplateType.ACADEMIC,
            tags=["academic", "scholarly", "peer-reviewed", "citations"],
            difficulty="hard",
            estimated_completion=2400,
        )
        super().__init__(metadata)
        self.steps = self.define_workflow()
        self.configuration = self.configure_agents()

    def define_workflow(self) -> List[ResearchStep]:
        """Define academic research workflow"""
        return [
            ResearchStep(
                name="literature_review",
                description="Conduct comprehensive literature review from academic sources",
                agent_type="academic_researcher",
                parameters={
                    "source_types": ["google_scholar", "pubmed", "arxiv", "jstor"],
                    "min_citations": 10,
                    "publication_years": 5,
                    "quality_threshold": 0.8,
                },
                expected_output="Annotated bibliography with 20-50 key sources",
                estimated_duration=600,
                priority=1,
            ),
            ResearchStep(
                name="gap_analysis",
                description="Identify research gaps and novel angles",
                agent_type="research_analyst",
                parameters={"analysis_depth": "deep", "gap_identification": True},
                dependencies=["literature_review"],
                expected_output="Research gap analysis with recommendations",
                estimated_duration=300,
                priority=2,
            ),
            ResearchStep(
                name="methodology_review",
                description="Analyze research methodologies used in literature",
                agent_type="methodology_expert",
                parameters={"methodology_focus": True, "comparative_analysis": True},
                dependencies=["literature_review"],
                expected_output="Methodology comparison and recommendations",
                estimated_duration=400,
                priority=2,
            ),
            ResearchStep(
                name="synthesis_analysis",
                description="Synthesize findings with critical analysis",
                agent_type="academic_writer",
                parameters={
                    "synthesis_style": "academic",
                    "citation_format": "apa",
                    "critical_analysis": True,
                },
                dependencies=["gap_analysis", "methodology_review"],
                expected_output="Academic synthesis with citations and analysis",
                estimated_duration=800,
                priority=1,
            ),
            ResearchStep(
                name="peer_review_simulation",
                description="Simulate peer review process for quality assurance",
                agent_type="peer_reviewer",
                parameters={
                    "review_criteria": ["novelty", "methodology", "significance", "clarity"],
                    "feedback_depth": "comprehensive",
                },
                dependencies=["synthesis_analysis"],
                expected_output="Peer review feedback and improvement suggestions",
                estimated_duration=300,
                priority=3,
            ),
        ]

    def configure_agents(self) -> Dict[str, Any]:
        """Configure agents for academic research"""
        return {
            "academic_researcher": {
                "role": "Academic Researcher",
                "goal": "Conduct thorough literature review with high-quality academic sources",
                "backstory": "Expert in academic research with access to scholarly databases and citation analysis",
                "tools": ["scholar_search", "citation_analyzer", "quality_assessor"],
                "verbose": True,
            },
            "research_analyst": {
                "role": "Research Gap Analyst",
                "goal": "Identify novel research opportunities and gaps in existing literature",
                "backstory": "Specialist in identifying research gaps and emerging trends in academic fields",
                "tools": ["gap_analyzer", "trend_detector"],
                "verbose": True,
            },
            "methodology_expert": {
                "role": "Research Methodology Expert",
                "goal": "Analyze and compare research methodologies for best practices",
                "backstory": "Expert in research design and methodology evaluation across disciplines",
                "tools": ["methodology_analyzer", "quality_checker"],
                "verbose": True,
            },
            "academic_writer": {
                "role": "Academic Writer",
                "goal": "Synthesize research findings into coherent academic analysis",
                "backstory": "Experienced academic writer specializing in synthesis and critical analysis",
                "tools": ["citation_manager", "academic_formatter", "synthesis_engine"],
                "verbose": True,
            },
            "peer_reviewer": {
                "role": "Peer Reviewer",
                "goal": "Provide constructive feedback using academic review standards",
                "backstory": "Experienced peer reviewer familiar with academic quality standards",
                "tools": ["review_analyzer", "feedback_generator"],
                "verbose": True,
            },
        }

    def get_required_fields(self) -> List[str]:
        """Required fields for academic research"""
        return ["query", "scope", "discipline", "publication_years", "source_quality"]


class MarketResearchTemplate(ResearchTemplate):
    """Template for market research and business intelligence"""

    def __init__(self):
        metadata = TemplateMetadata(
            name="Market Research",
            description="Comprehensive market analysis including trends, competitors, opportunities, and consumer insights",
            template_type=TemplateType.MARKET_RESEARCH,
            tags=["market", "business", "trends", "competitors", "consumer"],
            difficulty="medium",
            estimated_completion=1800,
        )
        super().__init__(metadata)
        self.steps = self.define_workflow()
        self.configuration = self.configure_agents()

    def define_workflow(self) -> List[ResearchStep]:
        """Define market research workflow"""
        return [
            ResearchStep(
                name="market_landscape",
                description="Analyze overall market landscape and size",
                agent_type="market_analyst",
                parameters={
                    "market_metrics": ["size", "growth", "segments", "trends"],
                    "time_horizon": "5_years",
                    "geographic_scope": "global",
                },
                expected_output="Market landscape overview with key metrics",
                estimated_duration=400,
                priority=1,
            ),
            ResearchStep(
                name="competitor_analysis",
                description="Comprehensive competitor landscape analysis",
                agent_type="competitive_analyst",
                parameters={
                    "competitor_depth": "deep",
                    "comparison_metrics": ["market_share", "pricing", "features", "positioning"],
                    "swot_analysis": True,
                },
                expected_output="Competitor analysis with SWOT and positioning",
                estimated_duration=500,
                priority=1,
            ),
            ResearchStep(
                name="consumer_insights",
                description="Analyze consumer behavior and preferences",
                agent_type="consumer_researcher",
                parameters={
                    "insight_categories": [
                        "demographics",
                        "behavior",
                        "preferences",
                        "pain_points",
                    ],
                    "sentiment_analysis": True,
                },
                expected_output="Consumer insights with behavioral patterns",
                estimated_duration=400,
                priority=2,
            ),
            ResearchStep(
                name="trend_identification",
                description="Identify emerging trends and opportunities",
                agent_type="trend_analyst",
                parameters={
                    "trend_categories": ["technology", "consumer", "regulatory", "economic"],
                    "prediction_horizon": "2_years",
                },
                dependencies=["market_landscape", "consumer_insights"],
                expected_output="Trend analysis with opportunity identification",
                estimated_duration=300,
                priority=2,
            ),
            ResearchStep(
                name="strategic_synthesis",
                description="Synthesize findings into actionable business insights",
                agent_type="business_strategist",
                parameters={
                    "strategy_focus": ["opportunities", "threats", "recommendations"],
                    "actionability": "high",
                },
                dependencies=["competitor_analysis", "trend_identification"],
                expected_output="Strategic recommendations with action plan",
                estimated_duration=200,
                priority=1,
            ),
        ]

    def configure_agents(self) -> Dict[str, Any]:
        """Configure agents for market research"""
        return {
            "market_analyst": {
                "role": "Market Analyst",
                "goal": "Analyze market size, trends, and growth opportunities",
                "backstory": "Expert in market analysis with access to industry reports and market data",
                "tools": ["market_data_tool", "trend_analyzer", "sizing_calculator"],
                "verbose": True,
            },
            "competitive_analyst": {
                "role": "Competitive Intelligence Analyst",
                "goal": "Conduct comprehensive competitor analysis and positioning",
                "backstory": "Specialist in competitive intelligence and market positioning analysis",
                "tools": ["competitor_tracker", "swot_analyzer", "positioning_mapper"],
                "verbose": True,
            },
            "consumer_researcher": {
                "role": "Consumer Insights Researcher",
                "goal": "Understand consumer behavior, preferences, and pain points",
                "backstory": "Expert in consumer psychology and market research methodologies",
                "tools": ["sentiment_analyzer", "survey_data_tool", "behavior_tracker"],
                "verbose": True,
            },
            "trend_analyst": {
                "role": "Trend Analyst",
                "goal": "Identify emerging trends and future opportunities",
                "backstory": "Futurist specializing in trend identification and opportunity mapping",
                "tools": ["trend_detector", "signal_scanner", "prediction_engine"],
                "verbose": True,
            },
            "business_strategist": {
                "role": "Business Strategist",
                "goal": "Synthesize research into actionable business strategy",
                "backstory": "Senior strategist with experience in translating research into business action",
                "tools": ["strategy_framework", "recommendation_engine"],
                "verbose": True,
            },
        }

    def get_required_fields(self) -> List[str]:
        """Required fields for market research"""
        return ["query", "scope", "industry", "geographic_region", "time_horizon"]


class CompetitiveAnalysisTemplate(ResearchTemplate):
    """Template for competitive intelligence and analysis"""

    def __init__(self):
        metadata = TemplateMetadata(
            name="Competitive Analysis",
            description="Deep competitive intelligence including competitor profiles, strategies, and market positioning",
            template_type=TemplateType.COMPETITIVE_ANALYSIS,
            tags=["competitive", "intelligence", "strategy", "positioning"],
            difficulty="medium",
            estimated_completion=1500,
        )
        super().__init__(metadata)
        self.steps = self.define_workflow()
        self.configuration = self.configure_agents()

    def define_workflow(self) -> List[ResearchStep]:
        """Define competitive analysis workflow"""
        return [
            ResearchStep(
                name="competitor_identification",
                description="Identify and categorize direct and indirect competitors",
                agent_type="competitor_scout",
                parameters={
                    "competitor_types": ["direct", "indirect", "potential"],
                    "discovery_depth": "comprehensive",
                    "market_mapping": True,
                },
                expected_output="Comprehensive competitor landscape map",
                estimated_duration=300,
                priority=1,
            ),
            ResearchStep(
                name="competitor_profiling",
                description="Create detailed profiles for key competitors",
                agent_type="profile_researcher",
                parameters={
                    "profile_depth": "deep",
                    "profile_categories": ["company", "products", "strategy", "financials", "team"],
                    "data_sources": ["public", "news", "social", "patents"],
                },
                dependencies=["competitor_identification"],
                expected_output="Detailed competitor profiles with key insights",
                estimated_duration=600,
                priority=1,
            ),
            ResearchStep(
                name="strategy_analysis",
                description="Analyze competitor strategies and positioning",
                agent_type="strategy_analyst",
                parameters={
                    "strategy_frameworks": ["porters_five", "swot", "positioning_map"],
                    "analysis_depth": "strategic",
                },
                dependencies=["competitor_profiling"],
                expected_output="Strategic analysis with positioning insights",
                estimated_duration=400,
                priority=2,
            ),
            ResearchStep(
                name="strengths_weaknesses",
                description="Identify competitor strengths and weaknesses",
                agent_type="swot_analyst",
                parameters={
                    "analysis_categories": ["strengths", "weaknesses", "opportunities", "threats"],
                    "comparative_analysis": True,
                },
                dependencies=["strategy_analysis"],
                expected_output="SWOT analysis with competitive gaps",
                estimated_duration=200,
                priority=2,
            ),
        ]

    def configure_agents(self) -> Dict[str, Any]:
        """Configure agents for competitive analysis"""
        return {
            "competitor_scout": {
                "role": "Competitor Scout",
                "goal": "Identify and map the competitive landscape comprehensively",
                "backstory": "Expert in competitive intelligence and market mapping",
                "tools": ["competitor_finder", "market_mapper", "classification_engine"],
                "verbose": True,
            },
            "profile_researcher": {
                "role": "Competitor Profile Researcher",
                "goal": "Create comprehensive competitor profiles and intelligence",
                "backstory": "Specialist in competitor research and intelligence gathering",
                "tools": ["company_profiler", "financial_analyzer", "news_tracker"],
                "verbose": True,
            },
            "strategy_analyst": {
                "role": "Strategy Analyst",
                "goal": "Analyze competitor strategies and market positioning",
                "backstory": "Expert in strategic analysis and competitive positioning",
                "tools": ["strategy_analyzer", "positioning_mapper", "framework_applier"],
                "verbose": True,
            },
            "swot_analyst": {
                "role": "SWOT Analyst",
                "goal": "Identify strengths, weaknesses, opportunities, and threats",
                "backstory": "Specialist in SWOT analysis and competitive gap identification",
                "tools": ["swot_framework", "gap_analyzer", "opportunity_finder"],
                "verbose": True,
            },
        }


class NewsResearchTemplate(ResearchTemplate):
    """Template for news research and current events analysis"""

    def __init__(self):
        metadata = TemplateMetadata(
            name="News Research",
            description="Current events research with timeline analysis, source verification, and trend identification",
            template_type=TemplateType.NEWS_RESEARCH,
            tags=["news", "current_events", "timeline", "verification"],
            difficulty="easy",
            estimated_completion=900,
        )
        super().__init__(metadata)
        self.steps = self.define_workflow()
        self.configuration = self.configure_agents()

    def define_workflow(self) -> List[ResearchStep]:
        """Define news research workflow"""
        return [
            ResearchStep(
                name="news_collection",
                description="Collect relevant news from multiple sources",
                agent_type="news_collector",
                parameters={
                    "source_diversity": True,
                    "time_range": "7_days",
                    "credibility_check": True,
                    "source_types": ["mainstream", "industry", "international"],
                },
                expected_output="Comprehensive news collection with source analysis",
                estimated_duration=300,
                priority=1,
            ),
            ResearchStep(
                name="fact_verification",
                description="Verify facts and cross-reference sources",
                agent_type="fact_checker",
                parameters={
                    "verification_depth": "comprehensive",
                    "cross_reference": True,
                    "source_credibility": True,
                },
                dependencies=["news_collection"],
                expected_output="Fact-checked news with credibility assessment",
                estimated_duration=400,
                priority=1,
            ),
            ResearchStep(
                name="timeline_construction",
                description="Construct chronological timeline of events",
                agent_type="timeline_builder",
                parameters={
                    "chronological_order": True,
                    "event_categorization": True,
                    "impact_assessment": True,
                },
                dependencies=["fact_verification"],
                expected_output="Detailed timeline with event analysis",
                estimated_duration=200,
                priority=2,
            ),
        ]

    def configure_agents(self) -> Dict[str, Any]:
        """Configure agents for news research"""
        return {
            "news_collector": {
                "role": "News Collector",
                "goal": "Gather comprehensive news coverage from diverse sources",
                "backstory": "Expert in news gathering with access to multiple news feeds and sources",
                "tools": ["news_aggregator", "source_diversity_tool", "credibility_checker"],
                "verbose": True,
            },
            "fact_checker": {
                "role": "Fact Checker",
                "goal": "Verify facts and assess source credibility",
                "backstory": "Professional fact-checker with expertise in source verification",
                "tools": [
                    "fact_verification_tool",
                    "cross_reference_checker",
                    "credibility_assessor",
                ],
                "verbose": True,
            },
            "timeline_builder": {
                "role": "Timeline Builder",
                "goal": "Construct accurate chronological timelines of events",
                "backstory": "Specialist in event sequencing and timeline construction",
                "tools": ["timeline_tool", "event_categorizer", "impact_analyzer"],
                "verbose": True,
            },
        }


class TechnicalResearchTemplate(ResearchTemplate):
    """Template for technical research and documentation"""

    def __init__(self):
        metadata = TemplateMetadata(
            name="Technical Research",
            description="Technical documentation research including specifications, implementation details, and best practices",
            template_type=TemplateType.TECHNICAL,
            tags=["technical", "documentation", "specifications", "implementation"],
            difficulty="medium",
            estimated_completion=1200,
        )
        super().__init__(metadata)
        self.steps = self.define_workflow()
        self.configuration = self.configure_agents()

    def define_workflow(self) -> List[ResearchStep]:
        """Define technical research workflow"""
        return [
            ResearchStep(
                name="specification_research",
                description="Research technical specifications and requirements",
                agent_type="technical_researcher",
                parameters={
                    "spec_types": ["official", "community", "implementation"],
                    "version_tracking": True,
                    "compatibility_check": True,
                },
                expected_output="Comprehensive technical specifications",
                estimated_duration=400,
                priority=1,
            ),
            ResearchStep(
                name="implementation_analysis",
                description="Analyze implementation examples and patterns",
                agent_type="implementation_analyst",
                parameters={
                    "pattern_recognition": True,
                    "best_practices": True,
                    "code_examples": True,
                },
                dependencies=["specification_research"],
                expected_output="Implementation patterns and best practices",
                estimated_duration=500,
                priority=1,
            ),
            ResearchStep(
                name="documentation_synthesis",
                description="Synthesize technical documentation",
                agent_type="technical_writer",
                parameters={
                    "documentation_style": "technical",
                    "code_formatting": True,
                    "example_integration": True,
                },
                dependencies=["implementation_analysis"],
                expected_output="Comprehensive technical documentation",
                estimated_duration=300,
                priority=2,
            ),
        ]

    def configure_agents(self) -> Dict[str, Any]:
        """Configure agents for technical research"""
        return {
            "technical_researcher": {
                "role": "Technical Researcher",
                "goal": "Research comprehensive technical specifications and requirements",
                "backstory": "Expert in technical research with deep understanding of specifications",
                "tools": ["spec_finder", "version_tracker", "compatibility_checker"],
                "verbose": True,
            },
            "implementation_analyst": {
                "role": "Implementation Analyst",
                "goal": "Analyze implementation patterns and identify best practices",
                "backstory": "Specialist in code analysis and implementation pattern recognition",
                "tools": ["pattern_analyzer", "best_practice_finder", "code_analyzer"],
                "verbose": True,
            },
            "technical_writer": {
                "role": "Technical Writer",
                "goal": "Create clear and comprehensive technical documentation",
                "backstory": "Expert technical writer specializing in developer documentation",
                "tools": ["documentation_formatter", "code_formatter", "example_generator"],
                "verbose": True,
            },
        }


class ResearchTemplateManager:
    """Manager for research templates and workflow orchestration"""

    def __init__(self):
        self.templates: Dict[TemplateType, ResearchTemplate] = {
            TemplateType.ACADEMIC: AcademicResearchTemplate(),
            TemplateType.MARKET_RESEARCH: MarketResearchTemplate(),
            TemplateType.COMPETITIVE_ANALYSIS: CompetitiveAnalysisTemplate(),
            TemplateType.NEWS_RESEARCH: NewsResearchTemplate(),
            TemplateType.TECHNICAL: TechnicalResearchTemplate(),
        }
        self.custom_templates: Dict[str, ResearchTemplate] = {}

    def get_template(self, template_type: Union[TemplateType, str]) -> Optional[ResearchTemplate]:
        """Get a research template by type"""
        if isinstance(template_type, str):
            # Try to get custom template first
            if template_type in self.custom_templates:
                return self.custom_templates[template_type]
            # Try to convert string to TemplateType
            try:
                template_type = TemplateType(template_type)
            except ValueError:
                return None

        return self.templates.get(template_type)

    def list_templates(self) -> Dict[str, Dict[str, Any]]:
        """List all available templates with metadata"""
        template_list = {}

        # Add built-in templates
        for template_type, template in self.templates.items():
            template_list[template_type.value] = {
                "name": template.metadata.name,
                "description": template.metadata.description,
                "difficulty": template.metadata.difficulty,
                "estimated_completion": template.metadata.estimated_completion,
                "tags": template.metadata.tags,
                "type": "built-in",
            }

        # Add custom templates
        for name, template in self.custom_templates.items():
            template_list[name] = {
                "name": template.metadata.name,
                "description": template.metadata.description,
                "difficulty": template.metadata.difficulty,
                "estimated_completion": template.metadata.estimated_completion,
                "tags": template.metadata.tags,
                "type": "custom",
            }

        return template_list

    def recommend_template(self, query: "ResearchQuery") -> List[TemplateType]:
        """Recommend templates based on research query"""
        recommendations = []

        # Simple keyword-based recommendation logic
        query_text = f"{query.main_question} {query.context}".lower()

        if any(
            keyword in query_text
            for keyword in ["academic", "research", "study", "literature", "scholarly"]
        ):
            recommendations.append(TemplateType.ACADEMIC)

        if any(
            keyword in query_text
            for keyword in ["market", "business", "industry", "consumer", "trends"]
        ):
            recommendations.append(TemplateType.MARKET_RESEARCH)

        if any(
            keyword in query_text
            for keyword in ["competitor", "competition", "versus", "compare", "rival"]
        ):
            recommendations.append(TemplateType.COMPETITIVE_ANALYSIS)

        if any(
            keyword in query_text for keyword in ["news", "current", "recent", "breaking", "events"]
        ):
            recommendations.append(TemplateType.NEWS_RESEARCH)

        if any(
            keyword in query_text
            for keyword in ["technical", "implementation", "code", "api", "documentation"]
        ):
            recommendations.append(TemplateType.TECHNICAL)

        # Default recommendation if no specific keywords found
        if not recommendations:
            recommendations = [TemplateType.ACADEMIC, TemplateType.MARKET_RESEARCH]

        return recommendations

    def register_custom_template(self, name: str, template: ResearchTemplate) -> None:
        """Register a custom research template"""
        self.custom_templates[name] = template

    def save_template(self, template: ResearchTemplate, filepath: Path) -> None:
        """Save template to file"""
        template_data = template.to_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(template_data, f, indent=2)

    def load_template(self, filepath: Path) -> Optional[ResearchTemplate]:
        """Load template from file"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                _ = json.load(f)  # data would be used for template reconstruction

            # This would need proper template reconstruction logic
            # For now, return None as this requires more complex implementation
            return None
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None

    def validate_template(self, template: ResearchTemplate) -> List[str]:
        """Validate template configuration"""
        issues = []

        if not template.steps:
            issues.append("Template has no defined steps")

        if not template.configuration:
            issues.append("Template has no agent configuration")

        # Check for circular dependencies
        step_names = {step.name for step in template.steps}
        for step in template.steps:
            for dep in step.dependencies:
                if dep not in step_names:
                    issues.append(f"Step '{step.name}' depends on unknown step '{dep}'")

        return issues
