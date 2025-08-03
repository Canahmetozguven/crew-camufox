#!/usr/bin/env python3
"""
Research Planner Agent
Specialized in creating comprehensive research strategies and plans
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from crewai import Agent, Task
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from src.models.research_models import ResearchQuery


class ResearchPlannerAgent:
    """Agent responsible for creating detailed research strategies and execution plans"""

    def __init__(
        self, model_name: str = "magistral:latest", ollama_base_url: str = "http://localhost:11434"
    ):
        self.model_name = model_name
        self.ollama_base_url = ollama_base_url
        self.llm = ChatOllama(
            model=model_name, base_url=ollama_base_url, temperature=0.2, num_ctx=40000
        )

        # Initialize the planner agent
        self.agent = Agent(
            role="Research Strategy Planner",
            goal="Create comprehensive, efficient, and targeted research plans for complex queries",
            backstory="""You are an expert research strategist with a PhD in Information Science 
            and 15+ years of experience in academic and corporate research. You specialize in 
            breaking down complex research questions into actionable investigation plans.
            
            Your expertise includes:
            - Query decomposition and analysis
            - Search strategy optimization
            - Source prioritization and evaluation criteria
            - Research methodology design
            - Quality assurance frameworks
            
            You think systematically and always consider multiple angles, potential biases,
            and the most efficient paths to comprehensive understanding.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )

    def create_comprehensive_plan(
        self,
        query: str,
        research_depth: str = "deep",
        max_sources: int = 25,
        focus_areas: Optional[List[str]] = None,
        time_range: Optional[str] = None,
        source_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a comprehensive research plan with detailed strategy"""

        # Define depth configurations
        depth_configs = {
            "surface": {
                "search_rounds": 1,
                "sources_per_round": min(max_sources, 8),
                "deep_dive_sources": 2,
                "cross_reference_depth": 1,
                "fact_check_rounds": 0,
                "time_limit_minutes": 15,
            },
            "medium": {
                "search_rounds": 2,
                "sources_per_round": min(max_sources // 2, 12),
                "deep_dive_sources": 5,
                "cross_reference_depth": 2,
                "fact_check_rounds": 1,
                "time_limit_minutes": 30,
            },
            "deep": {
                "search_rounds": 3,
                "sources_per_round": min(max_sources // 3, 15),
                "deep_dive_sources": 8,
                "cross_reference_depth": 3,
                "fact_check_rounds": 2,
                "time_limit_minutes": 60,
            },
            "exhaustive": {
                "search_rounds": 4,
                "sources_per_round": min(max_sources // 4, 20),
                "deep_dive_sources": 12,
                "cross_reference_depth": 4,
                "fact_check_rounds": 3,
                "time_limit_minutes": 120,
            },
        }

        config = depth_configs.get(research_depth, depth_configs["deep"])

        # Generate detailed research plan using LLM
        plan_prompt = self._create_planning_prompt(
            query, research_depth, config, focus_areas, time_range, source_types
        )

        try:
            response = self.llm.invoke(plan_prompt)
            llm_plan_text = response.content if hasattr(response, "content") else str(response)

            # Ensure we have a string for processing
            if not isinstance(llm_plan_text, str):
                llm_plan_text = str(llm_plan_text)

            # Parse and structure the LLM response
            structured_plan = self._parse_llm_plan(llm_plan_text)

            # Create comprehensive plan
            research_plan = {
                "id": f"plan_{int(datetime.now().timestamp())}",
                "query": query,
                "research_depth": research_depth,
                "config": config,
                "max_sources": max_sources,
                "focus_areas": focus_areas or [],
                "time_range": time_range,
                "source_types": source_types or self._get_default_source_types(),
                "created_at": datetime.now().isoformat(),
                # Strategic components
                "research_objectives": structured_plan.get("objectives", []),
                "key_questions": structured_plan.get("questions", []),
                "search_strategies": self._create_search_strategies(query, focus_areas),
                "source_priorities": self._define_source_priorities(),
                "quality_criteria": self._define_quality_criteria(),
                "execution_phases": self._create_execution_phases(config),
                # Advanced features
                "cross_reference_plan": self._create_cross_reference_plan(
                    config["cross_reference_depth"]
                ),
                "fact_check_strategy": self._create_fact_check_strategy(
                    config["fact_check_rounds"]
                ),
                "bias_mitigation": self._create_bias_mitigation_strategy(),
                "success_metrics": self._define_success_metrics(),
                # LLM-generated insights
                "llm_analysis": llm_plan_text,
                "strategic_notes": structured_plan.get("notes", []),
            }

            return research_plan

        except Exception as e:
            print(f"Error creating LLM-based plan: {e}")
            return self._create_fallback_plan(query, research_depth, config, max_sources)

    def _create_planning_prompt(
        self,
        query: str,
        depth: str,
        config: Dict[str, Any],
        focus_areas: Optional[List[str]],
        time_range: Optional[str],
        source_types: Optional[List[str]],
    ) -> str:
        """Create a detailed prompt for the LLM to generate research plans"""

        prompt = f"""As an expert research strategist, create a comprehensive research plan for the following query:

RESEARCH QUERY: "{query}"

RESEARCH PARAMETERS:
- Depth Level: {depth}
- Maximum Sources: {config['sources_per_round'] * config['search_rounds']}
- Search Rounds: {config['search_rounds']}
- Deep Analysis Sources: {config['deep_dive_sources']}
- Time Limit: {config['time_limit_minutes']} minutes
- Focus Areas: {focus_areas or 'To be determined'}
- Time Range: {time_range or 'Current/Recent'}
- Source Types: {source_types or 'Academic, News, Documentation, Analysis'}

Please provide a detailed research plan including:

1. RESEARCH OBJECTIVES (3-5 main goals):
   - What specific knowledge should be gained?
   - What questions need definitive answers?
   - What insights should be uncovered?

2. KEY RESEARCH QUESTIONS (5-10 questions):
   - Primary questions that must be answered
   - Secondary questions for deeper understanding
   - Validation questions for fact-checking

3. SEARCH STRATEGY:
   - Most effective search terms and phrases
   - Boolean search combinations
   - Alternative terminology to explore

4. SOURCE PRIORITIZATION:
   - Which types of sources to prioritize first
   - Quality indicators to look for
   - Red flags to avoid

5. POTENTIAL CHALLENGES:
   - What obstacles might be encountered?
   - How to overcome information gaps
   - Bias detection and mitigation strategies

6. SUCCESS CRITERIA:
   - How to measure research completeness
   - Quality benchmarks for sources
   - Confidence thresholds for conclusions

Format your response with clear sections and actionable guidance."""

        return prompt

    def _parse_llm_plan(self, llm_plan: str) -> Dict[str, List[str]]:
        """Parse structured information from LLM-generated plan"""
        parsed = {"objectives": [], "questions": [], "notes": []}

        lines = llm_plan.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Identify sections
            if "OBJECTIVE" in line.upper():
                current_section = "objectives"
                continue
            elif "QUESTION" in line.upper():
                current_section = "questions"
                continue
            elif any(keyword in line.upper() for keyword in ["STRATEGY", "CHALLENGE", "CRITERIA"]):
                current_section = "notes"

            # Extract content
            if line.startswith(("-", "*", "•")) or line[0:2].isdigit():
                content = line[1:].strip() if line.startswith(("-", "*", "•")) else line
                if current_section == "objectives" and len(parsed["objectives"]) < 5:
                    parsed["objectives"].append(content)
                elif (
                    current_section == "questions"
                    and "?" in content
                    and len(parsed["questions"]) < 10
                ):
                    parsed["questions"].append(content)
                elif current_section == "notes":
                    parsed["notes"].append(content)

        return parsed

    def _create_search_strategies(
        self, query: str, focus_areas: Optional[List[str]]
    ) -> Dict[str, List[str]]:
        """Create comprehensive search strategies"""
        strategies = {
            "primary_terms": [query],
            "secondary_terms": [],
            "phrase_searches": [],
            "boolean_combinations": [],
            "domain_specific": [],
        }

        # Generate variations
        query_words = query.lower().split()

        # Add focus areas
        if focus_areas:
            strategies["secondary_terms"].extend(focus_areas)

        # Create phrase searches
        if len(query_words) > 1:
            strategies["phrase_searches"].append(f'"{query}"')

        # Generate related terms based on query analysis
        if any(tech in query.lower() for tech in ["ai", "artificial intelligence", "technology"]):
            strategies["domain_specific"].extend(
                [
                    f"{query} research papers",
                    f"{query} academic studies",
                    f"{query} industry analysis",
                    f"{query} latest developments",
                    f"{query} future trends",
                ]
            )
        elif any(health in query.lower() for health in ["health", "medical", "disease"]):
            strategies["domain_specific"].extend(
                [
                    f"{query} clinical trials",
                    f"{query} medical research",
                    f"{query} treatment options",
                    f"{query} statistics",
                ]
            )
        else:
            # General enhancements
            strategies["secondary_terms"].extend(
                [
                    f"{query} analysis",
                    f"{query} overview",
                    f"{query} current status",
                    f"{query} 2024",
                    f"{query} recent developments",
                ]
            )

        # Boolean combinations
        if len(query_words) > 1:
            strategies["boolean_combinations"] = [
                f"{query_words[0]} AND {query_words[1]}",
                f'("{query}") OR ({query_words[0]} AND {query_words[1]})',
            ]

        return strategies

    def _define_source_priorities(self) -> List[Dict[str, Any]]:
        """Define source priority levels and criteria"""
        return [
            {
                "priority": 1,
                "types": ["academic", "research_institution", "government"],
                "indicators": [".edu", ".gov", ".org", "research", "university", "institute"],
                "weight": 1.0,
            },
            {
                "priority": 2,
                "types": ["news", "journalism", "analysis"],
                "indicators": ["reuters", "bbc", "ap news", "financial times", "economist"],
                "weight": 0.8,
            },
            {
                "priority": 3,
                "types": ["documentation", "official", "reference"],
                "indicators": ["documentation", "manual", "guide", "specification"],
                "weight": 0.7,
            },
            {
                "priority": 4,
                "types": ["blog", "article", "opinion"],
                "indicators": ["blog", "medium", "linkedin", "expert", "analysis"],
                "weight": 0.6,
            },
        ]

    def _define_quality_criteria(self) -> Dict[str, Any]:
        """Define comprehensive quality assessment criteria"""
        return {
            "content_quality": {
                "min_word_count": 200,
                "citation_presence": True,
                "structured_format": True,
                "recent_publication": "prefer",
            },
            "source_credibility": {
                "author_credentials": "required",
                "publication_reputation": "high_priority",
                "peer_review_status": "preferred",
                "fact_check_status": "required",
            },
            "relevance_factors": {
                "topic_alignment": 0.4,
                "depth_of_coverage": 0.3,
                "currency_of_information": 0.2,
                "unique_insights": 0.1,
            },
            "exclusion_criteria": [
                "paywall_protected",
                "login_required",
                "broken_links",
                "low_content_quality",
                "obvious_bias_without_disclosure",
            ],
        }

    def _create_execution_phases(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create detailed execution phases for research"""
        phases = []

        # Phase 1: Initial Discovery
        phases.append(
            {
                "phase": 1,
                "name": "Initial Discovery",
                "duration_minutes": config["time_limit_minutes"] * 0.3,
                "objectives": [
                    "Identify primary sources",
                    "Establish baseline understanding",
                    "Validate search strategies",
                ],
                "activities": [
                    "Execute primary search terms",
                    "Assess source landscape",
                    "Refine search approach",
                ],
                "success_criteria": [
                    "Find at least 5 relevant sources",
                    "Confirm query scope is appropriate",
                    "Identify key information gaps",
                ],
            }
        )

        # Phase 2: Deep Analysis
        if config["search_rounds"] > 1:
            phases.append(
                {
                    "phase": 2,
                    "name": "Deep Analysis",
                    "duration_minutes": config["time_limit_minutes"] * 0.4,
                    "objectives": [
                        "Gather comprehensive information",
                        "Cross-reference key facts",
                        "Identify expert perspectives",
                    ],
                    "activities": [
                        "Execute secondary searches",
                        "Analyze high-priority sources",
                        "Document key findings",
                    ],
                    "success_criteria": [
                        f"Analyze {config['deep_dive_sources']} sources thoroughly",
                        "Cross-reference major claims",
                        "Build comprehensive fact base",
                    ],
                }
            )

        # Phase 3: Validation & Synthesis
        phases.append(
            {
                "phase": len(phases) + 1,
                "name": "Validation & Synthesis",
                "duration_minutes": config["time_limit_minutes"] * 0.3,
                "objectives": [
                    "Validate key findings",
                    "Synthesize insights",
                    "Assess research completeness",
                ],
                "activities": [
                    "Fact-check critical claims",
                    "Identify contradictions",
                    "Prepare final synthesis",
                ],
                "success_criteria": [
                    "Verify accuracy of key facts",
                    "Resolve major contradictions",
                    "Achieve target confidence level",
                ],
            }
        )

        return phases

    def _create_cross_reference_plan(self, depth: int) -> Dict[str, Any]:
        """Create cross-referencing strategy"""
        return {
            "enabled": depth > 0,
            "depth_level": depth,
            "strategies": [
                "Compare claims across multiple sources",
                "Verify statistics and data points",
                "Check expert consensus on key topics",
                "Validate timeline accuracy",
                "Cross-check methodology descriptions",
            ][: depth + 1],
            "confidence_thresholds": {1: 0.6, 2: 0.7, 3: 0.8, 4: 0.9}.get(depth, 0.7),
        }

    def _create_fact_check_strategy(self, rounds: int) -> Dict[str, Any]:
        """Create fact-checking strategy"""
        return {
            "enabled": rounds > 0,
            "rounds": rounds,
            "focus_areas": (
                [
                    "Statistical claims",
                    "Historical facts",
                    "Expert quotes",
                    "Technical specifications",
                    "Current status information",
                ][: rounds + 2]
                if rounds > 0
                else []
            ),
            "verification_methods": [
                "Multiple source confirmation",
                "Primary source verification",
                "Expert opinion validation",
                "Official documentation check",
            ],
        }

    def _create_bias_mitigation_strategy(self) -> Dict[str, List[str]]:
        """Create bias detection and mitigation strategy"""
        return {
            "detection_methods": [
                "Source diversity analysis",
                "Perspective balance assessment",
                "Language neutrality check",
                "Commercial interest identification",
            ],
            "mitigation_strategies": [
                "Include opposing viewpoints",
                "Prioritize neutral sources",
                "Distinguish fact from opinion",
                "Identify funding sources",
                "Note potential conflicts of interest",
            ],
            "warning_signals": [
                "Single source dominance",
                "Emotional language",
                "Missing citations",
                "Commercial promotion",
                "Extreme claims without evidence",
            ],
        }

    def _define_success_metrics(self) -> Dict[str, Any]:
        """Define comprehensive success metrics for research"""
        return {
            "quantitative_metrics": {
                "source_count": {"target": ">=10", "weight": 0.2},
                "word_count": {"target": ">=5000", "weight": 0.1},
                "credibility_score": {"target": ">=0.7", "weight": 0.3},
                "diversity_score": {"target": ">=0.6", "weight": 0.2},
            },
            "qualitative_metrics": {
                "question_coverage": {"target": "80%", "weight": 0.1},
                "insight_quality": {"target": "high", "weight": 0.1},
            },
            "overall_threshold": 0.75,
            "critical_requirements": [
                "At least 3 high-credibility sources",
                "Coverage of main topic aspects",
                "No major information gaps",
                "Adequate cross-referencing",
            ],
        }

    def _get_default_source_types(self) -> List[str]:
        """Get default source types for research"""
        return [
            "academic",
            "news",
            "documentation",
            "analysis",
            "government",
            "reference",
            "expert_opinion",
        ]

    def _create_fallback_plan(
        self, query: str, depth: str, config: Dict[str, Any], max_sources: int
    ) -> Dict[str, Any]:
        """Create a basic fallback plan if LLM fails"""
        return {
            "id": f"fallback_plan_{int(datetime.now().timestamp())}",
            "query": query,
            "research_depth": depth,
            "config": config,
            "max_sources": max_sources,
            "created_at": datetime.now().isoformat(),
            "research_objectives": [
                f"Understand the fundamentals of {query}",
                f"Identify current developments in {query}",
                f"Analyze key challenges and opportunities",
            ],
            "key_questions": [
                f"What is {query}?",
                f"What are the latest developments in {query}?",
                f"Who are the key players/experts in {query}?",
                f"What are the main challenges in {query}?",
                f"What does the future hold for {query}?",
            ],
            "search_strategies": self._create_search_strategies(query, None),
            "source_priorities": self._define_source_priorities(),
            "quality_criteria": self._define_quality_criteria(),
            "execution_phases": self._create_execution_phases(config),
            "success_metrics": self._define_success_metrics(),
            "llm_analysis": "Fallback plan - LLM analysis unavailable",
            "note": "This is a fallback plan created due to LLM unavailability",
        }

    def validate_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and score a research plan"""
        score = 0.0
        issues = []

        # Check required components
        required_fields = [
            "research_objectives",
            "key_questions",
            "search_strategies",
            "source_priorities",
            "quality_criteria",
            "execution_phases",
        ]

        for field in required_fields:
            if field in plan and plan[field]:
                score += 1.0 / len(required_fields)
            else:
                issues.append(f"Missing or empty field: {field}")

        # Check quality of content
        if "research_objectives" in plan:
            if len(plan["research_objectives"]) >= 3:
                score += 0.1
            else:
                issues.append("Insufficient research objectives (need at least 3)")

        if "key_questions" in plan:
            if len(plan["key_questions"]) >= 5:
                score += 0.1
            else:
                issues.append("Insufficient key questions (need at least 5)")

        return {
            "is_valid": score >= 0.8,
            "score": round(score, 2),
            "issues": issues,
            "recommendations": [
                "Plan looks comprehensive" if score >= 0.8 else "Plan needs improvement",
                "Consider adding more detail to weak areas" if issues else "Good plan structure",
            ],
        }
