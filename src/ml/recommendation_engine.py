"""
Recommendation Engine Module

Provides personalized research recommendations based on user behavior,
successful patterns, and contextual analysis.
"""

import json
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from pathlib import Path

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

    # Mock numpy for basic operations
    class MockNumpy:
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0

        @staticmethod
        def dot(a, b):
            return sum(x * y for x, y in zip(a, b))

        @staticmethod
        def linalg_norm(data):
            return (sum(x * x for x in data)) ** 0.5

    np = MockNumpy()


class RecommendationType(Enum):
    """Types of recommendations"""

    QUERY_OPTIMIZATION = "query_optimization"
    SEARCH_STRATEGY = "search_strategy"
    SOURCE_FILTERING = "source_filtering"
    WORKFLOW_IMPROVEMENT = "workflow_improvement"
    TEMPLATE_SUGGESTION = "template_suggestion"
    COLLABORATION = "collaboration"


class RecommendationPriority(Enum):
    """Priority levels for recommendations"""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class UserProfile:
    """User behavior and preference profile"""

    user_id: str
    research_domains: List[str]
    preferred_sources: List[str]
    search_patterns: Dict[str, int]
    quality_preferences: Dict[str, float]
    workflow_preferences: Dict[str, Any]
    collaboration_style: str
    success_metrics: Dict[str, float]
    last_active: datetime = field(default_factory=datetime.now)


@dataclass
class ResearchContext:
    """Current research context for recommendations"""

    query: str
    domain: str
    urgency: str  # high, medium, low
    depth: str  # surface, medium, deep
    collaboration: bool
    quality_threshold: float
    available_time: int  # minutes
    previous_queries: List[str] = field(default_factory=list)


@dataclass
class ResearchRecommendation:
    """A specific research recommendation"""

    recommendation_id: str
    recommendation_type: RecommendationType
    priority: RecommendationPriority
    title: str
    description: str
    expected_benefit: str
    confidence_score: float
    implementation_effort: str  # low, medium, high
    estimated_improvement: float
    applicable_scenarios: List[str]
    implementation_steps: List[str]
    risk_factors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecommendationFeedback:
    """Feedback on recommendation effectiveness"""

    recommendation_id: str
    user_id: str
    was_helpful: bool
    improvement_observed: float
    implementation_difficulty: str
    additional_comments: str
    timestamp: datetime = field(default_factory=datetime.now)


class RecommendationEngine:
    """
    ML-enhanced recommendation engine that provides personalized
    research suggestions based on user behavior and successful patterns.
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the recommendation engine"""
        self.model_path = model_path or "data/ml_models/recommendation_engine"
        self.user_profiles: Dict[str, UserProfile] = {}
        self.recommendation_history: List[ResearchRecommendation] = []
        self.feedback_history: List[RecommendationFeedback] = []
        self.domain_expertise: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.recommendation_templates: Dict[RecommendationType, Dict] = {}

        # Initialize templates
        self._initialize_recommendation_templates()

        # Load existing data
        self._load_recommendation_data()

    def _initialize_recommendation_templates(self):
        """Initialize recommendation templates"""
        self.recommendation_templates = {
            RecommendationType.QUERY_OPTIMIZATION: {
                "keywords": ["expand", "refine", "focus", "clarify"],
                "benefits": ["Better search results", "More relevant sources", "Faster discovery"],
                "effort_levels": {
                    "synonym_expansion": "low",
                    "semantic_analysis": "medium",
                    "domain_mapping": "high",
                },
            },
            RecommendationType.SEARCH_STRATEGY: {
                "keywords": ["engine", "sequence", "timing", "parallel"],
                "benefits": ["Broader coverage", "Reduced redundancy", "Time efficiency"],
                "effort_levels": {
                    "engine_rotation": "low",
                    "parallel_search": "medium",
                    "adaptive_strategy": "high",
                },
            },
            RecommendationType.SOURCE_FILTERING: {
                "keywords": ["quality", "credibility", "relevance", "diversity"],
                "benefits": ["Higher quality results", "Reduced bias", "Better coverage"],
                "effort_levels": {
                    "basic_filters": "low",
                    "ML_scoring": "medium",
                    "expert_validation": "high",
                },
            },
            RecommendationType.WORKFLOW_IMPROVEMENT: {
                "keywords": ["efficiency", "automation", "optimization", "streamline"],
                "benefits": ["Time savings", "Consistency", "Reduced errors"],
                "effort_levels": {
                    "template_usage": "low",
                    "automation": "medium",
                    "custom_workflow": "high",
                },
            },
            RecommendationType.TEMPLATE_SUGGESTION: {
                "keywords": ["template", "structure", "framework", "approach"],
                "benefits": ["Structured approach", "Best practices", "Proven methods"],
                "effort_levels": {
                    "existing_template": "low",
                    "template_customization": "medium",
                    "new_template": "high",
                },
            },
            RecommendationType.COLLABORATION: {
                "keywords": ["team", "share", "review", "collaborate"],
                "benefits": ["Multiple perspectives", "Quality validation", "Knowledge sharing"],
                "effort_levels": {
                    "simple_sharing": "low",
                    "peer_review": "medium",
                    "full_collaboration": "high",
                },
            },
        }

    def _load_recommendation_data(self):
        """Load existing recommendation data"""
        try:
            data_path = Path(self.model_path)
            if data_path.exists():
                # Load user profiles
                if (data_path / "user_profiles.json").exists():
                    with open(data_path / "user_profiles.json", "r") as f:
                        profile_data = json.load(f)
                        self.user_profiles = {
                            k: UserProfile(
                                user_id=v["user_id"],
                                research_domains=v["research_domains"],
                                preferred_sources=v["preferred_sources"],
                                search_patterns=v["search_patterns"],
                                quality_preferences=v["quality_preferences"],
                                workflow_preferences=v["workflow_preferences"],
                                collaboration_style=v["collaboration_style"],
                                success_metrics=v["success_metrics"],
                                last_active=datetime.fromisoformat(v["last_active"]),
                            )
                            for k, v in profile_data.items()
                        }

                # Load domain expertise
                if (data_path / "domain_expertise.json").exists():
                    with open(data_path / "domain_expertise.json", "r") as f:
                        self.domain_expertise = json.load(f)

        except Exception as e:
            print(f"Warning: Could not load recommendation data: {e}")

    def _save_recommendation_data(self):
        """Save recommendation data to disk"""
        try:
            data_path = Path(self.model_path)
            data_path.mkdir(parents=True, exist_ok=True)

            # Save user profiles
            with open(data_path / "user_profiles.json", "w") as f:
                profile_data = {
                    k: {**v.__dict__, "last_active": v.last_active.isoformat()}
                    for k, v in self.user_profiles.items()
                }
                json.dump(profile_data, f, indent=2)

            # Save domain expertise
            with open(data_path / "domain_expertise.json", "w") as f:
                json.dump(dict(self.domain_expertise), f, indent=2)

        except Exception as e:
            print(f"Warning: Could not save recommendation data: {e}")

    def create_user_profile(self, user_id: str, initial_preferences: Dict[str, Any]) -> UserProfile:
        """Create a new user profile"""
        profile = UserProfile(
            user_id=user_id,
            research_domains=initial_preferences.get("domains", []),
            preferred_sources=initial_preferences.get("sources", []),
            search_patterns={},
            quality_preferences=initial_preferences.get(
                "quality_prefs", {"accuracy": 0.8, "objectivity": 0.7}
            ),
            workflow_preferences=initial_preferences.get("workflow_prefs", {}),
            collaboration_style=initial_preferences.get("collaboration", "independent"),
            success_metrics={},
        )

        self.user_profiles[user_id] = profile
        return profile

    def update_user_profile(self, user_id: str, session_data: Dict[str, Any]):
        """Update user profile based on research session"""
        if user_id not in self.user_profiles:
            self.create_user_profile(user_id, {})

        profile = self.user_profiles[user_id]

        # Update research domains
        if "domain" in session_data:
            if session_data["domain"] not in profile.research_domains:
                profile.research_domains.append(session_data["domain"])

        # Update search patterns
        if "search_engines" in session_data:
            for engine in session_data["search_engines"]:
                profile.search_patterns[engine] = profile.search_patterns.get(engine, 0) + 1

        # Update success metrics
        if "success_rate" in session_data:
            profile.success_metrics["average_success"] = (
                profile.success_metrics.get("average_success", 0.5) * 0.9
                + session_data["success_rate"] * 0.1
            )

        # Update workflow preferences
        if "template_used" in session_data and session_data.get("success_rate", 0) > 0.8:
            template = session_data["template_used"]
            profile.workflow_preferences[template] = (
                profile.workflow_preferences.get(template, 0) + 1
            )

        profile.last_active = datetime.now()

    def generate_recommendations(
        self, user_id: str, context: ResearchContext
    ) -> List[ResearchRecommendation]:
        """Generate personalized recommendations for a user and context"""
        recommendations = []

        # Get user profile
        profile = self.user_profiles.get(user_id)
        if not profile:
            profile = self.create_user_profile(user_id, {})

        # Generate different types of recommendations
        recommendations.extend(self._generate_query_recommendations(profile, context))
        recommendations.extend(self._generate_strategy_recommendations(profile, context))
        recommendations.extend(self._generate_workflow_recommendations(profile, context))
        recommendations.extend(self._generate_template_recommendations(profile, context))
        recommendations.extend(self._generate_collaboration_recommendations(profile, context))

        # Score and prioritize recommendations
        scored_recommendations = self._score_recommendations(recommendations, profile, context)

        # Sort by priority and confidence
        scored_recommendations.sort(
            key=lambda r: (r.priority.value == "high", r.confidence_score), reverse=True
        )

        return scored_recommendations[:10]  # Return top 10 recommendations

    def _generate_query_recommendations(
        self, profile: UserProfile, context: ResearchContext
    ) -> List[ResearchRecommendation]:
        """Generate query optimization recommendations"""
        recommendations = []

        # Analyze query characteristics
        query_length = len(context.query.split())
        query_complexity = self._assess_query_complexity(context.query)

        # Recommendation 1: Query expansion for short queries
        if query_length <= 3:
            rec = ResearchRecommendation(
                recommendation_id=f"query_expand_{hash(context.query) % 1000}",
                recommendation_type=RecommendationType.QUERY_OPTIMIZATION,
                priority=RecommendationPriority.HIGH,
                title="Expand Your Query",
                description="Your query is quite short. Adding specific terms could improve results.",
                expected_benefit="Find more targeted and relevant sources",
                confidence_score=0.8,
                implementation_effort="low",
                estimated_improvement=0.3,
                applicable_scenarios=["short queries", "broad searches"],
                implementation_steps=[
                    "Add specific domain terms",
                    "Include synonyms or related concepts",
                    "Specify time period if relevant",
                ],
            )
            recommendations.append(rec)

        # Recommendation 2: Query refinement for broad topics
        if query_complexity == "complex" and context.depth == "deep":
            rec = ResearchRecommendation(
                recommendation_id=f"query_refine_{hash(context.query) % 1000}",
                recommendation_type=RecommendationType.QUERY_OPTIMIZATION,
                priority=RecommendationPriority.MEDIUM,
                title="Refine Query Scope",
                description="Your query covers a broad topic. Consider narrowing the scope for deeper analysis.",
                expected_benefit="More focused and actionable results",
                confidence_score=0.7,
                implementation_effort="medium",
                estimated_improvement=0.25,
                applicable_scenarios=["complex queries", "deep research"],
                implementation_steps=[
                    "Identify specific aspects of interest",
                    "Break into sub-questions",
                    "Focus on specific time periods or regions",
                ],
            )
            recommendations.append(rec)

        return recommendations

    def _generate_strategy_recommendations(
        self, profile: UserProfile, context: ResearchContext
    ) -> List[ResearchRecommendation]:
        """Generate search strategy recommendations"""
        recommendations = []

        # Analyze user's search patterns
        preferred_engines = sorted(
            profile.search_patterns.items(), key=lambda x: x[1], reverse=True
        )

        # Recommendation: Engine diversification
        if len(preferred_engines) <= 2:
            rec = ResearchRecommendation(
                recommendation_id=f"strategy_diversify_{profile.user_id}",
                recommendation_type=RecommendationType.SEARCH_STRATEGY,
                priority=RecommendationPriority.MEDIUM,
                title="Diversify Search Engines",
                description="Using multiple search engines can uncover different perspectives and sources.",
                expected_benefit="More comprehensive and diverse results",
                confidence_score=0.75,
                implementation_effort="low",
                estimated_improvement=0.2,
                applicable_scenarios=["comprehensive research", "bias reduction"],
                implementation_steps=[
                    "Try academic databases for scholarly sources",
                    "Use news-specific engines for current events",
                    "Include specialized engines for your domain",
                ],
            )
            recommendations.append(rec)

        # Recommendation: Parallel search for urgent requests
        if context.urgency == "high":
            rec = ResearchRecommendation(
                recommendation_id=f"strategy_parallel_{hash(context.query) % 1000}",
                recommendation_type=RecommendationType.SEARCH_STRATEGY,
                priority=RecommendationPriority.HIGH,
                title="Use Parallel Search Strategy",
                description="For urgent research, run multiple searches simultaneously to save time.",
                expected_benefit="Faster results and better time efficiency",
                confidence_score=0.85,
                implementation_effort="medium",
                estimated_improvement=0.4,
                applicable_scenarios=["urgent research", "time constraints"],
                implementation_steps=[
                    "Launch searches on multiple engines simultaneously",
                    "Use different query variations",
                    "Aggregate results quickly",
                ],
            )
            recommendations.append(rec)

        return recommendations

    def _generate_workflow_recommendations(
        self, profile: UserProfile, context: ResearchContext
    ) -> List[ResearchRecommendation]:
        """Generate workflow improvement recommendations"""
        recommendations = []

        # Analyze user's success metrics
        avg_success = profile.success_metrics.get("average_success", 0.5)

        # Recommendation: Verification for low success rates
        if avg_success < 0.6:
            rec = ResearchRecommendation(
                recommendation_id=f"workflow_verify_{profile.user_id}",
                recommendation_type=RecommendationType.WORKFLOW_IMPROVEMENT,
                priority=RecommendationPriority.HIGH,
                title="Enable Source Verification",
                description="Your research success rate could be improved with better source verification.",
                expected_benefit="Higher quality and more reliable results",
                confidence_score=0.8,
                implementation_effort="low",
                estimated_improvement=0.3,
                applicable_scenarios=["quality improvement", "credibility concerns"],
                implementation_steps=[
                    "Enable automatic source verification",
                    "Set quality thresholds",
                    "Review verification reports",
                ],
            )
            recommendations.append(rec)

        # Recommendation: Automation for repetitive patterns
        if len(profile.workflow_preferences) > 3:
            most_used = max(profile.workflow_preferences.items(), key=lambda x: x[1])
            if most_used[1] > 5:  # Used more than 5 times
                rec = ResearchRecommendation(
                    recommendation_id=f"workflow_automate_{profile.user_id}",
                    recommendation_type=RecommendationType.WORKFLOW_IMPROVEMENT,
                    priority=RecommendationPriority.MEDIUM,
                    title="Automate Frequent Workflows",
                    description=f"You frequently use '{most_used[0]}' workflow. Consider automation.",
                    expected_benefit="Time savings and consistency",
                    confidence_score=0.7,
                    implementation_effort="medium",
                    estimated_improvement=0.25,
                    applicable_scenarios=["repetitive research", "efficiency improvement"],
                    implementation_steps=[
                        "Create automated workflow template",
                        "Set default parameters",
                        "Enable one-click execution",
                    ],
                )
                recommendations.append(rec)

        return recommendations

    def _generate_template_recommendations(
        self, profile: UserProfile, context: ResearchContext
    ) -> List[ResearchRecommendation]:
        """Generate template usage recommendations"""
        recommendations = []

        # Recommend templates based on domain
        domain_template_map = {
            "academic": "Academic Research Template",
            "market": "Market Research Template",
            "news": "News Research Template",
            "technical": "Technical Research Template",
            "competitive": "Competitive Analysis Template",
        }

        if context.domain in domain_template_map:
            template_name = domain_template_map[context.domain]
            rec = ResearchRecommendation(
                recommendation_id=f"template_{context.domain}_{hash(context.query) % 1000}",
                recommendation_type=RecommendationType.TEMPLATE_SUGGESTION,
                priority=RecommendationPriority.MEDIUM,
                title=f"Use {template_name}",
                description=f"The {template_name} is optimized for {context.domain} research.",
                expected_benefit="Structured approach and proven methodology",
                confidence_score=0.8,
                implementation_effort="low",
                estimated_improvement=0.2,
                applicable_scenarios=[f"{context.domain} research", "structured approach"],
                implementation_steps=[
                    f"Select {template_name}",
                    "Review template steps",
                    "Customize if needed",
                ],
            )
            recommendations.append(rec)

        return recommendations

    def _generate_collaboration_recommendations(
        self, profile: UserProfile, context: ResearchContext
    ) -> List[ResearchRecommendation]:
        """Generate collaboration recommendations"""
        recommendations = []

        # Recommend collaboration for complex research
        if context.depth == "deep" and not context.collaboration:
            rec = ResearchRecommendation(
                recommendation_id=f"collab_deep_{hash(context.query) % 1000}",
                recommendation_type=RecommendationType.COLLABORATION,
                priority=RecommendationPriority.LOW,
                title="Consider Collaborative Research",
                description="Deep research can benefit from multiple perspectives and expertise.",
                expected_benefit="Higher quality insights and reduced bias",
                confidence_score=0.6,
                implementation_effort="high",
                estimated_improvement=0.15,
                applicable_scenarios=["complex research", "multidisciplinary topics"],
                implementation_steps=[
                    "Identify potential collaborators",
                    "Define collaboration scope",
                    "Set up shared workspace",
                ],
            )
            recommendations.append(rec)

        return recommendations

    def _score_recommendations(
        self,
        recommendations: List[ResearchRecommendation],
        profile: UserProfile,
        context: ResearchContext,
    ) -> List[ResearchRecommendation]:
        """Score and prioritize recommendations"""
        for rec in recommendations:
            # Base score from confidence
            score = rec.confidence_score

            # Adjust based on user preferences
            if rec.recommendation_type.value in profile.workflow_preferences:
                score += 0.1  # User has used similar recommendations

            # Adjust based on context urgency
            if context.urgency == "high" and rec.implementation_effort == "low":
                score += 0.15
            elif context.urgency == "low" and rec.implementation_effort == "high":
                score -= 0.1

            # Update confidence score
            rec.confidence_score = min(score, 1.0)

            # Set priority based on score
            if score >= 0.8:
                rec.priority = RecommendationPriority.HIGH
            elif score >= 0.6:
                rec.priority = RecommendationPriority.MEDIUM
            else:
                rec.priority = RecommendationPriority.LOW

        return recommendations

    def _assess_query_complexity(self, query: str) -> str:
        """Assess complexity of research query"""
        word_count = len(query.split())

        # Check for complex indicators
        complex_indicators = ["and", "or", "vs", "versus", "compared to", "relationship between"]
        has_complex_indicators = any(indicator in query.lower() for indicator in complex_indicators)

        if word_count <= 3 and not has_complex_indicators:
            return "simple"
        elif word_count <= 8 or has_complex_indicators:
            return "medium"
        else:
            return "complex"

    def record_feedback(
        self,
        user_id: str,
        recommendation_id: str,
        was_helpful: bool,
        improvement_observed: float = 0.0,
        implementation_difficulty: str = "medium",
        comments: str = "",
    ):
        """Record user feedback on recommendations"""
        feedback = RecommendationFeedback(
            recommendation_id=recommendation_id,
            user_id=user_id,
            was_helpful=was_helpful,
            improvement_observed=improvement_observed,
            implementation_difficulty=implementation_difficulty,
            additional_comments=comments,
        )

        self.feedback_history.append(feedback)

        # Update domain expertise based on feedback
        # Find the original recommendation to get context
        # This would be implemented with proper recommendation tracking

        # Periodic save
        if len(self.feedback_history) % 10 == 0:
            self._save_recommendation_data()

    def get_recommendation_effectiveness(self) -> Dict[str, Any]:
        """Get statistics on recommendation effectiveness"""
        if not self.feedback_history:
            return {"status": "no_feedback_data"}

        # Calculate effectiveness metrics
        helpful_recommendations = sum(1 for f in self.feedback_history if f.was_helpful)
        total_feedback = len(self.feedback_history)
        helpfulness_rate = helpful_recommendations / total_feedback

        # Average improvement observed
        improvements = [
            f.improvement_observed for f in self.feedback_history if f.improvement_observed > 0
        ]
        avg_improvement = np.mean(improvements) if improvements else 0

        # Effectiveness by recommendation type
        type_effectiveness = defaultdict(list)
        for feedback in self.feedback_history:
            # This would need proper recommendation tracking to get type
            # For now, using a placeholder
            type_effectiveness["all"].append(feedback.was_helpful)

        return {
            "total_feedback_received": total_feedback,
            "helpfulness_rate": float(helpfulness_rate),
            "average_improvement": float(avg_improvement),
            "active_users": len(self.user_profiles),
            "recommendation_types_tested": len(type_effectiveness),
        }

    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about a specific user"""
        if user_id not in self.user_profiles:
            return {"status": "user_not_found"}

        profile = self.user_profiles[user_id]

        # Calculate user statistics
        total_searches = sum(profile.search_patterns.values())
        most_used_engine = (
            max(profile.search_patterns.items(), key=lambda x: x[1])[0]
            if profile.search_patterns
            else "none"
        )

        # Get user feedback
        user_feedback = [f for f in self.feedback_history if f.user_id == user_id]
        feedback_count = len(user_feedback)
        positive_feedback = sum(1 for f in user_feedback if f.was_helpful)

        return {
            "user_id": user_id,
            "research_domains": profile.research_domains,
            "total_searches": total_searches,
            "preferred_engine": most_used_engine,
            "success_rate": profile.success_metrics.get("average_success", 0.5),
            "collaboration_style": profile.collaboration_style,
            "feedback_provided": feedback_count,
            "positive_feedback_rate": (
                positive_feedback / feedback_count if feedback_count > 0 else 0
            ),
            "last_active": profile.last_active.isoformat(),
        }

    def export_recommendations(self, user_id: str, file_path: str):
        """Export user recommendations and insights"""
        user_insights = self.get_user_insights(user_id)
        effectiveness = self.get_recommendation_effectiveness()

        export_data = {
            "user_insights": user_insights,
            "system_effectiveness": effectiveness,
            "export_timestamp": datetime.now().isoformat(),
        }

        with open(file_path, "w") as f:
            json.dump(export_data, f, indent=2)
