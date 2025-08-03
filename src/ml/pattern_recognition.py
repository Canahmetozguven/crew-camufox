"""
Pattern Recognition Module

Identifies successful research patterns, automates repetitive tasks,
and provides insights for workflow optimization.
"""

import json
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from pathlib import Path
import re

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
        def std(data):
            if not data:
                return 0
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            return variance**0.5

    np = MockNumpy()


class PatternType(Enum):
    """Types of research patterns"""

    QUERY_PATTERN = "query_pattern"
    SEARCH_STRATEGY = "search_strategy"
    SOURCE_SELECTION = "source_selection"
    WORKFLOW_SEQUENCE = "workflow_sequence"
    ERROR_RECOVERY = "error_recovery"
    QUALITY_IMPROVEMENT = "quality_improvement"


class PatternContext(Enum):
    """Context in which patterns are observed"""

    ACADEMIC_RESEARCH = "academic_research"
    MARKET_ANALYSIS = "market_analysis"
    NEWS_GATHERING = "news_gathering"
    TECHNICAL_RESEARCH = "technical_research"
    COMPETITIVE_INTEL = "competitive_intel"


@dataclass
class PatternObservation:
    """Single observation of a research pattern"""

    pattern_id: str
    pattern_type: PatternType
    context: PatternContext
    actions: List[str]
    outcomes: Dict[str, float]
    metadata: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    success_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ResearchPattern:
    """Identified research pattern with success metrics"""

    pattern_id: str
    pattern_type: PatternType
    context: PatternContext
    description: str
    action_sequence: List[str]
    success_rate: float
    avg_quality_score: float
    usage_frequency: int
    last_observed: datetime
    confidence_score: float
    optimization_potential: float
    similar_patterns: List[str] = field(default_factory=list)
    effectiveness_factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class PatternRecommendation:
    """Recommendation based on identified patterns"""

    pattern_id: str
    recommendation_type: str
    description: str
    expected_improvement: float
    confidence: float
    applicable_contexts: List[PatternContext]
    implementation_steps: List[str]
    risk_factors: List[str] = field(default_factory=list)


class PatternRecognizer:
    """
    ML-enhanced pattern recognition system that identifies successful
    research workflows and suggests optimizations.
    """

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the pattern recognizer"""
        self.model_path = model_path or "data/ml_models/pattern_recognizer"
        self.observations: List[PatternObservation] = []
        self.identified_patterns: Dict[str, ResearchPattern] = {}
        self.pattern_templates: Dict[PatternType, Dict] = {}
        self.context_clusters: Dict[PatternContext, List[str]] = defaultdict(list)
        self.automation_candidates: List[ResearchPattern] = []

        # Initialize pattern templates
        self._initialize_pattern_templates()

        # Load existing data
        self._load_pattern_data()

    def _initialize_pattern_templates(self):
        """Initialize templates for different pattern types"""
        self.pattern_templates = {
            PatternType.QUERY_PATTERN: {
                "min_observations": 5,
                "success_threshold": 0.7,
                "similarity_threshold": 0.8,
                "key_features": ["query_structure", "keywords", "expansion_terms"],
            },
            PatternType.SEARCH_STRATEGY: {
                "min_observations": 3,
                "success_threshold": 0.75,
                "similarity_threshold": 0.7,
                "key_features": ["engine_sequence", "timing", "result_filtering"],
            },
            PatternType.SOURCE_SELECTION: {
                "min_observations": 4,
                "success_threshold": 0.8,
                "similarity_threshold": 0.75,
                "key_features": ["source_types", "quality_criteria", "diversity"],
            },
            PatternType.WORKFLOW_SEQUENCE: {
                "min_observations": 3,
                "success_threshold": 0.7,
                "similarity_threshold": 0.6,
                "key_features": ["step_sequence", "timing", "dependencies"],
            },
        }

    def _load_pattern_data(self):
        """Load existing pattern data"""
        try:
            data_path = Path(self.model_path)
            if data_path.exists():
                # Load observations
                if (data_path / "observations.json").exists():
                    with open(data_path / "observations.json", "r") as f:
                        obs_data = json.load(f)
                        self.observations = [
                            PatternObservation(
                                pattern_id=obs["pattern_id"],
                                pattern_type=PatternType(obs["pattern_type"]),
                                context=PatternContext(obs["context"]),
                                actions=obs["actions"],
                                outcomes=obs["outcomes"],
                                metadata=obs["metadata"],
                                timestamp=datetime.fromisoformat(obs["timestamp"]),
                                success_metrics=obs.get("success_metrics", {}),
                            )
                            for obs in obs_data[-500:]  # Keep recent observations
                        ]

                # Load identified patterns
                if (data_path / "patterns.json").exists():
                    with open(data_path / "patterns.json", "r") as f:
                        pattern_data = json.load(f)
                        self.identified_patterns = {
                            k: ResearchPattern(
                                pattern_id=v["pattern_id"],
                                pattern_type=PatternType(v["pattern_type"]),
                                context=PatternContext(v["context"]),
                                description=v["description"],
                                action_sequence=v["action_sequence"],
                                success_rate=v["success_rate"],
                                avg_quality_score=v["avg_quality_score"],
                                usage_frequency=v["usage_frequency"],
                                last_observed=datetime.fromisoformat(v["last_observed"]),
                                confidence_score=v["confidence_score"],
                                optimization_potential=v["optimization_potential"],
                                similar_patterns=v.get("similar_patterns", []),
                                effectiveness_factors=v.get("effectiveness_factors", {}),
                            )
                            for k, v in pattern_data.items()
                        }

        except Exception as e:
            print(f"Warning: Could not load pattern data: {e}")

    def _save_pattern_data(self):
        """Save pattern data to disk"""
        try:
            data_path = Path(self.model_path)
            data_path.mkdir(parents=True, exist_ok=True)

            # Save observations (recent only)
            with open(data_path / "observations.json", "w") as f:
                obs_data = [
                    {
                        "pattern_id": obs.pattern_id,
                        "pattern_type": obs.pattern_type.value,
                        "context": obs.context.value,
                        "actions": obs.actions,
                        "outcomes": obs.outcomes,
                        "metadata": obs.metadata,
                        "timestamp": obs.timestamp.isoformat(),
                        "success_metrics": obs.success_metrics,
                    }
                    for obs in self.observations[-500:]
                ]
                json.dump(obs_data, f, indent=2)

            # Save patterns
            with open(data_path / "patterns.json", "w") as f:
                pattern_data = {
                    k: {
                        "pattern_id": v.pattern_id,
                        "pattern_type": v.pattern_type.value,
                        "context": v.context.value,
                        "description": v.description,
                        "action_sequence": v.action_sequence,
                        "success_rate": v.success_rate,
                        "avg_quality_score": v.avg_quality_score,
                        "usage_frequency": v.usage_frequency,
                        "last_observed": v.last_observed.isoformat(),
                        "confidence_score": v.confidence_score,
                        "optimization_potential": v.optimization_potential,
                        "similar_patterns": v.similar_patterns,
                        "effectiveness_factors": v.effectiveness_factors,
                    }
                    for k, v in self.identified_patterns.items()
                }
                json.dump(pattern_data, f, indent=2)

        except Exception as e:
            print(f"Warning: Could not save pattern data: {e}")

    def observe_research_session(self, session_data: Dict[str, Any]) -> PatternObservation:
        """Observe and record a research session for pattern analysis"""

        # Extract pattern components
        pattern_id = self._generate_pattern_id(session_data)
        pattern_type = self._classify_pattern_type(session_data)
        context = self._determine_context(session_data)
        actions = self._extract_action_sequence(session_data)
        outcomes = self._extract_outcomes(session_data)
        metadata = self._extract_metadata(session_data)
        success_metrics = self._calculate_success_metrics(session_data)

        observation = PatternObservation(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            context=context,
            actions=actions,
            outcomes=outcomes,
            metadata=metadata,
            success_metrics=success_metrics,
        )

        self.observations.append(observation)

        # Trigger pattern analysis if enough observations
        if len(self.observations) % 10 == 0:
            self._analyze_patterns()

        return observation

    def _generate_pattern_id(self, session_data: Dict[str, Any]) -> str:
        """Generate unique pattern ID based on session characteristics"""
        # Extract key characteristics
        query_type = session_data.get("query_type", "unknown")
        search_engines = sorted(session_data.get("search_engines", []))
        agent_sequence = session_data.get("agent_sequence", [])

        # Create a hash-like ID
        components = [query_type] + search_engines + agent_sequence
        pattern_id = "_".join(components)[:50]  # Limit length

        return f"{pattern_id}_{datetime.now().strftime('%Y%m')}"

    def _classify_pattern_type(self, session_data: Dict[str, Any]) -> PatternType:
        """Classify the type of pattern observed"""
        # Analyze session data to determine pattern type
        if "query_optimization" in session_data:
            return PatternType.QUERY_PATTERN
        elif "search_strategy" in session_data:
            return PatternType.SEARCH_STRATEGY
        elif "source_selection" in session_data:
            return PatternType.SOURCE_SELECTION
        elif "workflow_steps" in session_data:
            return PatternType.WORKFLOW_SEQUENCE
        else:
            return PatternType.WORKFLOW_SEQUENCE  # Default

    def _determine_context(self, session_data: Dict[str, Any]) -> PatternContext:
        """Determine research context from session data"""
        query = session_data.get("query", "").lower()

        # Context classification based on keywords
        if any(term in query for term in ["academic", "research", "study", "paper"]):
            return PatternContext.ACADEMIC_RESEARCH
        elif any(term in query for term in ["market", "industry", "business", "company"]):
            return PatternContext.MARKET_ANALYSIS
        elif any(term in query for term in ["news", "current", "latest", "breaking"]):
            return PatternContext.NEWS_GATHERING
        elif any(term in query for term in ["technical", "implementation", "code", "api"]):
            return PatternContext.TECHNICAL_RESEARCH
        elif any(term in query for term in ["competitor", "competitive", "analysis"]):
            return PatternContext.COMPETITIVE_INTEL
        else:
            return PatternContext.ACADEMIC_RESEARCH  # Default

    def _extract_action_sequence(self, session_data: Dict[str, Any]) -> List[str]:
        """Extract sequence of actions from session data"""
        actions = []

        # Extract from different data sources
        if "workflow_steps" in session_data:
            actions.extend(session_data["workflow_steps"])

        if "search_engines" in session_data:
            for engine in session_data["search_engines"]:
                actions.append(f"search_{engine}")

        if "agent_sequence" in session_data:
            for agent in session_data["agent_sequence"]:
                actions.append(f"agent_{agent}")

        if "verification_steps" in session_data:
            actions.extend([f"verify_{step}" for step in session_data["verification_steps"]])

        return actions

    def _extract_outcomes(self, session_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract outcome metrics from session data"""
        outcomes = {}

        # Extract various outcome metrics
        outcomes["success_rate"] = session_data.get("success_rate", 0.0)
        outcomes["quality_score"] = session_data.get("quality_score", 0.0)
        outcomes["completion_time"] = session_data.get("completion_time", 0.0)
        outcomes["sources_found"] = float(session_data.get("sources_found", 0))
        outcomes["verification_score"] = session_data.get("verification_score", 0.0)
        outcomes["user_satisfaction"] = session_data.get("user_satisfaction", 0.0)

        return outcomes

    def _extract_metadata(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from session data"""
        metadata = {}

        # Extract relevant metadata
        metadata["query_length"] = len(session_data.get("query", ""))
        metadata["query_complexity"] = self._assess_query_complexity(session_data.get("query", ""))
        metadata["search_depth"] = session_data.get("search_depth", "medium")
        metadata["template_used"] = session_data.get("template_used", "none")
        metadata["verification_enabled"] = session_data.get("verification_enabled", False)
        metadata["collaboration_mode"] = session_data.get("collaboration_mode", "single")

        return metadata

    def _calculate_success_metrics(self, session_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive success metrics"""
        metrics = {}

        # Primary success metrics
        metrics["overall_success"] = session_data.get("success_rate", 0.0)
        metrics["quality_achievement"] = session_data.get("quality_score", 0.0)
        metrics["efficiency"] = 1.0 - min(session_data.get("completion_time", 60) / 120, 1.0)
        metrics["source_diversity"] = min(session_data.get("sources_found", 5) / 10, 1.0)

        # Composite success score
        weights = {
            "overall_success": 0.4,
            "quality_achievement": 0.3,
            "efficiency": 0.2,
            "source_diversity": 0.1,
        }
        metrics["composite_score"] = sum(metrics[k] * weights[k] for k in weights)

        return metrics

    def _assess_query_complexity(self, query: str) -> str:
        """Assess complexity of research query"""
        word_count = len(query.split())

        if word_count <= 3:
            return "simple"
        elif word_count <= 8:
            return "medium"
        else:
            return "complex"

    def _analyze_patterns(self):
        """Analyze observations to identify patterns"""
        # Group observations by pattern characteristics
        pattern_groups = defaultdict(list)

        for obs in self.observations:
            # Group by pattern type and context
            key = f"{obs.pattern_type.value}_{obs.context.value}"
            pattern_groups[key].append(obs)

        # Analyze each group for patterns
        for group_key, observations in pattern_groups.items():
            if len(observations) >= 3:  # Minimum observations for pattern
                pattern = self._identify_pattern_in_group(group_key, observations)
                if pattern:
                    self.identified_patterns[pattern.pattern_id] = pattern

        # Update automation candidates
        self._update_automation_candidates()

    def _identify_pattern_in_group(
        self, group_key: str, observations: List[PatternObservation]
    ) -> Optional[ResearchPattern]:
        """Identify pattern within a group of observations"""
        if len(observations) < 3:
            return None

        # Extract pattern components
        action_sequences = [obs.actions for obs in observations]
        success_rates = [obs.success_metrics.get("composite_score", 0) for obs in observations]
        quality_scores = [obs.outcomes.get("quality_score", 0) for obs in observations]

        # Find common action sequence
        common_actions = self._find_common_sequence(action_sequences)
        if not common_actions:
            return None

        # Calculate pattern metrics
        avg_success = np.mean(success_rates) if success_rates else 0
        avg_quality = np.mean(quality_scores) if quality_scores else 0

        # Check if pattern meets success threshold
        pattern_type_str, context_str = group_key.split("_", 1)
        pattern_type = PatternType(pattern_type_str)
        context = PatternContext(context_str)

        template = self.pattern_templates.get(pattern_type, {})
        success_threshold = template.get("success_threshold", 0.7)

        if avg_success < success_threshold:
            return None

        # Create pattern
        pattern_id = f"{group_key}_{len(self.identified_patterns)}"

        # Calculate confidence based on consistency
        confidence = self._calculate_pattern_confidence(observations)

        # Calculate optimization potential
        optimization_potential = self._calculate_optimization_potential(observations)

        # Find effectiveness factors
        effectiveness_factors = self._analyze_effectiveness_factors(observations)

        pattern = ResearchPattern(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            context=context,
            description=self._generate_pattern_description(common_actions, context),
            action_sequence=common_actions,
            success_rate=avg_success,
            avg_quality_score=avg_quality,
            usage_frequency=len(observations),
            last_observed=max(obs.timestamp for obs in observations),
            confidence_score=confidence,
            optimization_potential=optimization_potential,
            effectiveness_factors=effectiveness_factors,
        )

        return pattern

    def _find_common_sequence(self, sequences: List[List[str]]) -> List[str]:
        """Find common action sequence across multiple sequences"""
        if not sequences:
            return []

        # Find the most common subsequence
        # Simple approach: find actions that appear in majority of sequences
        action_counts = Counter()
        for sequence in sequences:
            action_counts.update(sequence)

        # Get actions that appear in at least 60% of sequences
        threshold = len(sequences) * 0.6
        common_actions = [action for action, count in action_counts.items() if count >= threshold]

        return common_actions

    def _calculate_pattern_confidence(self, observations: List[PatternObservation]) -> float:
        """Calculate confidence in pattern identification"""
        if not observations:
            return 0.0

        # Base confidence on consistency of outcomes
        success_scores = [obs.success_metrics.get("composite_score", 0) for obs in observations]

        if not success_scores:
            return 0.0

        # Calculate coefficient of variation (lower = more consistent)
        mean_score = np.mean(success_scores)
        std_score = np.std(success_scores)

        if mean_score == 0:
            return 0.0

        cv = std_score / mean_score
        confidence = max(0, 1.0 - cv)  # Higher consistency = higher confidence

        # Bonus for more observations
        observation_bonus = min(len(observations) / 10, 0.2)

        return min(confidence + observation_bonus, 1.0)

    def _calculate_optimization_potential(self, observations: List[PatternObservation]) -> float:
        """Calculate potential for optimizing this pattern"""
        if not observations:
            return 0.0

        # Look at variance in outcomes - high variance suggests optimization potential
        success_scores = [obs.success_metrics.get("composite_score", 0) for obs in observations]
        completion_times = [obs.outcomes.get("completion_time", 60) for obs in observations]

        # Normalize and calculate variance
        success_variance = np.std(success_scores) if success_scores else 0
        time_variance = np.std(completion_times) if completion_times else 0

        # Higher variance suggests more room for optimization
        optimization_potential = (success_variance + time_variance / 100) / 2

        return min(optimization_potential, 1.0)

    def _analyze_effectiveness_factors(
        self, observations: List[PatternObservation]
    ) -> Dict[str, float]:
        """Analyze factors that contribute to pattern effectiveness"""
        factors = {}

        # Analyze metadata correlations with success
        metadata_keys = set()
        for obs in observations:
            metadata_keys.update(obs.metadata.keys())

        for key in metadata_keys:
            if key in ["query_complexity", "search_depth", "template_used"]:
                # Calculate correlation with success (simplified)
                values = []
                successes = []

                for obs in observations:
                    if key in obs.metadata:
                        values.append(obs.metadata[key])
                        successes.append(obs.success_metrics.get("composite_score", 0))

                if len(values) >= 3:
                    # Simple correlation analysis
                    factors[key] = np.mean(successes) if successes else 0

        return factors

    def _generate_pattern_description(self, actions: List[str], context: PatternContext) -> str:
        """Generate human-readable pattern description"""
        action_summary = ", ".join(actions[:3])
        if len(actions) > 3:
            action_summary += f" and {len(actions) - 3} more steps"

        return f"Successful {context.value} pattern involving {action_summary}"

    def _update_automation_candidates(self):
        """Update list of patterns suitable for automation"""
        self.automation_candidates = []

        for pattern in self.identified_patterns.values():
            # Criteria for automation candidacy
            if (
                pattern.success_rate > 0.8
                and pattern.confidence_score > 0.7
                and pattern.usage_frequency >= 5
            ):
                self.automation_candidates.append(pattern)

        # Sort by optimization potential
        self.automation_candidates.sort(key=lambda p: p.optimization_potential, reverse=True)

    def get_pattern_recommendations(
        self, context: PatternContext, current_actions: List[str] = None
    ) -> List[PatternRecommendation]:
        """Get pattern-based recommendations for current research context"""
        recommendations = []

        # Find applicable patterns
        applicable_patterns = [
            pattern
            for pattern in self.identified_patterns.values()
            if pattern.context == context and pattern.success_rate > 0.7
        ]

        # Sort by success rate and confidence
        applicable_patterns.sort(key=lambda p: p.success_rate * p.confidence_score, reverse=True)

        for pattern in applicable_patterns[:5]:  # Top 5 recommendations
            recommendation = PatternRecommendation(
                pattern_id=pattern.pattern_id,
                recommendation_type="workflow_optimization",
                description=f"Apply {pattern.description} (Success rate: {pattern.success_rate:.1%})",
                expected_improvement=pattern.success_rate - 0.5,  # Improvement over baseline
                confidence=pattern.confidence_score,
                applicable_contexts=[pattern.context],
                implementation_steps=pattern.action_sequence,
                risk_factors=self._assess_pattern_risks(pattern),
            )
            recommendations.append(recommendation)

        return recommendations

    def _assess_pattern_risks(self, pattern: ResearchPattern) -> List[str]:
        """Assess potential risks of applying a pattern"""
        risks = []

        if pattern.confidence_score < 0.8:
            risks.append("Pattern confidence is moderate - results may vary")

        if pattern.usage_frequency < 5:
            risks.append("Limited usage history - pattern may not be robust")

        if pattern.optimization_potential > 0.7:
            risks.append("High variance in outcomes - results may be unpredictable")

        return risks

    def suggest_workflow_automation(self, min_usage: int = 5) -> List[ResearchPattern]:
        """Suggest patterns for workflow automation"""
        candidates = [
            pattern
            for pattern in self.automation_candidates
            if pattern.usage_frequency >= min_usage
        ]

        return candidates[:10]  # Top 10 automation candidates

    def get_pattern_insights(self) -> Dict[str, Any]:
        """Get insights about identified patterns"""
        if not self.identified_patterns:
            return {"status": "no_patterns_identified"}

        # Calculate insights
        pattern_count_by_type = Counter(
            p.pattern_type.value for p in self.identified_patterns.values()
        )
        pattern_count_by_context = Counter(
            p.context.value for p in self.identified_patterns.values()
        )

        avg_success_rate = np.mean([p.success_rate for p in self.identified_patterns.values()])
        avg_confidence = np.mean([p.confidence_score for p in self.identified_patterns.values()])

        return {
            "total_patterns": len(self.identified_patterns),
            "patterns_by_type": dict(pattern_count_by_type),
            "patterns_by_context": dict(pattern_count_by_context),
            "average_success_rate": float(avg_success_rate),
            "average_confidence": float(avg_confidence),
            "automation_candidates": len(self.automation_candidates),
            "total_observations": len(self.observations),
            "most_successful_context": (
                max(pattern_count_by_context, key=pattern_count_by_context.get)
                if pattern_count_by_context
                else None
            ),
        }

    def cleanup_old_patterns(self, days_threshold: int = 90):
        """Clean up old, unused patterns"""
        cutoff_date = datetime.now() - timedelta(days=days_threshold)

        # Remove old observations
        self.observations = [obs for obs in self.observations if obs.timestamp > cutoff_date]

        # Remove unused patterns
        patterns_to_remove = []
        for pattern_id, pattern in self.identified_patterns.items():
            if pattern.last_observed < cutoff_date and pattern.usage_frequency < 3:
                patterns_to_remove.append(pattern_id)

        for pattern_id in patterns_to_remove:
            del self.identified_patterns[pattern_id]

        # Update automation candidates
        self._update_automation_candidates()

        # Save cleaned data
        self._save_pattern_data()

    def export_patterns(self, file_path: str):
        """Export patterns to file for analysis or sharing"""
        export_data = {
            "patterns": {
                k: {
                    **v.__dict__,
                    "pattern_type": v.pattern_type.value,
                    "context": v.context.value,
                    "last_observed": v.last_observed.isoformat(),
                }
                for k, v in self.identified_patterns.items()
            },
            "insights": self.get_pattern_insights(),
            "export_timestamp": datetime.now().isoformat(),
        }

        with open(file_path, "w") as f:
            json.dump(export_data, f, indent=2)

    def import_patterns(self, file_path: str):
        """Import patterns from file"""
        try:
            with open(file_path, "r") as f:
                import_data = json.load(f)

            # Import patterns
            for k, v in import_data.get("patterns", {}).items():
                pattern = ResearchPattern(
                    pattern_id=v["pattern_id"],
                    pattern_type=PatternType(v["pattern_type"]),
                    context=PatternContext(v["context"]),
                    description=v["description"],
                    action_sequence=v["action_sequence"],
                    success_rate=v["success_rate"],
                    avg_quality_score=v["avg_quality_score"],
                    usage_frequency=v["usage_frequency"],
                    last_observed=datetime.fromisoformat(v["last_observed"]),
                    confidence_score=v["confidence_score"],
                    optimization_potential=v["optimization_potential"],
                    similar_patterns=v.get("similar_patterns", []),
                    effectiveness_factors=v.get("effectiveness_factors", {}),
                )
                self.identified_patterns[k] = pattern

            self._update_automation_candidates()
            self._save_pattern_data()

        except Exception as e:
            print(f"Error importing patterns: {e}")
