"""
ML Coordinator Module

Coordinates all ML components and provides a unified interface
for ML-enhanced research capabilities.
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime
from pathlib import Path

from src.ml.query_optimizer import QueryOptimizer, QueryExpansion, QueryMetrics
from src.ml.source_predictor import SourceQualityPredictor, QualityMetrics, QualityFeatures
from src.ml.pattern_recognition import PatternRecognizer, ResearchPattern, PatternObservation
from src.ml.recommendation_engine import RecommendationEngine, ResearchRecommendation, ResearchContext


class MLCapability(Enum):
    """Available ML capabilities"""

    QUERY_OPTIMIZATION = "query_optimization"
    SOURCE_QUALITY_PREDICTION = "source_quality_prediction"
    PATTERN_RECOGNITION = "pattern_recognition"
    RESEARCH_RECOMMENDATIONS = "research_recommendations"


@dataclass
class MLConfiguration:
    """Configuration for ML capabilities"""

    enabled_capabilities: List[MLCapability]
    model_path: str
    learning_rate: float = 0.01
    feedback_threshold: int = 10
    auto_save_interval: int = 50
    quality_threshold: float = 0.7


@dataclass
class MLInsights:
    """Comprehensive ML insights"""

    query_insights: Dict[str, Any]
    quality_insights: Dict[str, Any]
    pattern_insights: Dict[str, Any]
    recommendation_insights: Dict[str, Any]
    overall_performance: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class MLCoordinator:
    """
    Central coordinator for all ML-enhanced research capabilities.
    Provides unified interface and manages learning across components.
    """

    def __init__(self, config: MLConfiguration):
        """Initialize ML coordinator with configuration"""
        self.config = config
        self.components: Dict[MLCapability, Any] = {}
        self.session_counter = 0
        self.learning_enabled = True

        # Initialize components based on configuration
        self._initialize_components()

        # Load coordinator state
        self._load_coordinator_state()

    def _initialize_components(self):
        """Initialize ML components based on configuration"""
        model_path = Path(self.config.model_path)

        if MLCapability.QUERY_OPTIMIZATION in self.config.enabled_capabilities:
            self.components[MLCapability.QUERY_OPTIMIZATION] = QueryOptimizer(
                model_path=str(model_path / "query_optimizer")
            )

        if MLCapability.SOURCE_QUALITY_PREDICTION in self.config.enabled_capabilities:
            self.components[MLCapability.SOURCE_QUALITY_PREDICTION] = SourceQualityPredictor(
                model_path=str(model_path / "quality_predictor")
            )

        if MLCapability.PATTERN_RECOGNITION in self.config.enabled_capabilities:
            self.components[MLCapability.PATTERN_RECOGNITION] = PatternRecognizer(
                model_path=str(model_path / "pattern_recognizer")
            )

        if MLCapability.RESEARCH_RECOMMENDATIONS in self.config.enabled_capabilities:
            self.components[MLCapability.RESEARCH_RECOMMENDATIONS] = RecommendationEngine(
                model_path=str(model_path / "recommendation_engine")
            )

    def _load_coordinator_state(self):
        """Load coordinator state"""
        try:
            state_path = Path(self.config.model_path) / "coordinator_state.json"
            if state_path.exists():
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    self.session_counter = state.get("session_counter", 0)
                    self.learning_enabled = state.get("learning_enabled", True)
        except Exception as e:
            print(f"Warning: Could not load coordinator state: {e}")

    def _save_coordinator_state(self):
        """Save coordinator state"""
        try:
            state_path = Path(self.config.model_path) / "coordinator_state.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)

            state = {
                "session_counter": self.session_counter,
                "learning_enabled": self.learning_enabled,
                "last_updated": datetime.now().isoformat(),
            }

            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save coordinator state: {e}")

    async def optimize_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Optimize a research query using ML"""
        if MLCapability.QUERY_OPTIMIZATION not in self.components:
            return {"optimized_query": query, "expansion": None, "status": "not_enabled"}

        optimizer = self.components[MLCapability.QUERY_OPTIMIZATION]

        try:
            # Get query optimization
            optimized_query, expansion = optimizer.optimize_query(query)

            return {
                "original_query": query,
                "optimized_query": optimized_query,
                "expansion": {
                    "expanded_terms": expansion.expanded_terms,
                    "synonyms": expansion.synonyms,
                    "confidence_score": expansion.confidence_score,
                    "estimated_improvement": expansion.estimated_improvement,
                },
                "status": "success",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def predict_source_quality(
        self, url: str, content: str, content_type: str = "general"
    ) -> Dict[str, Any]:
        """Predict source quality using ML"""
        if MLCapability.SOURCE_QUALITY_PREDICTION not in self.components:
            return {"status": "not_enabled"}

        predictor = self.components[MLCapability.SOURCE_QUALITY_PREDICTION]

        try:
            # Extract features
            from .source_predictor import ContentType

            # Map string to ContentType enum
            content_type_map = {
                "academic": ContentType.ACADEMIC_PAPER,
                "news": ContentType.NEWS_ARTICLE,
                "blog": ContentType.BLOG_POST,
                "government": ContentType.GOVERNMENT_DOC,
                "corporate": ContentType.CORPORATE_INFO,
                "social": ContentType.SOCIAL_MEDIA,
                "technical": ContentType.TECHNICAL_DOC,
            }

            content_type_enum = content_type_map.get(content_type, ContentType.BLOG_POST)
            features = predictor.extract_features(url, content)
            quality_metrics = predictor.predict_quality(features, content_type_enum)

            return {
                "url": url,
                "overall_quality": quality_metrics.overall_quality,
                "credibility_level": quality_metrics.credibility_level.value,
                "quality_scores": {
                    "authority": quality_metrics.authority_score,
                    "accuracy": quality_metrics.accuracy_score,
                    "objectivity": quality_metrics.objectivity_score,
                    "currency": quality_metrics.currency_score,
                    "coverage": quality_metrics.coverage_score,
                    "relevance": quality_metrics.relevance_score,
                },
                "confidence_interval": quality_metrics.confidence_interval,
                "quality_factors": quality_metrics.quality_factors,
                "status": "success",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def analyze_research_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze research session for patterns"""
        if MLCapability.PATTERN_RECOGNITION not in self.components:
            return {"status": "not_enabled"}

        recognizer = self.components[MLCapability.PATTERN_RECOGNITION]

        try:
            # Record observation
            observation = recognizer.observe_research_session(session_data)

            # Get recommendations if available
            recommendations = []
            if MLCapability.RESEARCH_RECOMMENDATIONS in self.components:
                rec_engine = self.components[MLCapability.RESEARCH_RECOMMENDATIONS]
                context = self._create_research_context(session_data)
                user_id = session_data.get("user_id", "anonymous")
                recommendations = rec_engine.generate_recommendations(user_id, context)

            return {
                "observation_id": observation.pattern_id,
                "pattern_type": observation.pattern_type.value,
                "context": observation.context.value,
                "success_metrics": observation.success_metrics,
                "recommendations": [
                    {
                        "id": rec.recommendation_id,
                        "type": rec.recommendation_type.value,
                        "title": rec.title,
                        "description": rec.description,
                        "confidence": rec.confidence_score,
                        "priority": rec.priority.value,
                    }
                    for rec in recommendations[:5]
                ],
                "status": "success",
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def get_research_recommendations(
        self, user_id: str, research_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get personalized research recommendations"""
        if MLCapability.RESEARCH_RECOMMENDATIONS not in self.components:
            return []

        rec_engine = self.components[MLCapability.RESEARCH_RECOMMENDATIONS]

        try:
            context = self._create_research_context(research_context)
            recommendations = rec_engine.generate_recommendations(user_id, context)

            return [
                {
                    "id": rec.recommendation_id,
                    "type": rec.recommendation_type.value,
                    "priority": rec.priority.value,
                    "title": rec.title,
                    "description": rec.description,
                    "expected_benefit": rec.expected_benefit,
                    "confidence": rec.confidence_score,
                    "implementation_effort": rec.implementation_effort,
                    "estimated_improvement": rec.estimated_improvement,
                    "implementation_steps": rec.implementation_steps,
                }
                for rec in recommendations
            ]
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return []

    def learn_from_feedback(
        self,
        session_data: Dict[str, Any],
        outcomes: Dict[str, Any],
        feedback: Optional[Dict] = None,
    ):
        """Learn from research session outcomes and user feedback"""
        if not self.learning_enabled:
            return

        try:
            # Update query optimizer
            if MLCapability.QUERY_OPTIMIZATION in self.components:
                optimizer = self.components[MLCapability.QUERY_OPTIMIZATION]

                if "query" in session_data and "success_rate" in outcomes:
                    from .query_optimizer import QueryType

                    # Create query metrics
                    query_metrics = QueryMetrics(
                        query=session_data["query"],
                        query_type=QueryType.GENERAL,  # Could be classified
                        search_engines_used=session_data.get("search_engines", []),
                        total_results=outcomes.get("total_results", 0),
                        relevant_results=outcomes.get("relevant_results", 0),
                        quality_score=outcomes.get("quality_score", 0.0),
                        execution_time=outcomes.get("execution_time", 0.0),
                        success_rate=outcomes.get("success_rate", 0.0),
                    )

                    optimizer.learn_from_results(session_data["query"], query_metrics)

            # Update source quality predictor
            if MLCapability.SOURCE_QUALITY_PREDICTION in self.components and feedback:
                predictor = self.components[MLCapability.SOURCE_QUALITY_PREDICTION]

                # This would require source-specific feedback
                # Implementation depends on feedback structure

            # Update recommendation engine
            if MLCapability.RESEARCH_RECOMMENDATIONS in self.components:
                rec_engine = self.components[MLCapability.RESEARCH_RECOMMENDATIONS]
                user_id = session_data.get("user_id", "anonymous")
                rec_engine.update_user_profile(user_id, session_data)

                # Record recommendation feedback if available
                if feedback and "recommendation_feedback" in feedback:
                    for rec_feedback in feedback["recommendation_feedback"]:
                        rec_engine.record_feedback(
                            user_id=user_id,
                            recommendation_id=rec_feedback.get("recommendation_id", ""),
                            was_helpful=rec_feedback.get("was_helpful", False),
                            improvement_observed=rec_feedback.get("improvement", 0.0),
                            implementation_difficulty=rec_feedback.get("difficulty", "medium"),
                            comments=rec_feedback.get("comments", ""),
                        )

            # Increment session counter and periodic save
            self.session_counter += 1
            if self.session_counter % self.config.auto_save_interval == 0:
                self._save_all_components()

        except Exception as e:
            print(f"Error in learning from feedback: {e}")

    def _create_research_context(self, context_data: Dict[str, Any]) -> ResearchContext:
        """Create ResearchContext from context data"""
        return ResearchContext(
            query=context_data.get("query", ""),
            domain=context_data.get("domain", "general"),
            urgency=context_data.get("urgency", "medium"),
            depth=context_data.get("depth", "medium"),
            collaboration=context_data.get("collaboration", False),
            quality_threshold=context_data.get("quality_threshold", 0.7),
            available_time=context_data.get("available_time", 60),
            previous_queries=context_data.get("previous_queries", []),
        )

    def _save_all_components(self):
        """Save state for all components"""
        try:
            # Save individual components (they handle their own saving)
            for capability, component in self.components.items():
                if hasattr(component, "_save_optimization_data"):
                    component._save_optimization_data()
                elif hasattr(component, "_save_model_data"):
                    component._save_model_data()
                elif hasattr(component, "_save_pattern_data"):
                    component._save_pattern_data()
                elif hasattr(component, "_save_recommendation_data"):
                    component._save_recommendation_data()

            # Save coordinator state
            self._save_coordinator_state()

        except Exception as e:
            print(f"Error saving ML components: {e}")

    def get_ml_insights(self) -> MLInsights:
        """Get comprehensive ML insights across all components"""
        insights = MLInsights(
            query_insights={},
            quality_insights={},
            pattern_insights={},
            recommendation_insights={},
            overall_performance={},
        )

        try:
            # Query optimization insights
            if MLCapability.QUERY_OPTIMIZATION in self.components:
                optimizer = self.components[MLCapability.QUERY_OPTIMIZATION]
                insights.query_insights = optimizer.get_optimization_stats()

            # Quality prediction insights
            if MLCapability.SOURCE_QUALITY_PREDICTION in self.components:
                predictor = self.components[MLCapability.SOURCE_QUALITY_PREDICTION]
                insights.quality_insights = predictor.get_prediction_stats()

            # Pattern recognition insights
            if MLCapability.PATTERN_RECOGNITION in self.components:
                recognizer = self.components[MLCapability.PATTERN_RECOGNITION]
                insights.pattern_insights = recognizer.get_pattern_insights()

            # Recommendation insights
            if MLCapability.RESEARCH_RECOMMENDATIONS in self.components:
                rec_engine = self.components[MLCapability.RESEARCH_RECOMMENDATIONS]
                insights.recommendation_insights = rec_engine.get_recommendation_effectiveness()

            # Overall performance
            insights.overall_performance = {
                "total_sessions": self.session_counter,
                "enabled_capabilities": [cap.value for cap in self.config.enabled_capabilities],
                "learning_enabled": self.learning_enabled,
                "model_path": self.config.model_path,
            }

        except Exception as e:
            print(f"Error generating ML insights: {e}")

        return insights

    def enable_learning(self, enabled: bool = True):
        """Enable or disable learning across all components"""
        self.learning_enabled = enabled
        self._save_coordinator_state()

    def reset_ml_models(self, capability: Optional[MLCapability] = None):
        """Reset ML models (use with caution)"""
        if capability:
            # Reset specific capability
            if capability in self.components:
                # This would require reset methods in each component
                print(f"Reset capability: {capability.value}")
        else:
            # Reset all capabilities
            print("Reset all ML capabilities")
            self.session_counter = 0

    def export_ml_data(self, export_path: str):
        """Export all ML data for backup or analysis"""
        export_data = {
            "coordinator_state": {
                "session_counter": self.session_counter,
                "learning_enabled": self.learning_enabled,
                "configuration": {
                    "enabled_capabilities": [cap.value for cap in self.config.enabled_capabilities],
                    "model_path": self.config.model_path,
                    "learning_rate": self.config.learning_rate,
                },
            },
            "ml_insights": self.get_ml_insights().__dict__,
            "export_timestamp": datetime.now().isoformat(),
        }

        # Convert datetime objects to strings
        if "timestamp" in export_data["ml_insights"]:
            export_data["ml_insights"]["timestamp"] = export_data["ml_insights"][
                "timestamp"
            ].isoformat()

        export_path_obj = Path(export_path)
        export_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(export_path_obj, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)

    def get_capability_status(self) -> Dict[str, Any]:
        """Get status of all ML capabilities"""
        status = {}

        for capability in MLCapability:
            if capability in self.components:
                status[capability.value] = {
                    "enabled": True,
                    "initialized": True,
                    "component_type": type(self.components[capability]).__name__,
                }
            else:
                status[capability.value] = {
                    "enabled": False,
                    "initialized": False,
                    "component_type": None,
                }

        return {
            "capabilities": status,
            "learning_enabled": self.learning_enabled,
            "total_sessions": self.session_counter,
            "coordinator_active": True,
        }
