"""
Workflow Patterns Module

Provides workflow patterns and execution strategies for different research approaches
including linear research, deep dive analysis, comparative studies, and trend analysis.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .research_templates import ResearchTemplate, ResearchStep


class PatternType(Enum):
    """Types of workflow patterns available"""

    LINEAR = "linear"
    DEEP_DIVE = "deep_dive"
    COMPARATIVE = "comparative"
    TREND_ANALYSIS = "trend_analysis"
    ITERATIVE = "iterative"
    PARALLEL = "parallel"


@dataclass
class ExecutionStrategy:
    """Strategy for executing workflow steps"""

    name: str
    description: str
    parallel_execution: bool = False
    retry_attempts: int = 3
    timeout_seconds: int = 300
    priority_based: bool = True
    dependency_resolution: str = "strict"  # strict, relaxed, adaptive


@dataclass
class WorkflowMetrics:
    """Metrics for workflow execution"""

    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    skipped_steps: int = 0
    execution_time: float = 0.0
    success_rate: float = 0.0
    efficiency_score: float = 0.0


class WorkflowPattern(ABC):
    """Abstract base class for workflow patterns"""

    def __init__(self, pattern_type: PatternType, execution_strategy: ExecutionStrategy):
        self.pattern_type = pattern_type
        self.execution_strategy = execution_strategy
        self.metrics = WorkflowMetrics()

    @abstractmethod
    def organize_steps(self, steps: List["ResearchStep"]) -> List[List["ResearchStep"]]:
        """Organize steps according to pattern logic"""
        raise NotImplementedError("Subclasses must implement organize_steps")

    @abstractmethod
    def optimize_execution(
        self, step_groups: List[List["ResearchStep"]]
    ) -> List[List["ResearchStep"]]:
        """Optimize execution order and grouping"""
        raise NotImplementedError("Subclasses must implement optimize_execution")

    def validate_workflow(self, steps: List["ResearchStep"]) -> List[str]:
        """Validate workflow for pattern-specific requirements"""
        issues = []

        # Check for circular dependencies
        step_names = {step.name for step in steps}
        for step in steps:
            for dep in step.dependencies:
                if dep not in step_names:
                    issues.append(f"Step '{step.name}' depends on unknown step '{dep}'")

        return issues

    def estimate_execution_time(self, steps: List["ResearchStep"]) -> int:
        """Estimate total execution time for workflow"""
        if self.execution_strategy.parallel_execution:
            # For parallel execution, find the critical path
            return self._calculate_critical_path_time(steps)
        else:
            # For sequential execution, sum all step durations
            return sum(step.estimated_duration for step in steps)

    def _calculate_critical_path_time(self, steps: List["ResearchStep"]) -> int:
        """Calculate critical path execution time"""
        # Simplified critical path calculation
        step_times = {step.name: step.estimated_duration for step in steps}
        _ = {step.name: step.dependencies for step in steps}  # dependencies for future use

        # This would need proper critical path algorithm implementation
        # For now, return the maximum single step time as approximation
        return max(step_times.values()) if step_times else 0


class LinearResearchPattern(WorkflowPattern):
    """Linear research pattern - sequential execution of steps"""

    def __init__(self):
        strategy = ExecutionStrategy(
            name="Linear Sequential",
            description="Execute steps in sequential order with dependency resolution",
            parallel_execution=False,
            priority_based=True,
            dependency_resolution="strict",
        )
        super().__init__(PatternType.LINEAR, strategy)

    def organize_steps(self, steps: List["ResearchStep"]) -> List[List["ResearchStep"]]:
        """Organize steps in linear sequence"""
        # Sort by priority first, then by dependencies
        sorted_steps = self._topological_sort(steps)

        # Return as sequential groups (one step per group)
        return [[step] for step in sorted_steps]

    def optimize_execution(
        self, step_groups: List[List["ResearchStep"]]
    ) -> List[List["ResearchStep"]]:
        """Optimize linear execution order"""
        # For linear pattern, maintain the sequential order
        return step_groups

    def _topological_sort(self, steps: List["ResearchStep"]) -> List["ResearchStep"]:
        """Topologically sort steps based on dependencies"""
        # Create dependency graph
        _ = {step.name: step for step in steps}  # step_map for future use
        in_degree = {step.name: 0 for step in steps}

        # Calculate in-degrees
        for step in steps:
            for dep in step.dependencies:
                if dep in in_degree:
                    in_degree[step.name] += 1

        # Topological sort with priority consideration
        result = []
        available = [step for step in steps if in_degree[step.name] == 0]

        while available:
            # Sort by priority (lower number = higher priority)
            available.sort(key=lambda x: x.priority)
            current = available.pop(0)
            result.append(current)

            # Update in-degrees for dependent steps
            for step in steps:
                if current.name in step.dependencies:
                    in_degree[step.name] -= 1
                    if in_degree[step.name] == 0:
                        available.append(step)

        return result


class DeepDivePattern(WorkflowPattern):
    """Deep dive pattern - iterative refinement with feedback loops"""

    def __init__(self):
        strategy = ExecutionStrategy(
            name="Deep Dive Iterative",
            description="Iterative execution with refinement loops and quality gates",
            parallel_execution=False,
            priority_based=True,
            dependency_resolution="adaptive",
            retry_attempts=5,
        )
        super().__init__(PatternType.DEEP_DIVE, strategy)

    def organize_steps(self, steps: List["ResearchStep"]) -> List[List["ResearchStep"]]:
        """Organize steps in iterative phases"""
        # Group steps by research phase
        phases = {"exploration": [], "analysis": [], "synthesis": [], "validation": []}

        # Classify steps into phases
        for step in steps:
            step_name = step.name.lower()
            if any(keyword in step_name for keyword in ["collect", "gather", "search", "find"]):
                phases["exploration"].append(step)
            elif any(keyword in step_name for keyword in ["analyze", "examine", "investigate"]):
                phases["analysis"].append(step)
            elif any(keyword in step_name for keyword in ["synthesize", "combine", "integrate"]):
                phases["synthesis"].append(step)
            elif any(keyword in step_name for keyword in ["verify", "validate", "check", "review"]):
                phases["validation"].append(step)
            else:
                phases["analysis"].append(step)  # Default to analysis

        # Return phases as step groups
        result = []
        for phase_name in ["exploration", "analysis", "synthesis", "validation"]:
            if phases[phase_name]:
                result.append(phases[phase_name])

        return result

    def optimize_execution(
        self, step_groups: List[List["ResearchStep"]]
    ) -> List[List["ResearchStep"]]:
        """Optimize for iterative refinement"""
        # Add feedback loops between phases
        optimized = []

        for i, group in enumerate(step_groups):
            optimized.append(group)

            # Add quality check after each phase (except the last)
            if i < len(step_groups) - 1:
                quality_check = self._create_quality_gate(f"phase_{i+1}_quality")
                optimized.append([quality_check])

        return optimized

    def _create_quality_gate(self, name: str) -> "ResearchStep":
        """Create a quality gate step for phase validation"""
        from .research_templates import ResearchStep

        return ResearchStep(
            name=f"{name}_gate",
            description=f"Quality validation gate for {name}",
            agent_type="quality_validator",
            parameters={"validation_criteria": ["completeness", "accuracy", "relevance"]},
            expected_output="Quality validation report",
            estimated_duration=180,
            priority=1,
        )


class ComparativeAnalysisPattern(WorkflowPattern):
    """Comparative analysis pattern - parallel comparison with synthesis"""

    def __init__(self):
        strategy = ExecutionStrategy(
            name="Comparative Parallel",
            description="Parallel execution of comparative branches with final synthesis",
            parallel_execution=True,
            priority_based=True,
            dependency_resolution="relaxed",
        )
        super().__init__(PatternType.COMPARATIVE, strategy)

    def organize_steps(self, steps: List["ResearchStep"]) -> List[List["ResearchStep"]]:
        """Organize steps for comparative analysis"""
        # Separate into comparative branches and synthesis steps
        comparison_steps = []
        synthesis_steps = []

        for step in steps:
            step_name = step.name.lower()
            if any(
                keyword in step_name for keyword in ["compare", "versus", "contrast", "analyze"]
            ):
                comparison_steps.append(step)
            elif any(keyword in step_name for keyword in ["synthesize", "combine", "conclude"]):
                synthesis_steps.append(step)
            else:
                comparison_steps.append(step)  # Default to comparison

        # Create parallel groups for comparison, then synthesis
        result = []
        if comparison_steps:
            # Group comparison steps by similarity for parallel execution
            comparison_groups = self._group_similar_steps(comparison_steps)
            result.extend(comparison_groups)

        if synthesis_steps:
            result.append(synthesis_steps)

        return result

    def optimize_execution(
        self, step_groups: List[List["ResearchStep"]]
    ) -> List[List["ResearchStep"]]:
        """Optimize for parallel comparison"""
        # Ensure comparison steps can run in parallel
        optimized = []

        for group in step_groups:
            if len(group) > 1:
                # Remove dependencies within parallel groups
                for step in group:
                    step.dependencies = [
                        dep
                        for dep in step.dependencies
                        if not any(other.name == dep for other in group)
                    ]
            optimized.append(group)

        return optimized

    def _group_similar_steps(self, steps: List["ResearchStep"]) -> List[List["ResearchStep"]]:
        """Group similar steps for parallel execution"""
        # Simple grouping by agent type
        groups = {}
        for step in steps:
            agent_type = step.agent_type
            if agent_type not in groups:
                groups[agent_type] = []
            groups[agent_type].append(step)

        return list(groups.values())


class TrendAnalysisPattern(WorkflowPattern):
    """Trend analysis pattern - temporal analysis with pattern recognition"""

    def __init__(self):
        strategy = ExecutionStrategy(
            name="Trend Analysis Temporal",
            description="Temporal analysis with historical data and pattern recognition",
            parallel_execution=True,
            priority_based=True,
            dependency_resolution="adaptive",
        )
        super().__init__(PatternType.TREND_ANALYSIS, strategy)

    def organize_steps(self, steps: List["ResearchStep"]) -> List[List["ResearchStep"]]:
        """Organize steps for trend analysis"""
        # Group by temporal phases
        phases = {"historical": [], "current": [], "predictive": [], "synthesis": []}

        for step in steps:
            step_name = step.name.lower()
            if any(
                keyword in step_name for keyword in ["historical", "past", "archive", "timeline"]
            ):
                phases["historical"].append(step)
            elif any(keyword in step_name for keyword in ["current", "recent", "latest", "now"]):
                phases["current"].append(step)
            elif any(
                keyword in step_name for keyword in ["predict", "forecast", "future", "trend"]
            ):
                phases["predictive"].append(step)
            elif any(keyword in step_name for keyword in ["synthesize", "pattern", "analysis"]):
                phases["synthesis"].append(step)
            else:
                phases["current"].append(step)  # Default to current

        # Return temporal phases
        result = []
        for phase_name in ["historical", "current", "predictive", "synthesis"]:
            if phases[phase_name]:
                result.append(phases[phase_name])

        return result

    def optimize_execution(
        self, step_groups: List[List["ResearchStep"]]
    ) -> List[List["ResearchStep"]]:
        """Optimize for temporal analysis"""
        # Add pattern recognition steps between phases
        optimized = []

        for i, group in enumerate(step_groups):
            optimized.append(group)

            # Add pattern analysis after data collection phases
            if i < len(step_groups) - 1 and i < 2:  # After historical and current
                pattern_step = self._create_pattern_analysis_step(f"pattern_analysis_{i+1}")
                optimized.append([pattern_step])

        return optimized

    def _create_pattern_analysis_step(self, name: str) -> "ResearchStep":
        """Create pattern analysis step"""
        from .research_templates import ResearchStep

        return ResearchStep(
            name=name,
            description=f"Pattern analysis for {name}",
            agent_type="pattern_analyzer",
            parameters={"analysis_type": "temporal", "pattern_detection": True},
            expected_output="Pattern analysis report",
            estimated_duration=240,
            priority=2,
        )


class WorkflowManager:
    """Manager for workflow patterns and execution strategies"""

    def __init__(self):
        self.patterns: Dict[PatternType, WorkflowPattern] = {
            PatternType.LINEAR: LinearResearchPattern(),
            PatternType.DEEP_DIVE: DeepDivePattern(),
            PatternType.COMPARATIVE: ComparativeAnalysisPattern(),
            PatternType.TREND_ANALYSIS: TrendAnalysisPattern(),
        }

    def get_pattern(self, pattern_type: PatternType) -> Optional[WorkflowPattern]:
        """Get workflow pattern by type"""
        return self.patterns.get(pattern_type)

    def recommend_pattern(self, template: "ResearchTemplate") -> PatternType:
        """Recommend workflow pattern based on research template"""
        template_type = template.metadata.template_type.value.lower()

        # Pattern recommendation logic
        if "academic" in template_type:
            return PatternType.DEEP_DIVE
        elif "competitive" in template_type:
            return PatternType.COMPARATIVE
        elif "market" in template_type and "trend" in template.metadata.tags:
            return PatternType.TREND_ANALYSIS
        else:
            return PatternType.LINEAR

    def optimize_workflow(
        self, template: "ResearchTemplate", pattern_type: Optional[PatternType] = None
    ) -> List[List["ResearchStep"]]:
        """Optimize workflow for given template and pattern"""
        if pattern_type is None:
            pattern_type = self.recommend_pattern(template)

        pattern = self.get_pattern(pattern_type)
        if not pattern:
            # Fallback to linear pattern
            pattern = self.patterns[PatternType.LINEAR]

        # Validate workflow
        issues = pattern.validate_workflow(template.steps)
        if issues:
            # Log issues but continue with best effort
            pass

        # Organize and optimize steps
        step_groups = pattern.organize_steps(template.steps)
        optimized_groups = pattern.optimize_execution(step_groups)

        return optimized_groups

    def estimate_workflow_time(
        self, template: "ResearchTemplate", pattern_type: Optional[PatternType] = None
    ) -> int:
        """Estimate total workflow execution time"""
        if pattern_type is None:
            pattern_type = self.recommend_pattern(template)

        pattern = self.get_pattern(pattern_type)
        if not pattern:
            # Fallback calculation
            return sum(step.estimated_duration for step in template.steps)

        return pattern.estimate_execution_time(template.steps)

    def get_available_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available workflow patterns"""
        return {
            pattern_type.value: {
                "name": pattern.execution_strategy.name,
                "description": pattern.execution_strategy.description,
                "parallel_execution": pattern.execution_strategy.parallel_execution,
                "dependency_resolution": pattern.execution_strategy.dependency_resolution,
                "estimated_efficiency": self._calculate_pattern_efficiency(pattern),
            }
            for pattern_type, pattern in self.patterns.items()
        }

    def _calculate_pattern_efficiency(self, pattern: WorkflowPattern) -> float:
        """Calculate pattern efficiency score"""
        # Simplified efficiency calculation
        base_score = 0.7

        if pattern.execution_strategy.parallel_execution:
            base_score += 0.2

        if pattern.execution_strategy.dependency_resolution == "adaptive":
            base_score += 0.1

        return min(base_score, 1.0)
