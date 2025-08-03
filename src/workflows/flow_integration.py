"""
Flow Integration Module

Integrates CrewAI Flows 2.0 with existing workflow patterns and research templates.
Provides seamless migration path from traditional workflows to advanced flows.
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import logging

# Import existing workflow components
from src.templates.workflow_patterns import WorkflowPattern, WorkflowManager, PatternType
from src.templates.research_templates import ResearchTemplate, ResearchStep
from .flows_v2 import (
    EnhancedResearchFlowV2, 
    FlowOrchestrator, 
    AdvancedFlowState,
    FlowEvent, 
    FlowEventType, 
    FlowExecutionMode,
    WorkflowComposition,
    CREWAI_FLOWS_V2_AVAILABLE
)


@dataclass
class FlowMigrationConfig:
    """Configuration for migrating traditional workflows to Flows 2.0"""
    preserve_legacy_behavior: bool = True
    enable_advanced_features: bool = True
    fallback_to_traditional: bool = True
    monitoring_enabled: bool = True
    cache_enabled: bool = True


class FlowWorkflowAdapter:
    """Adapter to convert traditional workflow patterns to Flow 2.0 format"""
    
    def __init__(self, migration_config: Optional[FlowMigrationConfig] = None):
        self.config = migration_config or FlowMigrationConfig()
        self.logger = logging.getLogger(__name__)
        
    def convert_template_to_flow_context(self, template: ResearchTemplate) -> Dict[str, Any]:
        """Convert research template to flow context"""
        
        context = {
            "template_id": getattr(template.metadata, 'template_id', 'unknown'),
            "template_type": template.metadata.template_type.value,
            "scope": getattr(template.metadata, 'scope', 'general'),
            "difficulty": getattr(template.metadata, 'difficulty', 2),
            "estimated_duration": getattr(template.metadata, 'estimated_duration', 1800),
            
            # Flow-specific settings
            "execution_mode": self._determine_execution_mode(template),
            "priority": self._determine_priority(template),
            "max_parallel_steps": self._calculate_parallel_capacity(template),
            
            # Research parameters
            "search_terms": getattr(template, 'search_terms', []),
            "required_sources": self._extract_source_requirements(template),
            "depth_level": self._calculate_depth_level(template),
            "time_critical": getattr(template.metadata, 'difficulty', 2) > 3,
            
            # Quality settings
            "quality_threshold": 0.7,
            "enable_verification": True,
            "enable_bias_detection": True,
            
            # Steps conversion
            "workflow_steps": [self._convert_step_to_dict(step) for step in template.steps]
        }
        
        return context
    
    def _determine_execution_mode(self, template: ResearchTemplate) -> str:
        """Determine optimal execution mode for template"""
        
        # Analyze template characteristics
        step_dependencies = sum(1 for step in template.steps if step.dependencies)
        parallel_potential = len(template.steps) - step_dependencies
        
        difficulty = getattr(template.metadata, 'difficulty', 2)
        if difficulty >= 4:
            return FlowExecutionMode.ADAPTIVE.value
        elif parallel_potential >= 3:
            return FlowExecutionMode.PARALLEL.value
        else:
            return FlowExecutionMode.SEQUENTIAL.value
    
    def _determine_priority(self, template: ResearchTemplate) -> int:
        """Determine flow priority based on template"""
        difficulty = getattr(template.metadata, 'difficulty', 2)
        if difficulty >= 4:
            return 4  # Critical
        elif difficulty >= 3:
            return 3  # High
        elif difficulty >= 2:
            return 2  # Normal
        else:
            return 1  # Low
    
    def _calculate_parallel_capacity(self, template: ResearchTemplate) -> int:
        """Calculate maximum parallel steps capacity"""
        independent_steps = sum(1 for step in template.steps if not step.dependencies)
        return min(max(independent_steps // 2, 2), 6)  # Between 2-6 parallel steps
    
    def _extract_source_requirements(self, template: ResearchTemplate) -> List[str]:
        """Extract source requirements from template steps"""
        source_types = set()
        
        for step in template.steps:
            if "academic" in step.name.lower() or "scholar" in step.name.lower():
                source_types.add("academic")
            elif "news" in step.name.lower() or "current" in step.name.lower():
                source_types.add("news")
            elif "web" in step.name.lower() or "search" in step.name.lower():
                source_types.add("web")
            elif "expert" in step.name.lower() or "interview" in step.name.lower():
                source_types.add("expert")
        
        return list(source_types) if source_types else ["web", "academic"]
    
    def _calculate_depth_level(self, template: ResearchTemplate) -> int:
        """Calculate research depth level from template"""
        depth_indicators = [
            len(template.steps),
            getattr(template.metadata, 'difficulty', 2),
            getattr(template.metadata, 'estimated_duration', 1800) // 600  # Convert seconds to depth units
        ]
        
        return min(max(sum(depth_indicators) // 3, 1), 5)  # Depth level 1-5
    
    def _convert_step_to_dict(self, step: ResearchStep) -> Dict[str, Any]:
        """Convert research step to dictionary format"""
        return {
            "name": step.name,
            "description": step.description,
            "agent_type": step.agent_type,
            "parameters": step.parameters,
            "expected_output": step.expected_output,
            "estimated_duration": step.estimated_duration,
            "priority": step.priority,
            "dependencies": step.dependencies
        }


class EnhancedWorkflowManager:
    """Enhanced workflow manager with Flows 2.0 integration"""
    
    def __init__(self):
        self.traditional_manager = WorkflowManager()
        self.flow_orchestrator = FlowOrchestrator()
        self.adapter = FlowWorkflowAdapter()
        self.logger = logging.getLogger(__name__)
        
        # Track both traditional and flow-based executions
        self.execution_history: List[Dict[str, Any]] = []
        
    def execute_template(
        self, 
        template: ResearchTemplate, 
        use_flows_v2: Optional[bool] = None,
        pattern_type: Optional[PatternType] = None
    ) -> Dict[str, Any]:
        """Execute template using either traditional workflows or Flows 2.0"""
        
        # Auto-determine execution method if not specified
        if use_flows_v2 is None:
            use_flows_v2 = self._should_use_flows_v2(template)
        
        execution_start = datetime.now()
        
        try:
            if use_flows_v2 and CREWAI_FLOWS_V2_AVAILABLE:
                result = self._execute_with_flows_v2(template)
                execution_method = "flows_v2"
            else:
                result = self._execute_with_traditional(template, pattern_type)
                execution_method = "traditional"
            
            # Record execution metrics
            execution_time = (datetime.now() - execution_start).total_seconds()
            self._record_execution(template, execution_method, execution_time, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Execution failed: {e}")
            
            # Fallback to traditional workflow if Flows 2.0 fails
            if use_flows_v2 and execution_method == "flows_v2":
                self.logger.info("Falling back to traditional workflow execution")
                return self._execute_with_traditional(template, pattern_type)
            
            raise
    
    def _should_use_flows_v2(self, template: ResearchTemplate) -> bool:
        """Determine if template should use Flows 2.0"""
        
        # Criteria for using Flows 2.0
        criteria = [
            CREWAI_FLOWS_V2_AVAILABLE,  # Technical availability
            getattr(template.metadata, 'difficulty', 2) >= 3,  # Complex enough to benefit
            len(template.steps) >= 4,  # Sufficient complexity
            getattr(template.metadata, 'estimated_duration', 1800) >= 1800  # Long enough duration
        ]
        
        # Use Flows 2.0 if at least 3 criteria are met
        return sum(criteria) >= 3
    
    def _execute_with_flows_v2(self, template: ResearchTemplate) -> Dict[str, Any]:
        """Execute template using Flows 2.0"""
        
        # Convert template to flow context
        flow_context = self.adapter.convert_template_to_flow_context(template)
        
        # Create enhanced flow
        enhanced_flow = EnhancedResearchFlowV2()
        
        # Register with orchestrator
        flow_id = self.flow_orchestrator.register_flow(enhanced_flow)
        
        # Add event listeners for monitoring
        enhanced_flow.add_event_listener(
            FlowEventType.FLOW_STARTED, 
            self._on_flow_started
        )
        enhanced_flow.add_event_listener(
            FlowEventType.FLOW_COMPLETED, 
            self._on_flow_completed
        )
        enhanced_flow.add_event_listener(
            FlowEventType.STEP_FAILED, 
            self._on_step_failed
        )
        
        # Execute flow (this would normally use actual flow execution)
        # For now, simulate the execution with the enhanced flow logic
        init_result = enhanced_flow.initialize_enhanced_flow(flow_context)
        planning_result = enhanced_flow.dynamic_planning_phase(init_result)
        
        # Route execution based on planning
        execution_route = enhanced_flow.execution_strategy_router(planning_result)
        
        if execution_route == "deep_analysis_execution":
            execution_result = enhanced_flow.deep_analysis_execution(planning_result)
        elif execution_route == "parallel_execution":
            execution_result = enhanced_flow.parallel_execution(planning_result)
        else:
            execution_result = enhanced_flow.standard_execution(planning_result)
        
        # Synthesis and finalization
        synthesis_result = enhanced_flow.intelligent_synthesis(execution_result)
        final_result = enhanced_flow.quality_assurance_and_finalization(synthesis_result)
        
        # Add flow analytics
        final_result["flow_analytics"] = enhanced_flow.get_execution_analytics()
        final_result["execution_method"] = "flows_v2"
        final_result["flow_id"] = flow_id
        
        return final_result
    
    def _execute_with_traditional(
        self, 
        template: ResearchTemplate, 
        pattern_type: Optional[PatternType] = None
    ) -> Dict[str, Any]:
        """Execute template using traditional workflow patterns"""
        
        # Use traditional workflow manager
        optimized_groups = self.traditional_manager.optimize_workflow(template, pattern_type)
        estimated_time = self.traditional_manager.estimate_workflow_time(template, pattern_type)
        
        # Simulate traditional execution
        execution_result = {
            "execution_method": "traditional",
            "pattern_type": pattern_type.value if pattern_type else "auto",
            "workflow_groups": len(optimized_groups),
            "estimated_time": estimated_time,
            "total_steps": len(template.steps),
            "optimization_applied": True,
            "template_metadata": {
                "template_id": getattr(template.metadata, 'template_id', 'unknown'),
                "template_type": template.metadata.template_type.value,
                "difficulty": template.metadata.difficulty
            }
        }
        
        return execution_result
    
    def _record_execution(
        self, 
        template: ResearchTemplate, 
        method: str, 
        execution_time: float, 
        result: Dict[str, Any]
    ) -> None:
        """Record execution for analytics"""
        
        execution_record = {
            "timestamp": datetime.now().isoformat(),
            "template_id": getattr(template.metadata, 'template_id', 'unknown'),
            "template_type": template.metadata.template_type.value,
            "execution_method": method,
            "execution_time": execution_time,
            "success": "error" not in result,
            "step_count": len(template.steps),
            "difficulty": getattr(template.metadata, 'difficulty', 2),
            "result_summary": {
                "status": result.get("status", "unknown"),
                "quality_score": result.get("quality_metrics", {}).get("overall_score", 0),
                "confidence": result.get("confidence_score", 0)
            }
        }
        
        self.execution_history.append(execution_record)
    
    def _on_flow_started(self, event: FlowEvent) -> None:
        """Handle flow started event"""
        self.logger.info(f"Flow {event.flow_id} started")
    
    def _on_flow_completed(self, event: FlowEvent) -> None:
        """Handle flow completed event"""
        self.logger.info(f"Flow {event.flow_id} completed")
    
    def _on_step_failed(self, event: FlowEvent) -> None:
        """Handle step failed event"""
        self.logger.warning(f"Step {event.step_name} failed in flow {event.flow_id}")
    
    def get_execution_analytics(self) -> Dict[str, Any]:
        """Get comprehensive execution analytics"""
        
        if not self.execution_history:
            return {"message": "No execution history available"}
        
        # Analyze execution patterns
        total_executions = len(self.execution_history)
        flows_v2_executions = sum(1 for r in self.execution_history if r["execution_method"] == "flows_v2")
        traditional_executions = total_executions - flows_v2_executions
        
        # Calculate average execution times
        avg_execution_time = sum(r["execution_time"] for r in self.execution_history) / total_executions
        
        # Success rates
        successful_executions = sum(1 for r in self.execution_history if r["success"])
        success_rate = successful_executions / total_executions
        
        # Quality metrics
        quality_scores = [r["result_summary"]["quality_score"] for r in self.execution_history if r["result_summary"]["quality_score"] > 0]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return {
            "total_executions": total_executions,
            "execution_distribution": {
                "flows_v2": flows_v2_executions,
                "traditional": traditional_executions,
                "flows_v2_percentage": (flows_v2_executions / total_executions) * 100
            },
            "performance_metrics": {
                "average_execution_time": avg_execution_time,
                "success_rate": success_rate,
                "average_quality_score": avg_quality
            },
            "recent_executions": self.execution_history[-5:],  # Last 5 executions
            "flows_v2_available": CREWAI_FLOWS_V2_AVAILABLE
        }
    
    def compare_execution_methods(self, template: ResearchTemplate) -> Dict[str, Any]:
        """Compare traditional vs Flows 2.0 execution for a template"""
        
        comparison = {
            "template_analysis": {
                "template_id": getattr(template.metadata, 'template_id', 'unknown'),
                "complexity": getattr(template.metadata, 'difficulty', 2),
                "steps": len(template.steps),
                "estimated_duration": getattr(template.metadata, 'estimated_duration', 1800)
            },
            "execution_recommendations": {},
            "capability_comparison": {}
        }
        
        # Traditional workflow analysis
        traditional_pattern = self.traditional_manager.recommend_pattern(template)
        traditional_time = self.traditional_manager.estimate_workflow_time(template)
        
        comparison["traditional_workflow"] = {
            "recommended_pattern": traditional_pattern.value,
            "estimated_time": traditional_time,
            "parallel_capable": traditional_pattern in [PatternType.COMPARATIVE, PatternType.TREND_ANALYSIS],
            "optimization_level": "standard"
        }
        
        # Flows 2.0 analysis
        flow_context = self.adapter.convert_template_to_flow_context(template)
        
        comparison["flows_v2_workflow"] = {
            "execution_mode": flow_context["execution_mode"],
            "priority": flow_context["priority"],
            "parallel_capacity": flow_context["max_parallel_steps"],
            "adaptive_features": True,
            "monitoring_enabled": True,
            "optimization_level": "advanced"
        }
        
        # Recommendations
        should_use_flows = self._should_use_flows_v2(template)
        comparison["execution_recommendations"] = {
            "recommended_method": "flows_v2" if should_use_flows else "traditional",
            "reasoning": self._get_recommendation_reasoning(template, should_use_flows),
            "fallback_available": True
        }
        
        return comparison
    
    def _get_recommendation_reasoning(self, template: ResearchTemplate, use_flows: bool) -> List[str]:
        """Get reasoning for execution method recommendation"""
        
        reasons = []
        
        if use_flows:
            difficulty = getattr(template.metadata, 'difficulty', 2)
            estimated_duration = getattr(template.metadata, 'estimated_duration', 1800)
            
            if difficulty >= 3:
                reasons.append("High complexity benefits from advanced flow features")
            if len(template.steps) >= 4:
                reasons.append("Multiple steps can leverage parallel execution")
            if estimated_duration >= 1800:
                reasons.append("Long duration benefits from monitoring and checkpointing")
        else:
            difficulty = getattr(template.metadata, 'difficulty', 2)
            if difficulty < 3:
                reasons.append("Simple template doesn't require advanced features")
            if not CREWAI_FLOWS_V2_AVAILABLE:
                reasons.append("Flows 2.0 not available, using traditional workflow")
            if len(template.steps) < 4:
                reasons.append("Few steps make traditional workflow more suitable")
        
        return reasons


# Export integration classes
__all__ = [
    "FlowWorkflowAdapter",
    "EnhancedWorkflowManager",
    "FlowMigrationConfig"
]