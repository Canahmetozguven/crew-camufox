"""
Enhanced Workflow System

Comprehensive workflow orchestration with traditional patterns and advanced CrewAI Flows 2.0 integration.
Provides seamless migration from existing workflows to advanced flow-based execution.
"""

from typing import Optional

# Import traditional workflow components
from src.templates.workflow_patterns import (
    WorkflowPattern,
    WorkflowManager,
    PatternType,
    LinearResearchPattern,
    DeepDivePattern,
    ComparativeAnalysisPattern,
    TrendAnalysisPattern,
    ExecutionStrategy,
    WorkflowMetrics
)

# Import original enhanced research flow
from .enhanced_research_flow import (
    EnhancedResearchFlow,
    ResearchContext,
    ResearchState,
    CREWAI_FLOWS_AVAILABLE
)

# Import Flows 2.0 components
try:
    from .flows_v2 import (
        EnhancedResearchFlowV2,
        FlowOrchestrator,
        AdvancedFlowState,
        FlowEvent,
        FlowEventType,
        FlowExecutionMode,
        FlowPriority,
        WorkflowComposition,
        CREWAI_FLOWS_V2_AVAILABLE
    )
    
    from .flow_integration import (
        FlowWorkflowAdapter,
        EnhancedWorkflowManager,
        FlowMigrationConfig
    )
    
    FLOWS_V2_INTEGRATION_AVAILABLE = True
except ImportError:
    # Fallback if Flows 2.0 components are not available
    FLOWS_V2_INTEGRATION_AVAILABLE = False
    CREWAI_FLOWS_V2_AVAILABLE = False
    
    # Provide fallback classes
    class EnhancedResearchFlowV2:
        def __init__(self):
            raise ImportError("Flows 2.0 not available")
    
    class FlowOrchestrator:
        def __init__(self):
            raise ImportError("Flows 2.0 not available")
    
    class EnhancedWorkflowManager:
        def __init__(self):
            raise ImportError("Flows 2.0 integration not available")


def get_workflow_manager(use_flows_v2: Optional[bool] = None):
    """
    Get appropriate workflow manager based on availability and preference.
    
    Args:
        use_flows_v2: Whether to use Flows 2.0 (None for auto-detection)
    
    Returns:
        Workflow manager instance
    """
    if use_flows_v2 is None:
        # Auto-detect best available option
        use_flows_v2 = FLOWS_V2_INTEGRATION_AVAILABLE
    
    if use_flows_v2 and FLOWS_V2_INTEGRATION_AVAILABLE:
        return EnhancedWorkflowManager()
    else:
        return WorkflowManager()


def get_research_flow(use_flows_v2: Optional[bool] = None):
    """
    Get appropriate research flow based on availability and preference.
    
    Args:
        use_flows_v2: Whether to use Flows 2.0 (None for auto-detection)
    
    Returns:
        Research flow instance
    """
    if use_flows_v2 is None:
        # Auto-detect best available option
        use_flows_v2 = FLOWS_V2_INTEGRATION_AVAILABLE
    
    if use_flows_v2 and FLOWS_V2_INTEGRATION_AVAILABLE:
        return EnhancedResearchFlowV2()
    else:
        return EnhancedResearchFlow()


# Export all available components
__all__ = [
    # Traditional workflow patterns
    "WorkflowPattern",
    "WorkflowManager",
    "PatternType",
    "LinearResearchPattern",
    "DeepDivePattern",
    "ComparativeAnalysisPattern",
    "TrendAnalysisPattern",
    "ExecutionStrategy",
    "WorkflowMetrics",
    
    # Enhanced research flow (original)
    "EnhancedResearchFlow",
    "ResearchContext",
    "ResearchState",
    "CREWAI_FLOWS_AVAILABLE",
    
    # Flows 2.0 components (if available)
    "EnhancedResearchFlowV2",
    "FlowOrchestrator",
    "AdvancedFlowState",
    "FlowEvent",
    "FlowEventType",
    "FlowExecutionMode",
    "FlowPriority",
    "WorkflowComposition",
    "CREWAI_FLOWS_V2_AVAILABLE",
    
    # Integration components (if available)
    "FlowWorkflowAdapter",
    "EnhancedWorkflowManager",
    "FlowMigrationConfig",
    "FLOWS_V2_INTEGRATION_AVAILABLE",
    
    # Utility functions
    "get_workflow_manager",
    "get_research_flow"
]
