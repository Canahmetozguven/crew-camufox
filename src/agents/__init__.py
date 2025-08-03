#!/usr/bin/env python3
"""
Enhanced Agents Module for CrewAI
Advanced agent management, coordination, and hierarchical systems
"""

# Core hierarchical management
from .hierarchical_manager import (
    HierarchicalAgentManager,
    AgentRole,
    AgentStatus,
    TaskPriority,
    AgentCapability,
    HierarchicalTask,
    AgentProfile
)

# Agent coordination
from .coordination import (
    AgentCoordinator,
    MessageType,
    CoordinationStrategy,
    AgentMessage,
    CoordinationContext
)

# Legacy agents (maintain backwards compatibility)
try:
    from .enhanced_deep_researcher import EnhancedDeepResearcherAgent
    from .enhanced_multi_agent_orchestrator import EnhancedMultiAgentOrchestrator
    LEGACY_AGENTS_AVAILABLE = True
except ImportError:
    LEGACY_AGENTS_AVAILABLE = False

__all__ = [
    # Hierarchical Management
    'HierarchicalAgentManager',
    'AgentRole',
    'AgentStatus', 
    'TaskPriority',
    'AgentCapability',
    'HierarchicalTask',
    'AgentProfile',
    
    # Coordination
    'AgentCoordinator',
    'MessageType',
    'CoordinationStrategy',
    'AgentMessage',
    'CoordinationContext',
    
    # Legacy
    'LEGACY_AGENTS_AVAILABLE'
]

# Add legacy agents to exports if available
if LEGACY_AGENTS_AVAILABLE:
    __all__.extend([
        'EnhancedDeepResearcherAgent',
        'EnhancedMultiAgentOrchestrator'
    ])