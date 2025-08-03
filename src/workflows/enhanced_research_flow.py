"""
Enhanced Research Flow (Original)

This module contains the original enhanced research flow implementation
that is imported by the main workflows module.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

# Re-export from the main __init__.py file to maintain compatibility
try:
    from crewai.flow.flow import Flow, listen, start, router
    from crewai import Agent, Task, Crew
    from pydantic import BaseModel

    CREWAI_FLOWS_AVAILABLE = True
except ImportError:
    CREWAI_FLOWS_AVAILABLE = False
    
    # Mock classes for compatibility
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class Flow:
        def __init__(self, state_class=None):
            if state_class:
                self.state = state_class()
            else:
                self.state = None


@dataclass
class ResearchContext:
    """Context for research operations"""
    mission_id: str
    query: str
    scope: str
    priority: int = 1
    max_sources: int = 50
    quality_threshold: float = 0.7
    time_limit: int = 3600


class ResearchState(BaseModel):
    """State management for research flow"""
    mission_id: str = ""
    flow_id: str = ""
    current_phase: str = "initialization"
    phases_completed: List[str] = []
    final_report: str = ""
    confidence_score: float = 0.0


class EnhancedResearchFlow(Flow):
    """Enhanced research flow implementation"""
    
    def __init__(self):
        if CREWAI_FLOWS_AVAILABLE:
            super().__init__(ResearchState)
        else:
            self.state = ResearchState()
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current flow state"""
        return {
            "mission_id": getattr(self.state, 'mission_id', ''),
            "current_phase": getattr(self.state, 'current_phase', 'initialization'),
            "phases_completed": getattr(self.state, 'phases_completed', []),
            "confidence_score": getattr(self.state, 'confidence_score', 0.0)
        }