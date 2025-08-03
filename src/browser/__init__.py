"""
Enhanced Browser Management for Crew-Camufox
Advanced Camoufox integration with stealth capabilities, session management, and performance optimization
"""

from .camoufox_enhanced import (
    EnhancedCamoufoxManager,
    StealthProfile,
    SessionConfig,
    PerformanceConfig
)

from .camoufox_performance import (
    CamoufoxPerformanceOptimizer,
    PerformanceLevel,
    OptimizationConfig,
    PerformanceMetrics,
    ResourceBlockingRule,
    PERFORMANCE_PRESETS
)

__all__ = [
    "EnhancedCamoufoxManager",
    "StealthProfile", 
    "SessionConfig",
    "PerformanceConfig",
    "CamoufoxPerformanceOptimizer",
    "PerformanceLevel",
    "OptimizationConfig",
    "PerformanceMetrics",
    "ResourceBlockingRule",
    "PERFORMANCE_PRESETS"
]