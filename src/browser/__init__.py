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

try:
    from .captcha_solver import AdvancedCaptchaSolver, CaptchaChallenge
    from .anti_detection import AntiDetectionBrowserManager
    from .search_integration import AdvancedSearchManager
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False

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

# Add advanced features if available
if ADVANCED_FEATURES_AVAILABLE:
    __all__.extend([
        "AdvancedCaptchaSolver",
        "CaptchaChallenge", 
        "AntiDetectionBrowserManager",
        "AdvancedSearchManager"
    ])