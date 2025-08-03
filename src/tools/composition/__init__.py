"""
CrewAI Tool Composition System
Provides advanced tool chaining, parallel execution, and result transformation capabilities
"""

from .pipeline import ToolPipeline, PipelineStage, ToolResult, PipelineBuilder
from .search_pipeline import EnhancedSearchPipeline
from .transformers import (
    TransformationManager,
    TransformationConfig,
    TransformationResult,
    OutputFormat,
    TransformationMode,
    transform_to_json,
    transform_to_csv,
    transform_to_markdown,
    get_transformation_capabilities
)
from .integration import (
    ComposedToolManager,
    enhanced_search,
    batch_search,
    enhanced_search_with_transform,
    batch_search_with_transform,
    get_pipeline_capabilities
)

__all__ = [
    # Core pipeline components
    'ToolPipeline',
    'PipelineStage', 
    'ToolResult',
    'PipelineBuilder',
    
    # Enhanced search pipeline
    'EnhancedSearchPipeline',
    
    # Transformation system
    'TransformationManager',
    'TransformationConfig',
    'TransformationResult',
    'OutputFormat',
    'TransformationMode',
    'transform_to_json',
    'transform_to_csv',
    'transform_to_markdown',
    'get_transformation_capabilities',
    
    # Integration layer
    'ComposedToolManager',
    'enhanced_search',
    'batch_search',
    'enhanced_search_with_transform',
    'batch_search_with_transform',
    'get_pipeline_capabilities',
]

__version__ = "1.0.0"