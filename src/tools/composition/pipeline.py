"""
Core Tool Pipeline Framework
Provides composable tool execution with chaining, parallel processing, and error handling
"""

from typing import List, Dict, Any, Union, Optional, Callable, Awaitable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import asyncio
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline execution stages"""
    PREPROCESSING = "preprocessing"
    EXECUTION = "execution"
    POSTPROCESSING = "postprocessing"
    VALIDATION = "validation"


@dataclass
class ToolResult:
    """Standardized tool result format for pipeline communication"""
    data: Any
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    stage: Optional[PipelineStage] = None
    tool_name: Optional[str] = None
    execution_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "data": self.data,
            "metadata": self.metadata,
            "success": self.success,
            "error": self.error,
            "stage": self.stage.value if self.stage else None,
            "tool_name": self.tool_name,
            "execution_time": self.execution_time
        }


class ToolPipeline:
    """
    Composable tool pipeline with chaining, parallel execution, and robust error handling
    
    Features:
    - Sequential and parallel tool execution
    - Stage-based organization
    - Result transformation
    - Error handling with fallbacks
    - Performance monitoring
    """
    
    def __init__(self, name: str):
        self.name = name
        self.stages: List[PipelineStage] = []
        self.tools: Dict[PipelineStage, List[Callable]] = {}
        self.transformers: Dict[str, Callable] = {}
        self.error_handlers: Dict[str, Callable] = {}
        self.parallel_groups: List[List[str]] = []
        self.fallback_strategies: Dict[str, Callable] = {}
        self.stage_timeouts: Dict[PipelineStage, float] = {}
        
        # Performance tracking
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "stage_performance": {}
        }
    
    def add_stage(self, stage: PipelineStage, tools: List[Callable], timeout: Optional[float] = None):
        """Add tools to a pipeline stage with optional timeout"""
        if stage not in self.stages:
            self.stages.append(stage)
        self.tools[stage] = tools
        if timeout:
            self.stage_timeouts[stage] = timeout
        
        logger.info(f"Added stage {stage.value} with {len(tools)} tools to pipeline {self.name}")
    
    def add_transformer(self, name: str, transformer: Callable):
        """Add result transformer for data format conversion"""
        self.transformers[name] = transformer
        logger.debug(f"Added transformer {name} to pipeline {self.name}")
    
    def add_error_handler(self, stage_or_tool: str, handler: Callable):
        """Add error handler for specific stage or tool"""
        self.error_handlers[stage_or_tool] = handler
    
    def add_parallel_group(self, tool_names: List[str]):
        """Define tools that can run in parallel"""
        self.parallel_groups.append(tool_names)
        logger.debug(f"Added parallel group {tool_names} to pipeline {self.name}")
    
    def add_fallback_strategy(self, tool_name: str, fallback: Callable):
        """Add fallback strategy for tool failure"""
        self.fallback_strategies[tool_name] = fallback
    
    async def execute(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Execute the entire pipeline with performance tracking"""
        start_time = datetime.now()
        self.execution_stats["total_executions"] += 1
        
        pipeline_metadata = {
            "pipeline_name": self.name,
            "started_at": start_time.isoformat(),
            "stages_executed": [],
            "total_tools": sum(len(tools) for tools in self.tools.values()),
            "context": context or {}
        }
        
        current_data = input_data
        
        try:
            for stage in self.stages:
                logger.info(f"Executing stage {stage.value} in pipeline {self.name}")
                
                # Apply stage timeout if configured
                stage_timeout = self.stage_timeouts.get(stage)
                
                if stage_timeout:
                    stage_result = await asyncio.wait_for(
                        self._execute_stage(stage, current_data, context),
                        timeout=stage_timeout
                    )
                else:
                    stage_result = await self._execute_stage(stage, current_data, context)
                
                if not stage_result.success:
                    # Try error handler if available
                    error_handler = self.error_handlers.get(stage.value)
                    if error_handler:
                        try:
                            recovery_result = await error_handler(stage_result, current_data, context)
                            if recovery_result.success:
                                current_data = recovery_result.data
                                pipeline_metadata["stages_executed"].append(f"{stage.value} (recovered)")
                                continue
                        except Exception as recovery_error:
                            logger.error(f"Error handler failed for stage {stage.value}: {recovery_error}")
                    
                    # Stage failed and no recovery possible
                    self.execution_stats["failed_executions"] += 1
                    pipeline_metadata["failed_at"] = stage.value
                    pipeline_metadata["completed_at"] = datetime.now().isoformat()
                    
                    return ToolResult(
                        data=None,
                        metadata=pipeline_metadata,
                        success=False,
                        error=f"Stage {stage.value} failed: {stage_result.error}"
                    )
                
                current_data = stage_result.data
                pipeline_metadata["stages_executed"].append(stage.value)
                
                # Update stage performance stats
                stage_key = stage.value
                if stage_key not in self.execution_stats["stage_performance"]:
                    self.execution_stats["stage_performance"][stage_key] = {
                        "executions": 0,
                        "successes": 0,
                        "average_time": 0.0
                    }
                
                stage_stats = self.execution_stats["stage_performance"][stage_key]
                stage_stats["executions"] += 1
                stage_stats["successes"] += 1
                
                if stage_result.execution_time:
                    stage_stats["average_time"] = (
                        (stage_stats["average_time"] * (stage_stats["executions"] - 1) + stage_result.execution_time)
                        / stage_stats["executions"]
                    )
            
            # Pipeline completed successfully
            execution_time = (datetime.now() - start_time).total_seconds()
            self.execution_stats["successful_executions"] += 1
            
            # Update average execution time
            total_executions = self.execution_stats["total_executions"]
            current_avg = self.execution_stats["average_execution_time"]
            self.execution_stats["average_execution_time"] = (
                (current_avg * (total_executions - 1) + execution_time) / total_executions
            )
            
            pipeline_metadata["completed_at"] = datetime.now().isoformat()
            pipeline_metadata["execution_time"] = execution_time
            
            return ToolResult(
                data=current_data,
                metadata=pipeline_metadata,
                success=True,
                execution_time=execution_time
            )
            
        except asyncio.TimeoutError:
            self.execution_stats["failed_executions"] += 1
            pipeline_metadata["completed_at"] = datetime.now().isoformat()
            
            return ToolResult(
                data=None,
                metadata=pipeline_metadata,
                success=False,
                error=f"Pipeline execution timed out"
            )
            
        except Exception as e:
            self.execution_stats["failed_executions"] += 1
            pipeline_metadata["completed_at"] = datetime.now().isoformat()
            
            logger.error(f"Pipeline {self.name} execution failed: {e}")
            return ToolResult(
                data=None,
                metadata=pipeline_metadata,
                success=False,
                error=f"Pipeline execution failed: {str(e)}"
            )
    
    async def _execute_stage(self, stage: PipelineStage, data: Any, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Execute a single pipeline stage with parallel and sequential tools"""
        stage_start_time = datetime.now()
        stage_tools = self.tools.get(stage, [])
        
        if not stage_tools:
            return ToolResult(
                data=data, 
                metadata={}, 
                success=True, 
                stage=stage,
                execution_time=0.0
            )
        
        # Separate parallel and sequential tools
        parallel_tools = []
        sequential_tools = []
        
        for tool in stage_tools:
            tool_name = getattr(tool, '__name__', str(tool))
            is_parallel = any(tool_name in group for group in self.parallel_groups)
            
            if is_parallel:
                parallel_tools.append(tool)
            else:
                sequential_tools.append(tool)
        
        # Execute parallel tools first
        if parallel_tools:
            logger.debug(f"Executing {len(parallel_tools)} parallel tools in stage {stage.value}")
            parallel_results = await self._execute_parallel_tools(parallel_tools, data, context)
            
            # Check for failures in parallel execution
            failed_tools = [r.tool_name for r in parallel_results if not r.success]
            if failed_tools:
                # Try fallback strategies for failed tools
                for failed_tool in failed_tools:
                    if failed_tool:  # Ensure failed_tool is not None
                        fallback = self.fallback_strategies.get(failed_tool)
                        if fallback:
                            try:
                                fallback_result = await fallback(data, context)
                                if fallback_result.success:
                                    # Replace failed result with fallback result
                                    for i, result in enumerate(parallel_results):
                                        if result.tool_name == failed_tool:
                                            parallel_results[i] = fallback_result
                                            break
                            except Exception as fallback_error:
                                logger.error(f"Fallback failed for {failed_tool}: {fallback_error}")
                
                # Check if critical failures remain
                still_failed = [r.tool_name for r in parallel_results if not r.success]
                if still_failed:
                    execution_time = (datetime.now() - stage_start_time).total_seconds()
                    return ToolResult(
                        data=None,
                        metadata={"failed_tools": still_failed},
                        success=False,
                        error=f"Parallel tools failed: {still_failed}",
                        stage=stage,
                        execution_time=execution_time
                    )
            
            # Combine parallel results
            data = self._combine_parallel_results(parallel_results, data)
        
        # Execute sequential tools
        for tool in sequential_tools:
            tool_name = getattr(tool, '__name__', str(tool))
            logger.debug(f"Executing sequential tool {tool_name} in stage {stage.value}")
            
            try:
                tool_start_time = datetime.now()
                
                # Pass context to tool if it accepts it
                try:
                    result = await tool(data, context) if context else await tool(data)
                except TypeError:
                    # Tool doesn't accept context parameter
                    result = await tool(data)
                
                tool_execution_time = (datetime.now() - tool_start_time).total_seconds()
                
                if isinstance(result, ToolResult):
                    if not result.success:
                        # Try fallback strategy
                        fallback = self.fallback_strategies.get(tool_name)
                        if fallback:
                            fallback_result = await fallback(data, context)
                            if fallback_result.success:
                                data = fallback_result.data
                                continue
                        
                        execution_time = (datetime.now() - stage_start_time).total_seconds()
                        return ToolResult(
                            data=None,
                            metadata={"tool_name": tool_name},
                            success=False,
                            error=result.error,
                            stage=stage,
                            execution_time=execution_time
                        )
                    data = result.data
                else:
                    data = result
                    
            except Exception as e:
                # Try fallback strategy
                fallback = self.fallback_strategies.get(tool_name)
                if fallback:
                    try:
                        fallback_result = await fallback(data, context)
                        if fallback_result.success:
                            data = fallback_result.data
                            continue
                    except Exception as fallback_error:
                        logger.error(f"Fallback failed for {tool_name}: {fallback_error}")
                
                execution_time = (datetime.now() - stage_start_time).total_seconds()
                return ToolResult(
                    data=None,
                    metadata={"tool_name": tool_name},
                    success=False,
                    error=f"Tool {tool_name} failed: {str(e)}",
                    stage=stage,
                    execution_time=execution_time
                )
        
        execution_time = (datetime.now() - stage_start_time).total_seconds()
        return ToolResult(
            data=data, 
            metadata={}, 
            success=True, 
            stage=stage,
            execution_time=execution_time
        )
    
    async def _execute_parallel_tools(self, tools: List[Callable], data: Any, context: Optional[Dict[str, Any]] = None) -> List[ToolResult]:
        """Execute tools in parallel with proper error handling"""
        tasks = []
        for tool in tools:
            task = asyncio.create_task(self._execute_single_tool(tool, data, context))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to ToolResult objects
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                tool_name = getattr(tools[i], '__name__', str(tools[i]))
                processed_results.append(ToolResult(
                    data=None,
                    metadata={},
                    success=False,
                    error=str(result),
                    tool_name=tool_name
                ))
            elif isinstance(result, ToolResult):
                processed_results.append(result)
            else:
                # Shouldn't happen, but handle gracefully
                tool_name = getattr(tools[i], '__name__', str(tools[i]))
                processed_results.append(ToolResult(
                    data=result,
                    metadata={},
                    success=True,
                    tool_name=tool_name
                ))
        
        return processed_results
    
    async def _execute_single_tool(self, tool: Callable, data: Any, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Execute a single tool with comprehensive error handling"""
        tool_name = getattr(tool, '__name__', str(tool))
        start_time = datetime.now()
        
        try:
            # Pass context to tool if it accepts it
            try:
                result = await tool(data, context) if context else await tool(data)
            except TypeError:
                # Tool doesn't accept context parameter
                result = await tool(data)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if isinstance(result, ToolResult):
                result.tool_name = tool_name
                result.execution_time = execution_time
                return result
            else:
                return ToolResult(
                    data=result,
                    metadata={"execution_time": execution_time},
                    success=True,
                    tool_name=tool_name,
                    execution_time=execution_time
                )
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Tool {tool_name} failed: {e}")
            
            return ToolResult(
                data=None,
                metadata={"execution_time": execution_time},
                success=False,
                error=str(e),
                tool_name=tool_name,
                execution_time=execution_time
            )
    
    def _combine_parallel_results(self, results: List[ToolResult], original_data: Any) -> Any:
        """
        Combine results from parallel tool execution
        Override this method in subclasses for custom combination logic
        """
        combined_data = {
            "original": original_data,
            "parallel_results": {},
            "metadata": {
                "total_parallel_tools": len(results),
                "successful_tools": len([r for r in results if r.success]),
                "failed_tools": len([r for r in results if not r.success])
            }
        }
        
        for result in results:
            if result.success and result.tool_name:
                combined_data["parallel_results"][result.tool_name] = result.data
        
        return combined_data
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics for the pipeline"""
        success_rate = (
            self.execution_stats["successful_executions"] / max(self.execution_stats["total_executions"], 1)
        ) * 100
        
        return {
            "pipeline_name": self.name,
            "total_executions": self.execution_stats["total_executions"],
            "successful_executions": self.execution_stats["successful_executions"],
            "failed_executions": self.execution_stats["failed_executions"],
            "success_rate": round(success_rate, 2),
            "average_execution_time": round(self.execution_stats["average_execution_time"], 3),
            "stage_performance": self.execution_stats["stage_performance"],
            "configured_stages": [stage.value for stage in self.stages],
            "parallel_groups": self.parallel_groups,
            "fallback_strategies": list(self.fallback_strategies.keys())
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.execution_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "stage_performance": {}
        }
        logger.info(f"Reset performance statistics for pipeline {self.name}")


class PipelineBuilder:
    """Builder pattern for creating complex pipelines"""
    
    def __init__(self, name: str):
        self.pipeline = ToolPipeline(name)
    
    def add_preprocessing(self, tools: List[Callable], timeout: Optional[float] = None):
        """Add preprocessing stage"""
        self.pipeline.add_stage(PipelineStage.PREPROCESSING, tools, timeout)
        return self
    
    def add_execution(self, tools: List[Callable], timeout: Optional[float] = None):
        """Add execution stage"""
        self.pipeline.add_stage(PipelineStage.EXECUTION, tools, timeout)
        return self
    
    def add_postprocessing(self, tools: List[Callable], timeout: Optional[float] = None):
        """Add postprocessing stage"""
        self.pipeline.add_stage(PipelineStage.POSTPROCESSING, tools, timeout)
        return self
    
    def add_validation(self, tools: List[Callable], timeout: Optional[float] = None):
        """Add validation stage"""
        self.pipeline.add_stage(PipelineStage.VALIDATION, tools, timeout)
        return self
    
    def with_parallel_group(self, tool_names: List[str]):
        """Add parallel execution group"""
        self.pipeline.add_parallel_group(tool_names)
        return self
    
    def with_fallback(self, tool_name: str, fallback: Callable):
        """Add fallback strategy"""
        self.pipeline.add_fallback_strategy(tool_name, fallback)
        return self
    
    def with_error_handler(self, stage_or_tool: str, handler: Callable):
        """Add error handler"""
        self.pipeline.add_error_handler(stage_or_tool, handler)
        return self
    
    def build(self) -> ToolPipeline:
        """Build and return the configured pipeline"""
        return self.pipeline