"""
Integration Layer for Tool Composition
Provides integration with existing CrewAI agents and seamless tool composition management
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import asyncio

from .pipeline import ToolResult
from .search_pipeline import EnhancedSearchPipeline
from .transformers import TransformationManager, TransformationConfig, OutputFormat, TransformationMode

logger = logging.getLogger(__name__)


class ComposedToolManager:
    """
    Manager for composed tools integration with existing CrewAI agents
    
    Features:
    - Seamless integration with DeepResearcherAgent
    - Pipeline performance monitoring
    - Context management across tool chains
    - Fallback strategies for tool failures
    - Result caching and optimization
    """
    
    def __init__(self, headless: bool = True):
        self.search_pipeline = EnhancedSearchPipeline(headless=headless)
        self.transformation_manager = TransformationManager()
        self.active_pipelines: Dict[str, Any] = {}
        self.pipeline_registry: Dict[str, Any] = {}
        self.performance_stats: Dict[str, Dict[str, Any]] = {}
        self.context_store: Dict[str, Any] = {}
        
        # Register available pipelines
        self._register_pipelines()
        
        logger.info("ComposedToolManager initialized with enhanced search pipeline and transformation system")
    
    def _register_pipelines(self):
        """Register available tool pipelines"""
        self.pipeline_registry = {
            "enhanced_search": {
                "pipeline": self.search_pipeline,
                "description": "Multi-engine search with intelligent filtering and ranking",
                "features": ["parallel_execution", "deduplication", "quality_scoring", "relevance_ranking"],
                "input_format": {"query": "str", "max_results": "int", "context": "dict"},
                "output_format": {"results": "list", "metadata": "dict"}
            }
        }
    
    async def enhanced_search(self, query: str, max_results: int = 10, context: Optional[Dict[str, Any]] = None, headless: bool = True) -> ToolResult:
        """
        Enhanced search using tool composition pipeline
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            context: Optional context for search customization
            
        Returns:
            ToolResult with search results and comprehensive metadata
        """
        if not query or not isinstance(query, str):
            return ToolResult(
                data=[],
                metadata={"error": "Invalid query provided"},
                success=False,
                error="Query must be a non-empty string"
            )
        
        search_id = f"search_{int(datetime.now().timestamp())}_{hash(query) % 10000}"
        
        # Prepare input data for pipeline
        query_data = {
            "query": query.strip(),
            "max_results": max(1, min(max_results, 50)),  # Limit between 1-50
            "search_id": search_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Merge with provided context
        pipeline_context = context.copy() if context else {}
        pipeline_context.update({
            "search_id": search_id,
            "manager": "ComposedToolManager",
            "version": "1.0.0"
        })
        
        try:
            logger.info(f"Starting enhanced search: '{query}' (ID: {search_id})")
            
            # Track active pipeline
            self.active_pipelines[search_id] = {
                "pipeline": "enhanced_search",
                "query": query,
                "started_at": datetime.now(),
                "status": "running"
            }
            
            # Execute the search pipeline
            # Update headless mode if provided
            if hasattr(self.search_pipeline, "headless"):
                self.search_pipeline.headless = headless
            result = await self.search_pipeline.execute(query_data, pipeline_context)
            
            # Update pipeline tracking
            self.active_pipelines[search_id]["status"] = "completed" if result.success else "failed"
            self.active_pipelines[search_id]["completed_at"] = datetime.now()
            self.active_pipelines[search_id]["execution_time"] = result.execution_time
            
            if result.success:
                search_results = result.data.get("results", [])
                
                # Update performance statistics
                self._update_performance_stats("enhanced_search", result)
                
                # Store context for potential reuse
                self._store_context(search_id, query_data, pipeline_context, result)
                
                logger.info(f"Enhanced search completed: {len(search_results)} results (ID: {search_id})")
                
                return ToolResult(
                    data=search_results,
                    metadata={
                        "search_id": search_id,
                        "pipeline": "enhanced_search",
                        "query": query,
                        "total_results": len(search_results),
                        "pipeline_metadata": result.metadata,
                        "processing_summary": result.data.get("metadata", {}),
                        "performance": {
                            "execution_time": result.execution_time,
                            "pipeline_stages": len(self.search_pipeline.stages),
                            "engines_used": result.data.get("metadata", {}).get("search_summary", {}).get("engines_used", [])
                        }
                    },
                    success=True,
                    execution_time=result.execution_time
                )
            else:
                logger.error(f"Enhanced search failed: {result.error} (ID: {search_id})")
                
                return ToolResult(
                    data=[],
                    metadata={
                        "search_id": search_id,
                        "pipeline": "enhanced_search",
                        "query": query,
                        "error_details": result.metadata,
                        "pipeline_error": result.error
                    },
                    success=False,
                    error=f"Enhanced search failed: {result.error}",
                    execution_time=result.execution_time
                )
                
        except Exception as e:
            logger.error(f"Enhanced search exception: {e} (ID: {search_id})")
            
            # Update pipeline tracking
            if search_id in self.active_pipelines:
                self.active_pipelines[search_id]["status"] = "error"
                self.active_pipelines[search_id]["completed_at"] = datetime.now()
                self.active_pipelines[search_id]["error"] = str(e)
            
            return ToolResult(
                data=[],
                metadata={
                    "search_id": search_id,
                    "pipeline": "enhanced_search",
                    "query": query,
                    "exception": str(e)
                },
                success=False,
                error=f"Enhanced search exception: {e}"
            )
        
        finally:
            # Clean up active pipeline tracking after delay
            asyncio.create_task(self._cleanup_pipeline_tracking(search_id, delay=300))  # 5 minute delay
    
    async def batch_search(self, queries: List[str], max_results_per_query: int = 5, context: Optional[Dict[str, Any]] = None) -> List[ToolResult]:
        """
        Execute multiple searches in parallel using tool composition
        
        Args:
            queries: List of search query strings
            max_results_per_query: Maximum results per individual query
            context: Optional context for search customization
            
        Returns:
            List of ToolResult objects, one per query
        """
        if not queries or not isinstance(queries, list):
            return [ToolResult(
                data=[],
                metadata={"error": "Invalid queries provided"},
                success=False,
                error="Queries must be a non-empty list of strings"
            )]
        
        # Filter valid queries
        valid_queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        
        if not valid_queries:
            return [ToolResult(
                data=[],
                metadata={"error": "No valid queries provided"},
                success=False,
                error="At least one valid query string is required"
            )]
        
        batch_id = f"batch_{int(datetime.now().timestamp())}"
        logger.info(f"Starting batch search: {len(valid_queries)} queries (Batch ID: {batch_id})")
        
        # Prepare batch context
        batch_context = context.copy() if context else {}
        batch_context.update({
            "batch_id": batch_id,
            "batch_size": len(valid_queries),
            "batch_mode": True
        })
        
        # Create search tasks
        search_tasks = []
        for i, query in enumerate(valid_queries):
            query_context = batch_context.copy()
            query_context["batch_index"] = i
            query_context["batch_query"] = query
            
            task = asyncio.create_task(
                self.enhanced_search(query, max_results_per_query, query_context),
                name=f"batch_search_{batch_id}_{i}"
            )
            search_tasks.append(task)
        
        # Execute searches in parallel
        try:
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(ToolResult(
                        data=[],
                        metadata={
                            "batch_id": batch_id,
                            "batch_index": i,
                            "query": valid_queries[i],
                            "exception": str(result)
                        },
                        success=False,
                        error=f"Batch search task failed: {result}"
                    ))
                elif isinstance(result, ToolResult):
                    # Add batch metadata
                    result.metadata.update({
                        "batch_id": batch_id,
                        "batch_index": i,
                        "batch_size": len(valid_queries)
                    })
                    processed_results.append(result)
                else:
                    # Shouldn't happen, but handle gracefully
                    processed_results.append(ToolResult(
                        data=[],
                        metadata={
                            "batch_id": batch_id,
                            "batch_index": i,
                            "query": valid_queries[i],
                            "unexpected_result": str(result)
                        },
                        success=False,
                        error="Unexpected result format"
                    ))
            
            successful_searches = sum(1 for r in processed_results if r.success)
            logger.info(f"Batch search completed: {successful_searches}/{len(valid_queries)} successful (Batch ID: {batch_id})")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Batch search failed: {e} (Batch ID: {batch_id})")
            
            # Return error results for all queries
            error_results = []
            for i, query in enumerate(valid_queries):
                error_results.append(ToolResult(
                    data=[],
                    metadata={
                        "batch_id": batch_id,
                        "batch_index": i,
                        "query": query,
                        "batch_error": str(e)
                    },
                    success=False,
                    error=f"Batch search failed: {e}"
                ))
            
            return error_results
    
    async def enhanced_search_with_transform(
        self, 
        query: str, 
        max_results: int = 10, 
        output_format: OutputFormat = OutputFormat.JSON,
        transformation_mode: TransformationMode = TransformationMode.PRESERVE,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enhanced search with automatic result transformation
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            output_format: Desired output format for results
            transformation_mode: Transformation processing mode
            context: Optional context for search customization
            
        Returns:
            Dictionary containing both search results and transformed output
        """
        # Perform enhanced search
        search_result = await self.enhanced_search(query, max_results, context)
        
        if not search_result.success:
            return {
                "search_result": search_result,
                "transformed_result": None,
                "error": "Search failed, transformation skipped"
            }
        
        # Configure transformation
        transform_config = TransformationConfig(
            output_format=output_format,
            mode=transformation_mode,
            include_metadata=True,
            include_performance_stats=True
        )
        
        # Prepare data for transformation
        transform_data = {
            "query": query,
            "sources": search_result.data,
            "metadata": search_result.metadata,
            "performance_stats": {
                "search_execution_time": search_result.execution_time,
                "total_results": len(search_result.data),
                "search_pipeline_performance": self.search_pipeline.get_performance_stats()
            }
        }
        
        # Apply transformation
        try:
            transformation_result = await self.transformation_manager.transform(
                transform_data, 
                transform_config, 
                context
            )
            
            return {
                "search_result": search_result,
                "transformed_result": transformation_result,
                "combined_metadata": {
                    "search_metadata": search_result.metadata,
                    "transformation_metadata": transformation_result.metadata,
                    "total_processing_time": (search_result.execution_time or 0) + transformation_result.transformation_time
                }
            }
            
        except Exception as e:
            logger.error(f"Transformation failed for query '{query}': {e}")
            return {
                "search_result": search_result,
                "transformed_result": None,
                "error": f"Transformation failed: {e}"
            }
    
    async def batch_search_with_transform(
        self,
        queries: List[str],
        max_results_per_query: int = 5,
        output_format: OutputFormat = OutputFormat.JSON,
        transformation_mode: TransformationMode = TransformationMode.OPTIMIZE,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Batch search with transformation for multiple queries
        
        Args:
            queries: List of search query strings
            max_results_per_query: Maximum results per individual query
            output_format: Desired output format for results
            transformation_mode: Transformation processing mode
            context: Optional context for search customization
            
        Returns:
            List of dictionaries containing search and transformation results
        """
        if not queries or not isinstance(queries, list):
            return [{
                "search_result": None,
                "transformed_result": None,
                "error": "Invalid queries provided"
            }]
        
        # Create tasks for parallel processing
        transform_tasks = []
        for query in queries:
            task = asyncio.create_task(
                self.enhanced_search_with_transform(
                    query, 
                    max_results_per_query, 
                    output_format, 
                    transformation_mode, 
                    context
                )
            )
            transform_tasks.append(task)
        
        # Execute all transformations in parallel
        try:
            results = await asyncio.gather(*transform_tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "search_result": None,
                        "transformed_result": None,
                        "error": f"Batch transform task failed: {result}",
                        "query": queries[i] if i < len(queries) else "Unknown"
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Batch search with transform failed: {e}")
            return [{
                "search_result": None,
                "transformed_result": None,
                "error": f"Batch processing failed: {e}",
                "query": query
            } for query in queries]

            
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about pipeline performance"""
        return {
            "available_pipelines": list(self.pipeline_registry.keys()),
            "active_pipelines": len(self.active_pipelines),
            "pipeline_details": {
                name: {
                    "description": info["description"],
                    "features": info["features"],
                    "performance": self.performance_stats.get(name, {})
                }
                for name, info in self.pipeline_registry.items()
            },
            "search_pipeline_performance": self.search_pipeline.get_performance_stats(),
            "context_store_size": len(self.context_store),
            "recent_activity": self._get_recent_activity()
        }
    
    def get_active_pipelines(self) -> Dict[str, Any]:
        """Get information about currently active pipelines"""
        return {
            "count": len(self.active_pipelines),
            "pipelines": self.active_pipelines.copy()
        }
    
    def get_context(self, search_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored context for a search ID"""
        return self.context_store.get(search_id)
    
    def clear_performance_stats(self):
        """Clear all performance statistics"""
        self.performance_stats.clear()
        self.search_pipeline.reset_stats()
        logger.info("Performance statistics cleared")
    
    def _update_performance_stats(self, pipeline_name: str, result: ToolResult):
        """Update performance statistics for a pipeline"""
        if pipeline_name not in self.performance_stats:
            self.performance_stats[pipeline_name] = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "total_execution_time": 0.0,
                "average_execution_time": 0.0,
                "last_execution": None
            }
        
        stats = self.performance_stats[pipeline_name]
        stats["total_executions"] += 1
        stats["last_execution"] = datetime.now().isoformat()
        
        if result.success:
            stats["successful_executions"] += 1
        else:
            stats["failed_executions"] += 1
        
        if result.execution_time:
            stats["total_execution_time"] += result.execution_time
            stats["average_execution_time"] = stats["total_execution_time"] / stats["total_executions"]
    
    def _store_context(self, search_id: str, query_data: Dict[str, Any], context: Dict[str, Any], result: ToolResult):
        """Store context for potential reuse and analysis"""
        self.context_store[search_id] = {
            "query_data": query_data,
            "context": context,
            "result_metadata": result.metadata,
            "success": result.success,
            "execution_time": result.execution_time,
            "stored_at": datetime.now().isoformat()
        }
        
        # Limit context store size (keep last 100 entries)
        if len(self.context_store) > 100:
            # Remove oldest entries
            oldest_keys = sorted(
                self.context_store.keys(),
                key=lambda k: self.context_store[k]["stored_at"]
            )[:len(self.context_store) - 100]
            
            for key in oldest_keys:
                del self.context_store[key]
    
    def _get_recent_activity(self) -> List[Dict[str, Any]]:
        """Get recent pipeline activity summary"""
        recent_activity = []
        
        # Get recent completed pipelines
        completed_pipelines = [
            (search_id, info) for search_id, info in self.active_pipelines.items()
            if info.get("status") in ["completed", "failed", "error"]
        ]
        
        # Sort by completion time
        completed_pipelines.sort(
            key=lambda x: x[1].get("completed_at", datetime.min),
            reverse=True
        )
        
        # Return last 10 activities
        for search_id, info in completed_pipelines[:10]:
            activity = {
                "search_id": search_id,
                "pipeline": info.get("pipeline"),
                "query": info.get("query", "")[:50],  # Truncate long queries
                "status": info.get("status"),
                "execution_time": info.get("execution_time"),
                "completed_at": info.get("completed_at", "").isoformat() if isinstance(info.get("completed_at"), datetime) else info.get("completed_at")
            }
            recent_activity.append(activity)
        
        return recent_activity
    
    async def _cleanup_pipeline_tracking(self, search_id: str, delay: int = 300):
        """Clean up pipeline tracking after delay"""
        await asyncio.sleep(delay)
        
        if search_id in self.active_pipelines:
            pipeline_info = self.active_pipelines[search_id]
            
            # Only remove if completed and not recently accessed
            if pipeline_info.get("status") in ["completed", "failed", "error"]:
                completed_at = pipeline_info.get("completed_at")
                if completed_at and isinstance(completed_at, datetime):
                    time_since_completion = (datetime.now() - completed_at).total_seconds()
                    if time_since_completion >= delay:
                        del self.active_pipelines[search_id]
                        logger.debug(f"Cleaned up pipeline tracking for search ID: {search_id}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all pipelines and components"""
        health_status = {
            "overall_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Check search pipeline health
        try:
            search_stats = self.search_pipeline.get_performance_stats()
            search_health = {
                "status": "healthy",
                "total_executions": search_stats["total_executions"],
                "success_rate": search_stats["success_rate"],
                "average_execution_time": search_stats["average_execution_time"],
                "configured_stages": len(search_stats["configured_stages"])
            }
            
            # Determine health based on success rate
            if search_stats["success_rate"] < 50:
                search_health["status"] = "degraded"
                health_status["overall_status"] = "degraded"
            elif search_stats["success_rate"] < 20:
                search_health["status"] = "unhealthy"
                health_status["overall_status"] = "unhealthy"
            
        except Exception as e:
            search_health = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "unhealthy"
        
        health_status["components"]["search_pipeline"] = search_health
        
        # Check manager health
        manager_health = {
            "status": "healthy",
            "active_pipelines": len(self.active_pipelines),
            "context_store_size": len(self.context_store),
            "registered_pipelines": len(self.pipeline_registry)
        }
        
        # Check for resource issues
        if len(self.active_pipelines) > 20:
            manager_health["status"] = "degraded"
            manager_health["warning"] = "High number of active pipelines"
            health_status["overall_status"] = "degraded"
        
        if len(self.context_store) > 150:
            manager_health["status"] = "degraded"
            manager_health["warning"] = "Large context store size"
            health_status["overall_status"] = "degraded"
        
        health_status["components"]["tool_manager"] = manager_health
        
        return health_status


# Convenience functions for integration with existing code
async def enhanced_search(query: str, max_results: int = 10, context: Optional[Dict[str, Any]] = None, headless: bool = True) -> ToolResult:
    """
    Convenience function for enhanced search
    Creates a temporary ComposedToolManager instance for one-off searches
    """
    manager = ComposedToolManager(headless=headless)
    return await manager.enhanced_search(query, max_results, context, headless=headless)


async def batch_search(queries: List[str], max_results_per_query: int = 5, context: Optional[Dict[str, Any]] = None) -> List[ToolResult]:
    """
    Convenience function for batch search
    Creates a temporary ComposedToolManager instance for batch searches
    """
    manager = ComposedToolManager()
    return await manager.batch_search(queries, max_results_per_query, context)


# New convenience functions for transformation capabilities
async def enhanced_search_with_transform(
    query: str, 
    max_results: int = 10, 
    output_format: OutputFormat = OutputFormat.JSON,
    transformation_mode: TransformationMode = TransformationMode.PRESERVE,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function for enhanced search with transformation
    Creates a temporary ComposedToolManager instance for one-off searches with transformation
    """
    manager = ComposedToolManager()
    return await manager.enhanced_search_with_transform(
        query, max_results, output_format, transformation_mode, context
    )


async def batch_search_with_transform(
    queries: List[str],
    max_results_per_query: int = 5,
    output_format: OutputFormat = OutputFormat.JSON,
    transformation_mode: TransformationMode = TransformationMode.OPTIMIZE,
    context: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function for batch search with transformation
    Creates a temporary ComposedToolManager instance for batch searches with transformation
    """
    manager = ComposedToolManager()
    return await manager.batch_search_with_transform(
        queries, max_results_per_query, output_format, transformation_mode, context
    )


def get_pipeline_capabilities() -> Dict[str, Any]:
    """
    Get information about available pipeline capabilities including transformation features
    """
    manager = ComposedToolManager()
    return {
        "version": "1.0.0",
        "pipelines": manager.pipeline_registry,
        "features": [
            "parallel_search_execution",
            "intelligent_deduplication",
            "relevance_scoring",
            "quality_validation",
            "error_recovery",
            "performance_monitoring",
            "batch_processing",
            "context_management",
            "result_transformation",
            "multi_format_output",
            "data_optimization",
            "content_summarization",
            "metadata_enrichment"
        ],
        "transformation": {
            "supported_formats": [format.value for format in OutputFormat],
            "transformation_modes": [mode.value for mode in TransformationMode],
            "features": [
                "json_output",
                "csv_export",
                "markdown_reports",
                "xml_structured_data",
                "html_presentation",
                "data_filtering",
                "content_optimization",
                "batch_transformation"
            ]
        },
        "supported_engines": ["google", "bing", "duckduckgo", "scholar"],
        "max_results_limit": 50,
        "max_batch_size": 20
    }