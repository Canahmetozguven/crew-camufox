#!/usr/bin/env python3
"""
Result Transformation System for Tool Composition
Provides advanced data format conversion and processing capabilities
"""

import json
import csv
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import re
import hashlib
import base64


class TransformationError(Exception):
    """Exception raised during result transformation"""
    pass


class OutputFormat(Enum):
    """Supported output formats for transformation"""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    YAML = "yaml"
    RSS = "rss"
    JSONL = "jsonl"  # JSON Lines format
    EXCEL = "excel"


class TransformationMode(Enum):
    """Transformation processing modes"""
    PRESERVE = "preserve"  # Keep all original data
    OPTIMIZE = "optimize"  # Remove redundant/unnecessary fields
    SUMMARIZE = "summarize"  # Generate summary versions
    ENRICH = "enrich"  # Add computed fields and metadata
    FILTER = "filter"  # Apply filtering criteria
    AGGREGATE = "aggregate"  # Group and aggregate data


@dataclass
class TransformationConfig:
    """Configuration for result transformation"""
    output_format: OutputFormat
    mode: TransformationMode = TransformationMode.PRESERVE
    include_metadata: bool = True
    include_performance_stats: bool = True
    custom_fields: Optional[Dict[str, Any]] = None
    filter_criteria: Optional[Dict[str, Any]] = None
    sorting_criteria: Optional[List[str]] = None
    limit: Optional[int] = None
    encoding: str = "utf-8"
    pretty_print: bool = True
    compression: bool = False


@dataclass
class TransformationResult:
    """Result of a transformation operation"""
    success: bool
    output_format: OutputFormat
    data: Union[str, bytes, Dict[str, Any]]
    metadata: Dict[str, Any]
    transformation_time: float
    original_size: int
    transformed_size: int
    error: Optional[str] = None


class BaseTransformer(ABC):
    """Abstract base class for result transformers"""
    
    def __init__(self, config: TransformationConfig):
        self.config = config
        self.transformation_stats = {
            "total_transformations": 0,
            "successful_transformations": 0,
            "failed_transformations": 0,
            "total_processing_time": 0.0,
        }
    
    @abstractmethod
    async def transform(self, data: Any, context: Optional[Dict[str, Any]] = None) -> TransformationResult:
        """Transform input data to the target format"""
        pass
    
    def _calculate_size(self, data: Any) -> int:
        """Calculate the size of data in bytes"""
        if isinstance(data, str):
            return len(data.encode(self.config.encoding))
        elif isinstance(data, bytes):
            return len(data)
        elif isinstance(data, (dict, list)):
            return len(json.dumps(data, ensure_ascii=False).encode(self.config.encoding))
        else:
            return len(str(data).encode(self.config.encoding))
    
    def _add_metadata(self, data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Add transformation metadata to the result"""
        if not self.config.include_metadata:
            return data
        
        metadata = {
            "transformation": {
                "timestamp": datetime.now().isoformat(),
                "format": self.config.output_format.value,
                "mode": self.config.mode.value,
                "transformer": self.__class__.__name__,
            }
        }
        
        if context:
            metadata["transformation"]["context"] = str(context)
        
        if self.config.custom_fields:
            metadata["transformation"]["custom_fields"] = str(self.config.custom_fields)
        
        # Add metadata to the data
        if isinstance(data, dict):
            data["_metadata"] = metadata
        
        return data
    
    def _filter_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply filtering criteria to data"""
        if not self.config.filter_criteria:
            return data
        
        filtered_data = []
        criteria = self.config.filter_criteria
        
        for item in data:
            include_item = True
            
            # Apply filters
            for field, condition in criteria.items():
                if field not in item:
                    continue
                
                value = item[field]
                
                if isinstance(condition, dict):
                    # Advanced condition
                    if "min" in condition and value < condition["min"]:
                        include_item = False
                        break
                    if "max" in condition and value > condition["max"]:
                        include_item = False
                        break
                    if "contains" in condition and condition["contains"].lower() not in str(value).lower():
                        include_item = False
                        break
                    if "regex" in condition and not re.search(condition["regex"], str(value)):
                        include_item = False
                        break
                else:
                    # Simple equality check
                    if value != condition:
                        include_item = False
                        break
            
            if include_item:
                filtered_data.append(item)
        
        return filtered_data
    
    def _sort_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort data based on criteria"""
        if not self.config.sorting_criteria:
            return data
        
        def sort_key(item):
            key_values = []
            for field in (self.config.sorting_criteria or []):
                # Handle reverse sorting (field prefixed with -)
                reverse = field.startswith('-')
                actual_field = field[1:] if reverse else field
                
                value = item.get(actual_field, 0)
                if reverse:
                    # For reverse sorting, negate numeric values
                    if isinstance(value, (int, float)):
                        value = -value
                    else:
                        # For strings, we'll handle this in sorted() reverse parameter
                        pass
                
                key_values.append(value)
            
            return tuple(key_values)
        
        return sorted(data, key=sort_key)
    
    def _limit_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply limit to data"""
        if self.config.limit:
            return data[:self.config.limit]
        return data


class JSONTransformer(BaseTransformer):
    """Transformer for JSON format"""
    
    async def transform(self, data: Any, context: Optional[Dict[str, Any]] = None) -> TransformationResult:
        start_time = datetime.now()
        original_size = self._calculate_size(data)
        
        try:
            # Prepare data based on mode
            if self.config.mode == TransformationMode.OPTIMIZE:
                transformed_data = self._optimize_data(data)
            elif self.config.mode == TransformationMode.SUMMARIZE:
                transformed_data = self._summarize_data(data)
            elif self.config.mode == TransformationMode.ENRICH:
                transformed_data = self._enrich_data(data)
            else:
                transformed_data = data
            
            # Add metadata
            transformed_data = self._add_metadata(transformed_data, context)
            
            # Convert to JSON
            if self.config.pretty_print:
                json_output = json.dumps(transformed_data, indent=2, ensure_ascii=False, default=str)
            else:
                json_output = json.dumps(transformed_data, ensure_ascii=False, default=str)
            
            transformation_time = (datetime.now() - start_time).total_seconds()
            transformed_size = len(json_output.encode(self.config.encoding))
            
            self.transformation_stats["total_transformations"] += 1
            self.transformation_stats["successful_transformations"] += 1
            self.transformation_stats["total_processing_time"] += transformation_time
            
            return TransformationResult(
                success=True,
                output_format=OutputFormat.JSON,
                data=json_output,
                metadata={
                    "encoding": self.config.encoding,
                    "pretty_print": self.config.pretty_print,
                    "compression": self.config.compression,
                },
                transformation_time=transformation_time,
                original_size=original_size,
                transformed_size=transformed_size
            )
            
        except Exception as e:
            self.transformation_stats["total_transformations"] += 1
            self.transformation_stats["failed_transformations"] += 1
            
            return TransformationResult(
                success=False,
                output_format=OutputFormat.JSON,
                data={},
                metadata={},
                transformation_time=(datetime.now() - start_time).total_seconds(),
                original_size=original_size,
                transformed_size=0,
                error=str(e)
            )
    
    def _optimize_data(self, data: Any) -> Any:
        """Optimize data by removing redundant fields"""
        if isinstance(data, dict):
            # Remove common redundant fields
            optimized = {k: v for k, v in data.items() 
                        if k not in ['_internal', '_temp', '_debug'] and not k.startswith('__')}
            
            # Recursively optimize nested structures
            for key, value in optimized.items():
                optimized[key] = self._optimize_data(value)
            
            return optimized
        elif isinstance(data, list):
            return [self._optimize_data(item) for item in data]
        else:
            return data
    
    def _summarize_data(self, data: Any) -> Any:
        """Create summary version of data"""
        if isinstance(data, dict):
            summary = {}
            
            # Keep essential fields
            essential_fields = ['title', 'url', 'summary', 'score', 'timestamp', 'id']
            for field in essential_fields:
                if field in data:
                    summary[field] = data[field]
            
            # Add summary statistics for lists
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 5:
                    summary[f"{key}_count"] = len(value)
                    summary[f"{key}_sample"] = value[:3]  # First 3 items as sample
            
            return summary
        elif isinstance(data, list):
            if len(data) > 10:
                # For large lists, provide summary
                return {
                    "total_items": len(data),
                    "sample_items": data[:5],
                    "summary_type": "truncated_list"
                }
            else:
                return [self._summarize_data(item) for item in data]
        else:
            return data
    
    def _enrich_data(self, data: Any) -> Any:
        """Enrich data with computed fields"""
        if isinstance(data, dict):
            enriched = data.copy()
            
            # Add computed hash for data integrity
            data_str = json.dumps(data, sort_keys=True, default=str)
            enriched["_hash"] = hashlib.md5(data_str.encode()).hexdigest()
            
            # Add enrichment timestamp
            enriched["_enriched_at"] = datetime.now().isoformat()
            
            # Add size information
            enriched["_size_bytes"] = self._calculate_size(data)
            
            # Recursively enrich nested structures
            for key, value in enriched.items():
                if not key.startswith('_'):  # Don't enrich metadata fields
                    enriched[key] = self._enrich_data(value)
            
            return enriched
        elif isinstance(data, list):
            return [self._enrich_data(item) for item in data]
        else:
            return data


class CSVTransformer(BaseTransformer):
    """Transformer for CSV format"""
    
    async def transform(self, data: Any, context: Optional[Dict[str, Any]] = None) -> TransformationResult:
        start_time = datetime.now()
        original_size = self._calculate_size(data)
        
        try:
            # Convert data to list of dictionaries if needed
            if isinstance(data, dict):
                if "sources" in data and isinstance(data["sources"], list):
                    table_data = data["sources"]
                else:
                    # Convert single dict to list
                    table_data = [data]
            elif isinstance(data, list):
                table_data = data
            else:
                raise TransformationError(f"Cannot convert {type(data)} to CSV")
            
            # Apply filtering and sorting
            table_data = self._filter_data(table_data)
            table_data = self._sort_data(table_data)
            table_data = self._limit_data(table_data)
            
            if not table_data:
                raise TransformationError("No data to convert to CSV")
            
            # Generate CSV content
            import io
            output = io.StringIO()
            
            # Get all unique keys from all dictionaries
            all_keys = set()
            for item in table_data:
                if isinstance(item, dict):
                    all_keys.update(item.keys())
            
            fieldnames = sorted(list(all_keys))
            
            # Remove metadata fields if in optimize mode
            if self.config.mode == TransformationMode.OPTIMIZE:
                fieldnames = [f for f in fieldnames if not f.startswith('_')]
            
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in table_data:
                if isinstance(item, dict):
                    # Flatten complex values
                    row = {}
                    for key in fieldnames:
                        value = item.get(key, '')
                        if isinstance(value, (dict, list)):
                            row[key] = json.dumps(value, default=str)
                        else:
                            row[key] = str(value) if value is not None else ''
                    writer.writerow(row)
            
            csv_output = output.getvalue()
            
            transformation_time = (datetime.now() - start_time).total_seconds()
            transformed_size = len(csv_output.encode(self.config.encoding))
            
            self.transformation_stats["total_transformations"] += 1
            self.transformation_stats["successful_transformations"] += 1
            self.transformation_stats["total_processing_time"] += transformation_time
            
            return TransformationResult(
                success=True,
                output_format=OutputFormat.CSV,
                data=csv_output,
                metadata={
                    "rows": len(table_data),
                    "columns": len(fieldnames),
                    "fieldnames": fieldnames,
                },
                transformation_time=transformation_time,
                original_size=original_size,
                transformed_size=transformed_size
            )
            
        except Exception as e:
            self.transformation_stats["total_transformations"] += 1
            self.transformation_stats["failed_transformations"] += 1
            
            return TransformationResult(
                success=False,
                output_format=OutputFormat.CSV,
                data="",
                metadata={},
                transformation_time=(datetime.now() - start_time).total_seconds(),
                original_size=original_size,
                transformed_size=0,
                error=str(e)
            )


class MarkdownTransformer(BaseTransformer):
    """Transformer for Markdown format"""
    
    async def transform(self, data: Any, context: Optional[Dict[str, Any]] = None) -> TransformationResult:
        start_time = datetime.now()
        original_size = self._calculate_size(data)
        
        try:
            markdown_content = self._generate_markdown(data, context)
            
            transformation_time = (datetime.now() - start_time).total_seconds()
            transformed_size = len(markdown_content.encode(self.config.encoding))
            
            self.transformation_stats["total_transformations"] += 1
            self.transformation_stats["successful_transformations"] += 1
            self.transformation_stats["total_processing_time"] += transformation_time
            
            return TransformationResult(
                success=True,
                output_format=OutputFormat.MARKDOWN,
                data=markdown_content,
                metadata={
                    "sections": self._count_markdown_sections(markdown_content),
                    "word_count": len(markdown_content.split()),
                    "line_count": len(markdown_content.split('\n')),
                },
                transformation_time=transformation_time,
                original_size=original_size,
                transformed_size=transformed_size
            )
            
        except Exception as e:
            self.transformation_stats["total_transformations"] += 1
            self.transformation_stats["failed_transformations"] += 1
            
            return TransformationResult(
                success=False,
                output_format=OutputFormat.MARKDOWN,
                data="",
                metadata={},
                transformation_time=(datetime.now() - start_time).total_seconds(),
                original_size=original_size,
                transformed_size=0,
                error=str(e)
            )
    
    def _generate_markdown(self, data: Any, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate Markdown content from data"""
        md_lines = []
        
        # Title
        if isinstance(data, dict) and "query" in data:
            md_lines.append(f"# Research Results: {data['query']}")
        else:
            md_lines.append("# Research Results")
        
        md_lines.append("")
        
        # Metadata section
        if self.config.include_metadata:
            md_lines.append("## Metadata")
            md_lines.append("")
            
            if isinstance(data, dict):
                if "mission_id" in data:
                    md_lines.append(f"- **Mission ID:** {data['mission_id']}")
                if "started_at" in data:
                    md_lines.append(f"- **Started:** {data['started_at']}")
                if "completed_at" in data:
                    md_lines.append(f"- **Completed:** {data['completed_at']}")
                if "total_sources_analyzed" in data:
                    md_lines.append(f"- **Sources Analyzed:** {data['total_sources_analyzed']}")
            
            md_lines.append("")
        
        # Main content
        if isinstance(data, dict):
            # Handle research results structure
            if "outputs" in data and "research_results" in data["outputs"]:
                research_results = data["outputs"]["research_results"]
                sources = research_results.get("sources", [])
                
                md_lines.append("## Sources")
                md_lines.append("")
                
                for i, source in enumerate(sources, 1):
                    md_lines.append(f"### {i}. {source.get('title', 'Unknown Title')}")
                    md_lines.append("")
                    md_lines.append(f"**URL:** {source.get('url', 'N/A')}")
                    md_lines.append(f"**Domain:** {source.get('domain', 'N/A')}")
                    md_lines.append(f"**Credibility Score:** {source.get('credibility_score', 0):.2f}")
                    md_lines.append(f"**Relevance Score:** {source.get('relevance_score', 0):.2f}")
                    
                    if "content" in source and len(source["content"]) > 100:
                        content_preview = source["content"][:200] + "..."
                        md_lines.append("")
                        md_lines.append("**Content Preview:**")
                        md_lines.append(f"> {content_preview}")
                    
                    md_lines.append("")
                    md_lines.append("---")
                    md_lines.append("")
            
            # Handle quality metrics
            if "quality_metrics" in data:
                metrics = data["quality_metrics"]
                md_lines.append("## Quality Metrics")
                md_lines.append("")
                
                metrics_table = [
                    "| Metric | Value |",
                    "|--------|-------|",
                ]
                
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        formatted_value = f"{value:.3f}" if isinstance(value, float) else str(value)
                    else:
                        formatted_value = str(value)
                    metrics_table.append(f"| {key.replace('_', ' ').title()} | {formatted_value} |")
                
                md_lines.extend(metrics_table)
                md_lines.append("")
        
        elif isinstance(data, list):
            # Handle list of items
            md_lines.append("## Items")
            md_lines.append("")
            
            for i, item in enumerate(data, 1):
                if isinstance(item, dict):
                    title = item.get('title', item.get('name', f'Item {i}'))
                    md_lines.append(f"### {i}. {title}")
                    md_lines.append("")
                    
                    for key, value in item.items():
                        if key not in ['title', 'name'] and not key.startswith('_'):
                            if isinstance(value, str) and len(value) < 100:
                                md_lines.append(f"**{key.replace('_', ' ').title()}:** {value}")
                            elif isinstance(value, (int, float)):
                                md_lines.append(f"**{key.replace('_', ' ').title()}:** {value}")
                    
                    md_lines.append("")
                else:
                    md_lines.append(f"{i}. {str(item)}")
        
        # Performance stats
        if self.config.include_performance_stats and isinstance(data, dict) and "performance_stats" in data:
            stats = data["performance_stats"]
            md_lines.append("## Performance Statistics")
            md_lines.append("")
            
            if "search_pipeline_performance" in stats:
                search_stats = stats["search_pipeline_performance"]
                md_lines.append("### Search Pipeline Performance")
                md_lines.append("")
                md_lines.append(f"- **Total Executions:** {search_stats.get('total_executions', 0)}")
                md_lines.append(f"- **Success Rate:** {search_stats.get('success_rate', 0)}%")
                md_lines.append(f"- **Average Execution Time:** {search_stats.get('average_execution_time', 0):.2f}s")
                md_lines.append("")
        
        # Footer
        md_lines.append("---")
        md_lines.append(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Result Transformation System*")
        
        return "\n".join(md_lines)
    
    def _count_markdown_sections(self, content: str) -> int:
        """Count the number of sections in markdown content"""
        return len(re.findall(r'^#+\s', content, re.MULTILINE))


class TransformationManager:
    """Manager for coordinating result transformations"""
    
    def __init__(self):
        self.transformers = {
            OutputFormat.JSON: JSONTransformer,
            OutputFormat.CSV: CSVTransformer,
            OutputFormat.MARKDOWN: MarkdownTransformer,
            # Add more transformers as needed
        }
        self.transformation_history = []
    
    async def transform(
        self, 
        data: Any, 
        config: TransformationConfig, 
        context: Optional[Dict[str, Any]] = None
    ) -> TransformationResult:
        """Transform data to the specified format"""
        
        if config.output_format not in self.transformers:
            return TransformationResult(
                success=False,
                output_format=config.output_format,
                data={},
                metadata={},
                transformation_time=0.0,
                original_size=0,
                transformed_size=0,
                error=f"Unsupported output format: {config.output_format}"
            )
        
        transformer_class = self.transformers[config.output_format]
        transformer = transformer_class(config)
        
        result = await transformer.transform(data, context)
        
        # Record transformation history
        self.transformation_history.append({
            "timestamp": datetime.now().isoformat(),
            "format": config.output_format.value,
            "mode": config.mode.value,
            "success": result.success,
            "transformation_time": result.transformation_time,
            "size_reduction": ((result.original_size - result.transformed_size) / result.original_size * 100) 
                            if result.original_size > 0 else 0,
        })
        
        return result
    
    async def batch_transform(
        self, 
        data: Any, 
        configs: List[TransformationConfig], 
        context: Optional[Dict[str, Any]] = None
    ) -> List[TransformationResult]:
        """Transform data to multiple formats simultaneously"""
        
        results = []
        for config in configs:
            result = await self.transform(data, config, context)
            results.append(result)
        
        return results
    
    def get_transformation_stats(self) -> Dict[str, Any]:
        """Get transformation statistics"""
        
        total_transformations = len(self.transformation_history)
        successful_transformations = sum(1 for t in self.transformation_history if t["success"])
        
        if total_transformations == 0:
            return {
                "total_transformations": 0,
                "success_rate": 0,
                "average_transformation_time": 0,
                "formats_used": [],
            }
        
        return {
            "total_transformations": total_transformations,
            "success_rate": (successful_transformations / total_transformations) * 100,
            "average_transformation_time": sum(t["transformation_time"] for t in self.transformation_history) / total_transformations,
            "formats_used": list(set(t["format"] for t in self.transformation_history)),
            "average_size_reduction": sum(t["size_reduction"] for t in self.transformation_history) / total_transformations,
        }
    
    def get_supported_formats(self) -> List[OutputFormat]:
        """Get list of supported output formats"""
        return list(self.transformers.keys())


# Convenience functions for easy access
async def transform_to_json(data: Any, mode: TransformationMode = TransformationMode.PRESERVE) -> TransformationResult:
    """Quick transformation to JSON format"""
    config = TransformationConfig(
        output_format=OutputFormat.JSON,
        mode=mode
    )
    manager = TransformationManager()
    return await manager.transform(data, config)


async def transform_to_csv(data: Any, mode: TransformationMode = TransformationMode.OPTIMIZE) -> TransformationResult:
    """Quick transformation to CSV format"""
    config = TransformationConfig(
        output_format=OutputFormat.CSV,
        mode=mode
    )
    manager = TransformationManager()
    return await manager.transform(data, config)


async def transform_to_markdown(data: Any, include_metadata: bool = True) -> TransformationResult:
    """Quick transformation to Markdown format"""
    config = TransformationConfig(
        output_format=OutputFormat.MARKDOWN,
        mode=TransformationMode.ENRICH,
        include_metadata=include_metadata
    )
    manager = TransformationManager()
    return await manager.transform(data, config)


def get_transformation_capabilities() -> Dict[str, Any]:
    """Get transformation system capabilities"""
    return {
        "version": "1.0.0",
        "supported_formats": [format.value for format in OutputFormat],
        "transformation_modes": [mode.value for mode in TransformationMode],
        "features": [
            "multi_format_output",
            "data_optimization",
            "content_summarization",
            "metadata_enrichment",
            "filtering_and_sorting",
            "batch_processing",
            "performance_tracking",
        ]
    }