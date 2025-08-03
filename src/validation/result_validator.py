#!/usr/bin/env python3
"""
Core Result Validation System for CrewAI
Advanced validation capabilities for research results and agent outputs
"""

import asyncio
import json
import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from pathlib import Path

try:
    from rich.console import Console
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    
    class MockConsole:
        def print(self, *args, **kwargs):
            print(*args)
    
    console = MockConsole()

class ValidationLevel(Enum):
    """Validation strictness levels"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    COMPREHENSIVE = "comprehensive"

class ValidationError(Exception):
    """Custom exception for validation errors"""
    def __init__(self, message: str, validation_type: str, severity: str = "error"):
        super().__init__(message)
        self.validation_type = validation_type
        self.severity = severity

@dataclass
class ValidationRule:
    """Individual validation rule definition"""
    name: str
    description: str
    validator_function: Callable
    severity: str = "error"  # error, warning, info
    enabled: bool = True
    category: str = "general"
    weight: float = 1.0

@dataclass
class ValidationResult:
    """Result of validation process"""
    is_valid: bool
    overall_score: float
    rule_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    validation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class ResultValidator:
    """
    Core result validation system for CrewAI research outputs
    """
    
    def __init__(
        self,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        enable_custom_rules: bool = True,
        min_passing_score: float = 0.7,
        output_directory: str = "validation_reports"
    ):
        self.validation_level = validation_level
        self.enable_custom_rules = enable_custom_rules
        self.min_passing_score = min_passing_score
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        # Validation rules storage
        self.rules: Dict[str, ValidationRule] = {}
        self.rule_categories: Dict[str, List[str]] = {}
        
        # Validation history and statistics
        self.validation_history: List[ValidationResult] = []
        self.validation_stats: Dict[str, Any] = {
            "total_validations": 0,
            "passed_validations": 0,
            "failed_validations": 0,
            "avg_score": 0.0
        }
        
        # Setup default validation rules
        self._setup_default_rules()
        self._setup_logging()
        
        console.print(f"[green]✅ Result Validator initialized[/green]")
        console.print(f"[cyan]   • Validation level: {validation_level.value}[/cyan]")
        console.print(f"[cyan]   • Min passing score: {min_passing_score}[/cyan]")
        console.print(f"[cyan]   • Rules loaded: {len(self.rules)}[/cyan]")
    
    def _setup_logging(self):
        """Setup validation logging"""
        
        log_file = self.output_directory / "validation.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('ResultValidator')
    
    def _setup_default_rules(self):
        """Setup default validation rules"""
        
        # Content quality rules
        self.add_rule(ValidationRule(
            name="content_length",
            description="Validate minimum content length",
            validator_function=self._validate_content_length,
            category="content_quality",
            weight=0.8
        ))
        
        self.add_rule(ValidationRule(
            name="structure_coherence",
            description="Validate content structure and coherence",
            validator_function=self._validate_structure,
            category="content_quality",
            weight=1.0
        ))
        
        # Data integrity rules
        self.add_rule(ValidationRule(
            name="data_completeness",
            description="Validate data completeness",
            validator_function=self._validate_completeness,
            category="data_integrity",
            weight=1.2
        ))
        
        self.add_rule(ValidationRule(
            name="format_consistency",
            description="Validate format consistency",
            validator_function=self._validate_format,
            category="data_integrity",
            weight=0.9
        ))
        
        # Source credibility rules
        self.add_rule(ValidationRule(
            name="source_verification",
            description="Validate source credibility",
            validator_function=self._validate_sources,
            category="credibility",
            weight=1.3
        ))
        
        # Accuracy rules
        self.add_rule(ValidationRule(
            name="fact_accuracy",
            description="Validate factual accuracy",
            validator_function=self._validate_facts,
            category="accuracy",
            weight=1.5
        ))
        
        console.print(f"[green]📋 Loaded {len(self.rules)} default validation rules[/green]")
    
    def add_rule(self, rule: ValidationRule):
        """Add validation rule"""
        
        self.rules[rule.name] = rule
        
        # Update category mapping
        if rule.category not in self.rule_categories:
            self.rule_categories[rule.category] = []
        self.rule_categories[rule.category].append(rule.name)
        
        console.print(f"[green]➕ Added validation rule: {rule.name}[/green]")
    
    def remove_rule(self, rule_name: str) -> bool:
        """Remove validation rule"""
        
        if rule_name in self.rules:
            rule = self.rules[rule_name]
            del self.rules[rule_name]
            
            # Update category mapping
            if rule.category in self.rule_categories:
                self.rule_categories[rule.category].remove(rule_name)
                if not self.rule_categories[rule.category]:
                    del self.rule_categories[rule.category]
            
            console.print(f"[yellow]➖ Removed validation rule: {rule_name}[/yellow]")
            return True
        
        return False
    
    def enable_rule(self, rule_name: str) -> bool:
        """Enable validation rule"""
        
        if rule_name in self.rules:
            self.rules[rule_name].enabled = True
            console.print(f"[green]✅ Enabled rule: {rule_name}[/green]")
            return True
        
        return False
    
    def disable_rule(self, rule_name: str) -> bool:
        """Disable validation rule"""
        
        if rule_name in self.rules:
            self.rules[rule_name].enabled = False
            console.print(f"[yellow]❌ Disabled rule: {rule_name}[/yellow]")
            return True
        
        return False
    
    async def validate(
        self,
        data: Any,
        context: Optional[Dict[str, Any]] = None,
        specific_rules: Optional[List[str]] = None
    ) -> ValidationResult:
        """Perform comprehensive validation"""
        
        start_time = datetime.now()
        context = context or {}
        
        console.print(f"[blue]🔍 Starting validation process...[/blue]")
        
        # Initialize result
        result = ValidationResult(
            is_valid=True,
            overall_score=0.0,
            metadata={
                "validation_level": self.validation_level.value,
                "data_type": type(data).__name__,
                "context": context
            }
        )
        
        # Determine which rules to run
        rules_to_run = specific_rules or list(self.rules.keys())
        enabled_rules = [
            name for name in rules_to_run 
            if name in self.rules and self.rules[name].enabled
        ]
        
        total_weight = sum(self.rules[name].weight for name in enabled_rules)
        weighted_score = 0.0
        
        # Run validation rules
        for rule_name in enabled_rules:
            rule = self.rules[rule_name]
            
            try:
                console.print(f"[cyan]   • Running rule: {rule_name}[/cyan]")
                
                # Execute validation function
                rule_result = await self._execute_rule(rule, data, context)
                
                result.rule_results[rule_name] = rule_result
                
                # Calculate weighted score contribution
                rule_score = rule_result.get("score", 0.0)
                weighted_score += rule_score * rule.weight
                
                # Collect messages by severity
                if rule_result.get("severity") == "error":
                    result.errors.append(f"{rule_name}: {rule_result.get('message', '')}")
                elif rule_result.get("severity") == "warning":
                    result.warnings.append(f"{rule_name}: {rule_result.get('message', '')}")
                else:
                    result.info.append(f"{rule_name}: {rule_result.get('message', '')}")
                
            except Exception as e:
                self.logger.error(f"Error executing rule {rule_name}: {e}")
                result.errors.append(f"{rule_name}: Validation rule failed - {str(e)}")
                result.rule_results[rule_name] = {
                    "passed": False,
                    "score": 0.0,
                    "message": f"Rule execution failed: {str(e)}",
                    "severity": "error"
                }
        
        # Calculate overall score
        result.overall_score = weighted_score / total_weight if total_weight > 0 else 0.0
        result.is_valid = result.overall_score >= self.min_passing_score and len(result.errors) == 0
        
        # Calculate validation time
        end_time = datetime.now()
        result.validation_time = (end_time - start_time).total_seconds()
        
        # Update statistics
        self._update_statistics(result)
        
        # Store in history
        self.validation_history.append(result)
        
        console.print(f"[green]✅ Validation completed[/green]")
        console.print(f"[cyan]   • Overall score: {result.overall_score:.2f}[/cyan]")
        console.print(f"[cyan]   • Valid: {result.is_valid}[/cyan]")
        console.print(f"[cyan]   • Errors: {len(result.errors)}[/cyan]")
        console.print(f"[cyan]   • Warnings: {len(result.warnings)}[/cyan]")
        
        return result
    
    async def _execute_rule(
        self, 
        rule: ValidationRule, 
        data: Any, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single validation rule"""
        
        try:
            if asyncio.iscoroutinefunction(rule.validator_function):
                return await rule.validator_function(data, context)
            else:
                return rule.validator_function(data, context)
        except Exception as e:
            return {
                "passed": False,
                "score": 0.0,
                "message": f"Rule execution error: {str(e)}",
                "severity": "error"
            }
    
    # Default validation rule implementations
    
    def _validate_content_length(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate content has minimum length"""
        
        min_length = context.get("min_length", 100)
        
        if isinstance(data, str):
            content_length = len(data.strip())
        elif isinstance(data, dict):
            content_length = len(str(data).strip())
        else:
            content_length = len(str(data))
        
        passed = content_length >= min_length
        score = min(1.0, content_length / min_length) if min_length > 0 else 1.0
        
        return {
            "passed": passed,
            "score": score,
            "message": f"Content length: {content_length} (min: {min_length})",
            "severity": "error" if not passed else "info",
            "details": {"actual_length": content_length, "min_length": min_length}
        }
    
    def _validate_structure(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate content structure and coherence"""
        
        if isinstance(data, dict):
            # Check for required keys
            required_keys = context.get("required_keys", [])
            missing_keys = [key for key in required_keys if key not in data]
            
            passed = len(missing_keys) == 0
            score = 1.0 - (len(missing_keys) / len(required_keys)) if required_keys else 1.0
            
            return {
                "passed": passed,
                "score": score,
                "message": f"Structure check: {len(missing_keys)} missing keys",
                "severity": "error" if not passed else "info",
                "details": {"missing_keys": missing_keys}
            }
        
        elif isinstance(data, str):
            # Basic structure checks for text content
            lines = data.strip().split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            
            # Check for reasonable structure
            has_structure = len(non_empty_lines) > 1
            score = 0.8 if has_structure else 0.3
            
            return {
                "passed": has_structure,
                "score": score,
                "message": f"Text structure: {len(non_empty_lines)} content lines",
                "severity": "warning" if not has_structure else "info",
                "details": {"total_lines": len(lines), "content_lines": len(non_empty_lines)}
            }
        
        return {
            "passed": True,
            "score": 0.5,
            "message": "Structure validation not applicable",
            "severity": "info"
        }
    
    def _validate_completeness(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data completeness"""
        
        completeness_threshold = context.get("completeness_threshold", 0.8)
        
        if isinstance(data, dict):
            total_fields = len(data)
            empty_fields = sum(1 for value in data.values() if not value)
            completeness = (total_fields - empty_fields) / total_fields if total_fields > 0 else 0.0
            
        elif isinstance(data, list):
            total_items = len(data)
            empty_items = sum(1 for item in data if not item)
            completeness = (total_items - empty_items) / total_items if total_items > 0 else 0.0
            
        else:
            # For other types, check if data exists and is not empty
            completeness = 1.0 if data and str(data).strip() else 0.0
        
        passed = completeness >= completeness_threshold
        
        return {
            "passed": passed,
            "score": completeness,
            "message": f"Data completeness: {completeness:.1%}",
            "severity": "error" if not passed else "info",
            "details": {"completeness": completeness, "threshold": completeness_threshold}
        }
    
    def _validate_format(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate format consistency"""
        
        expected_format = context.get("expected_format", "auto")
        
        if expected_format == "json" and isinstance(data, dict):
            # JSON format validation
            try:
                json.dumps(data)
                return {
                    "passed": True,
                    "score": 1.0,
                    "message": "Valid JSON format",
                    "severity": "info"
                }
            except Exception as e:
                return {
                    "passed": False,
                    "score": 0.0,
                    "message": f"Invalid JSON format: {str(e)}",
                    "severity": "error"
                }
        
        elif expected_format == "text" and isinstance(data, str):
            # Text format validation
            printable_ratio = sum(1 for c in data if c.isprintable()) / len(data) if data else 0
            passed = printable_ratio > 0.9
            
            return {
                "passed": passed,
                "score": printable_ratio,
                "message": f"Text format: {printable_ratio:.1%} printable characters",
                "severity": "warning" if not passed else "info"
            }
        
        # Auto-detect format consistency
        return {
            "passed": True,
            "score": 0.8,
            "message": "Format validation passed (auto-detect)",
            "severity": "info"
        }
    
    def _validate_sources(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate source credibility"""
        
        sources = []
        
        # Extract sources from data
        if isinstance(data, dict):
            sources = data.get("sources", [])
        elif isinstance(data, str):
            # Simple URL extraction
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
            sources = re.findall(url_pattern, data)
        
        if not sources:
            return {
                "passed": False,
                "score": 0.0,
                "message": "No sources found",
                "severity": "warning",
                "details": {"source_count": 0}
            }
        
        # Basic source validation
        valid_sources = 0
        for source in sources:
            if isinstance(source, str) and (source.startswith('http') or source.startswith('www')):
                valid_sources += 1
        
        source_score = valid_sources / len(sources) if sources else 0.0
        passed = source_score > 0.5
        
        return {
            "passed": passed,
            "score": source_score,
            "message": f"Source validation: {valid_sources}/{len(sources)} valid sources",
            "severity": "warning" if not passed else "info",
            "details": {"total_sources": len(sources), "valid_sources": valid_sources}
        }
    
    def _validate_facts(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate factual accuracy (basic implementation)"""
        
        # This is a simplified fact validation
        # In a real implementation, this would use external fact-checking APIs
        
        if isinstance(data, str):
            # Look for obvious factual claims
            fact_indicators = [
                r'\d{4}',  # Years
                r'\d+%',   # Percentages
                r'\$\d+',  # Dollar amounts
                r'\d+\s+(million|billion|thousand)',  # Large numbers
            ]
            
            fact_count = sum(len(re.findall(pattern, data, re.IGNORECASE)) 
                           for pattern in fact_indicators)
            
            # Basic heuristic: presence of factual claims
            if fact_count > 0:
                # Assume facts are accurate for now (simplified)
                accuracy_score = 0.8  # Conservative estimate
            else:
                accuracy_score = 0.9  # No specific facts to verify
            
            return {
                "passed": accuracy_score > 0.7,
                "score": accuracy_score,
                "message": f"Fact validation: {fact_count} factual claims detected",
                "severity": "info",
                "details": {"fact_count": fact_count}
            }
        
        return {
            "passed": True,
            "score": 0.8,
            "message": "Fact validation not applicable",
            "severity": "info"
        }
    
    def _update_statistics(self, result: ValidationResult):
        """Update validation statistics"""
        
        self.validation_stats["total_validations"] += 1
        
        if result.is_valid:
            self.validation_stats["passed_validations"] += 1
        else:
            self.validation_stats["failed_validations"] += 1
        
        # Update average score
        total = self.validation_stats["total_validations"]
        current_avg = self.validation_stats["avg_score"]
        self.validation_stats["avg_score"] = (
            (current_avg * (total - 1) + result.overall_score) / total
        )
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics"""
        
        recent_validations = self.validation_history[-10:] if self.validation_history else []
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_stats": self.validation_stats,
            "rule_stats": {
                "total_rules": len(self.rules),
                "enabled_rules": len([r for r in self.rules.values() if r.enabled]),
                "categories": list(self.rule_categories.keys()),
                "rules_by_category": dict(self.rule_categories)
            },
            "recent_performance": {
                "recent_validations": len(recent_validations),
                "recent_avg_score": sum(v.overall_score for v in recent_validations) / len(recent_validations) if recent_validations else 0,
                "recent_pass_rate": sum(1 for v in recent_validations if v.is_valid) / len(recent_validations) if recent_validations else 0
            }
        }
    
    async def export_validation_report(
        self, 
        result: ValidationResult, 
        output_path: Optional[str] = None
    ) -> str:
        """Export detailed validation report"""
        
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.output_directory / f"validation_report_{timestamp}.json")
        
        report_data = {
            "validation_summary": {
                "is_valid": result.is_valid,
                "overall_score": result.overall_score,
                "validation_time": result.validation_time,
                "validation_level": self.validation_level.value
            },
            "rule_results": result.rule_results,
            "messages": {
                "errors": result.errors,
                "warnings": result.warnings,
                "info": result.info
            },
            "metadata": result.metadata,
            "statistics": self.get_validation_statistics()
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            console.print(f"[green]📊 Validation report exported to {output_path}[/green]")
            return output_path
            
        except Exception as e:
            console.print(f"[red]❌ Failed to export report: {e}[/red]")
            return ""