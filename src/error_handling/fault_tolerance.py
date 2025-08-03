#!/usr/bin/env python3
"""
Enhanced Error Handling and Fault Tolerance for CrewAI
Advanced error recovery, retry mechanisms, and fault tolerance capabilities
"""

import asyncio
import functools
import inspect
import logging
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

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

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryStrategy(Enum):
    """Recovery strategy types"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    ESCALATION = "escalation"

@dataclass
class ErrorRecord:
    """Record of an error occurrence"""
    error_id: str
    timestamp: datetime
    component: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""
    recovery_attempts: int = 0
    resolved: bool = False
    resolution_strategy: Optional[str] = None

@dataclass
class RetryConfig:
    """Configuration for retry mechanisms"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: List[Type[Exception]] = field(default_factory=lambda: [Exception])
    stop_on: List[Type[Exception]] = field(default_factory=list)

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern"""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: Type[Exception] = Exception
    name: str = ""

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self.success_count = 0
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                console.print(f"[yellow]🔄 Circuit breaker {self.config.name} transitioning to HALF_OPEN[/yellow]")
            else:
                raise Exception(f"Circuit breaker {self.config.name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except self.config.expected_exception as e:
            self._record_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        return bool(
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.config.recovery_timeout
        )
    
    def _record_success(self):
        """Record successful execution"""
        self.failure_count = 0
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            console.print(f"[green]✅ Circuit breaker {self.config.name} CLOSED[/green]")
    
    def _record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            console.print(f"[red]🚫 Circuit breaker {self.config.name} OPEN[/red]")

class FaultToleranceManager:
    """
    Comprehensive fault tolerance and error handling manager
    """
    
    def __init__(
        self,
        enable_circuit_breakers: bool = True,
        enable_auto_recovery: bool = True,
        error_log_path: str = "error_logs",
        max_error_history: int = 1000
    ):
        self.enable_circuit_breakers = enable_circuit_breakers
        self.enable_auto_recovery = enable_auto_recovery
        self.error_log_path = Path(error_log_path)
        self.error_log_path.mkdir(exist_ok=True)
        
        # Error tracking
        self.error_history: deque = deque(maxlen=max_error_history)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.component_errors: Dict[str, List[ErrorRecord]] = defaultdict(list)
        
        # Recovery mechanisms
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_configs: Dict[str, RetryConfig] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        
        # Monitoring
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self.recovery_success_rates: Dict[str, float] = {}
        
        # Configure logging
        self._setup_logging()
        
        console.print(f"[green]🛡️ Fault Tolerance Manager initialized[/green]")
        console.print(f"[cyan]   • Circuit breakers: {enable_circuit_breakers}[/cyan]")
        console.print(f"[cyan]   • Auto recovery: {enable_auto_recovery}[/cyan]")
        console.print(f"[cyan]   • Error log path: {error_log_path}[/cyan]")
    
    def _setup_logging(self):
        """Setup error logging configuration"""
        
        log_file = self.error_log_path / "fault_tolerance.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('FaultTolerance')
    
    def register_circuit_breaker(
        self,
        name: str,
        config: CircuitBreakerConfig
    ) -> None:
        """Register a circuit breaker for a component"""
        
        if self.enable_circuit_breakers:
            config.name = name
            self.circuit_breakers[name] = CircuitBreaker(config)
            console.print(f"[green]⚡ Registered circuit breaker: {name}[/green]")
    
    def register_retry_config(
        self,
        component: str,
        config: RetryConfig
    ) -> None:
        """Register retry configuration for a component"""
        
        self.retry_configs[component] = config
        console.print(f"[green]🔄 Registered retry config for: {component}[/green]")
    
    def register_fallback_handler(
        self,
        component: str,
        handler: Callable
    ) -> None:
        """Register fallback handler for a component"""
        
        self.fallback_handlers[component] = handler
        console.print(f"[green]🔄 Registered fallback handler for: {component}[/green]")
    
    def register_recovery_strategy(
        self,
        component: str,
        strategy: RecoveryStrategy
    ) -> None:
        """Register recovery strategy for a component"""
        
        self.recovery_strategies[component] = strategy
        console.print(f"[green]📋 Registered recovery strategy for {component}: {strategy.value}[/green]")
    
    async def handle_error(
        self,
        error: Exception,
        component: str,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ) -> Optional[Any]:
        """Handle an error with appropriate recovery strategy"""
        
        error_record = self._create_error_record(
            error, component, context or {}, severity
        )
        
        self.error_history.append(error_record)
        self.error_counts[component] += 1
        self.component_errors[component].append(error_record)
        
        # Log the error
        self.logger.error(f"Error in {component}: {error}", extra={
            'component': component,
            'error_type': type(error).__name__,
            'severity': severity.value,
            'context': context
        })
        
        # Determine recovery strategy
        recovery_strategy = self.recovery_strategies.get(
            component, RecoveryStrategy.RETRY
        )
        
        console.print(f"[red]❌ Error in {component}: {error}[/red]")
        console.print(f"[yellow]🔧 Applying recovery strategy: {recovery_strategy.value}[/yellow]")
        
        try:
            if recovery_strategy == RecoveryStrategy.RETRY:
                return await self._handle_retry_recovery(error_record, component)
            elif recovery_strategy == RecoveryStrategy.FALLBACK:
                return await self._handle_fallback_recovery(error_record, component)
            elif recovery_strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                return await self._handle_circuit_breaker_recovery(error_record, component)
            elif recovery_strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                return await self._handle_graceful_degradation(error_record, component)
            elif recovery_strategy == RecoveryStrategy.ESCALATION:
                return await self._handle_escalation(error_record, component)
            else:
                console.print(f"[yellow]⚠️ No recovery strategy for {component}[/yellow]")
                return None
                
        except Exception as recovery_error:
            console.print(f"[red]❌ Recovery failed for {component}: {recovery_error}[/red]")
            error_record.recovery_attempts += 1
            return None
    
    def _create_error_record(
        self,
        error: Exception,
        component: str,
        context: Dict[str, Any],
        severity: ErrorSeverity
    ) -> ErrorRecord:
        """Create an error record"""
        
        import traceback
        import uuid
        
        return ErrorRecord(
            error_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            component=component,
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            context=context,
            stack_trace=traceback.format_exc()
        )
    
    async def _handle_retry_recovery(
        self,
        error_record: ErrorRecord,
        component: str
    ) -> Optional[Any]:
        """Handle recovery using retry mechanism"""
        
        retry_config = self.retry_configs.get(component, RetryConfig())
        
        for attempt in range(retry_config.max_attempts):
            if attempt > 0:
                delay = self._calculate_retry_delay(retry_config, attempt)
                console.print(f"[yellow]⏳ Retrying {component} in {delay:.2f}s (attempt {attempt + 1}/{retry_config.max_attempts})[/yellow]")
                await asyncio.sleep(delay)
            
            error_record.recovery_attempts += 1
            
            # Here you would retry the original operation
            # This is a placeholder - in real implementation, you'd need to
            # store the original function call and parameters
            console.print(f"[blue]🔄 Retry attempt {attempt + 1} for {component}[/blue]")
            
            # Simulate retry success/failure
            if attempt >= retry_config.max_attempts - 1:
                console.print(f"[red]❌ All retry attempts failed for {component}[/red]")
                return None
        
        error_record.resolved = True
        error_record.resolution_strategy = "retry"
        console.print(f"[green]✅ Retry recovery successful for {component}[/green]")
        return "retry_success"
    
    async def _handle_fallback_recovery(
        self,
        error_record: ErrorRecord,
        component: str
    ) -> Optional[Any]:
        """Handle recovery using fallback mechanism"""
        
        fallback_handler = self.fallback_handlers.get(component)
        
        if not fallback_handler:
            console.print(f"[yellow]⚠️ No fallback handler registered for {component}[/yellow]")
            return None
        
        try:
            console.print(f"[blue]🔄 Executing fallback for {component}[/blue]")
            
            if asyncio.iscoroutinefunction(fallback_handler):
                result = await fallback_handler(error_record)
            else:
                result = fallback_handler(error_record)
            
            error_record.resolved = True
            error_record.resolution_strategy = "fallback"
            console.print(f"[green]✅ Fallback recovery successful for {component}[/green]")
            return result
            
        except Exception as fallback_error:
            console.print(f"[red]❌ Fallback failed for {component}: {fallback_error}[/red]")
            return None
    
    async def _handle_circuit_breaker_recovery(
        self,
        error_record: ErrorRecord,
        component: str
    ) -> Optional[Any]:
        """Handle recovery using circuit breaker pattern"""
        
        circuit_breaker = self.circuit_breakers.get(component)
        
        if not circuit_breaker:
            console.print(f"[yellow]⚠️ No circuit breaker registered for {component}[/yellow]")
            return await self._handle_retry_recovery(error_record, component)
        
        if circuit_breaker.state == CircuitBreakerState.OPEN:
            console.print(f"[red]🚫 Circuit breaker OPEN for {component} - operation blocked[/red]")
            return None
        
        # Circuit breaker would be used in the original operation call
        # This is a placeholder for the recovery mechanism
        console.print(f"[blue]⚡ Circuit breaker recovery for {component}[/blue]")
        
        error_record.resolved = True
        error_record.resolution_strategy = "circuit_breaker"
        return "circuit_breaker_recovery"
    
    async def _handle_graceful_degradation(
        self,
        error_record: ErrorRecord,
        component: str
    ) -> Optional[Any]:
        """Handle recovery using graceful degradation"""
        
        console.print(f"[blue]📉 Applying graceful degradation for {component}[/blue]")
        
        # Implement graceful degradation logic
        # This could involve:
        # - Reducing functionality
        # - Using cached data
        # - Simplified operations
        # - Default responses
        
        degraded_result = {
            "status": "degraded",
            "component": component,
            "message": "Operating in degraded mode",
            "timestamp": datetime.now().isoformat()
        }
        
        error_record.resolved = True
        error_record.resolution_strategy = "graceful_degradation"
        console.print(f"[green]📉 Graceful degradation applied for {component}[/green]")
        return degraded_result
    
    async def _handle_escalation(
        self,
        error_record: ErrorRecord,
        component: str
    ) -> Optional[Any]:
        """Handle recovery using escalation"""
        
        console.print(f"[red]🚨 Escalating error for {component}[/red]")
        
        # Implement escalation logic
        # This could involve:
        # - Notifying administrators
        # - Creating incident tickets
        # - Triggering alerts
        # - Switching to backup systems
        
        escalation_result = {
            "status": "escalated",
            "component": component,
            "error_id": error_record.error_id,
            "severity": error_record.severity.value,
            "timestamp": datetime.now().isoformat()
        }
        
        error_record.resolution_strategy = "escalation"
        console.print(f"[red]🚨 Error escalated for {component}[/red]")
        return escalation_result
    
    def _calculate_retry_delay(self, config: RetryConfig, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter"""
        
        delay = min(
            config.base_delay * (config.exponential_base ** attempt),
            config.max_delay
        )
        
        if config.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay
    
    def with_fault_tolerance(
        self,
        component: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    ):
        """Decorator for adding fault tolerance to functions"""
        
        def decorator(func: Callable):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    # Register configurations if provided
                    if retry_config:
                        self.register_retry_config(component, retry_config)
                    
                    if circuit_breaker_config and self.enable_circuit_breakers:
                        self.register_circuit_breaker(component, circuit_breaker_config)
                    
                    # Execute function with circuit breaker if available
                    if component in self.circuit_breakers:
                        return self.circuit_breakers[component].call(func, *args, **kwargs)
                    else:
                        if asyncio.iscoroutinefunction(func):
                            return await func(*args, **kwargs)
                        else:
                            return func(*args, **kwargs)
                
                except Exception as e:
                    context = {
                        'function': func.__name__,
                        'args': str(args)[:200],  # Limit context size
                        'kwargs': str(kwargs)[:200]
                    }
                    
                    return await self.handle_error(e, component, context, severity)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    # Register configurations if provided
                    if retry_config:
                        self.register_retry_config(component, retry_config)
                    
                    if circuit_breaker_config and self.enable_circuit_breakers:
                        self.register_circuit_breaker(component, circuit_breaker_config)
                    
                    # Execute function with circuit breaker if available
                    if component in self.circuit_breakers:
                        return self.circuit_breakers[component].call(func, *args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                
                except Exception as e:
                    context = {
                        'function': func.__name__,
                        'args': str(args)[:200],
                        'kwargs': str(kwargs)[:200]
                    }
                    
                    # For sync functions, we can't use async error handling
                    # Log the error and apply basic recovery
                    error_record = self._create_error_record(e, component, context, severity)
                    self.error_history.append(error_record)
                    self.error_counts[component] += 1
                    
                    console.print(f"[red]❌ Error in {component}: {e}[/red]")
                    
                    # Apply fallback if available
                    if component in self.fallback_handlers:
                        try:
                            return self.fallback_handlers[component](error_record)
                        except Exception as fallback_error:
                            console.print(f"[red]❌ Fallback failed: {fallback_error}[/red]")
                    
                    raise e
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        
        total_errors = len(self.error_history)
        
        # Error distribution by component
        component_stats = {}
        for component, errors in self.component_errors.items():
            resolved_count = len([e for e in errors if e.resolved])
            component_stats[component] = {
                'total_errors': len(errors),
                'resolved_errors': resolved_count,
                'resolution_rate': (resolved_count / len(errors)) * 100 if errors else 0,
                'avg_recovery_attempts': sum(e.recovery_attempts for e in errors) / len(errors) if errors else 0
            }
        
        # Error distribution by severity
        severity_stats = {}
        for severity in ErrorSeverity:
            count = len([e for e in self.error_history if e.severity == severity])
            severity_stats[severity.value] = count
        
        # Error distribution by type
        error_type_stats = {}
        for error in self.error_history:
            error_type_stats[error.error_type] = error_type_stats.get(error.error_type, 0) + 1
        
        # Recovery strategy effectiveness
        strategy_stats = {}
        for error in self.error_history:
            if error.resolution_strategy:
                strategy = error.resolution_strategy
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {'total': 0, 'successful': 0}
                strategy_stats[strategy]['total'] += 1
                if error.resolved:
                    strategy_stats[strategy]['successful'] += 1
        
        # Calculate success rates
        for strategy, stats in strategy_stats.items():
            stats['success_rate'] = (stats['successful'] / stats['total']) * 100 if stats['total'] > 0 else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_errors': total_errors,
            'component_statistics': component_stats,
            'severity_distribution': severity_stats,
            'error_type_distribution': error_type_stats,
            'recovery_strategy_effectiveness': strategy_stats,
            'circuit_breaker_states': {
                name: breaker.state.value
                for name, breaker in self.circuit_breakers.items()
            }
        }
    
    async def export_error_report(self, export_path: Optional[str] = None) -> str:
        """Export comprehensive error report"""
        
        if not export_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = str(self.error_log_path / f"error_report_{timestamp}.json")
        
        report_data = {
            'report_timestamp': datetime.now().isoformat(),
            'statistics': self.get_error_statistics(),
            'error_history': [
                {
                    'error_id': error.error_id,
                    'timestamp': error.timestamp.isoformat(),
                    'component': error.component,
                    'error_type': error.error_type,
                    'error_message': error.error_message,
                    'severity': error.severity.value,
                    'recovery_attempts': error.recovery_attempts,
                    'resolved': error.resolved,
                    'resolution_strategy': error.resolution_strategy,
                    'context': error.context
                }
                for error in list(self.error_history)
            ]
        }
        
        try:
            import json
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            console.print(f"[green]✅ Error report exported to {export_path}[/green]")
            return str(export_path)
            
        except Exception as e:
            console.print(f"[red]❌ Failed to export error report: {e}[/red]")
            return ""
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': 'healthy',
            'components': {},
            'alerts': []
        }
        
        # Check each component's health
        for component in self.component_errors.keys():
            recent_errors = [
                e for e in self.component_errors[component]
                if e.timestamp > datetime.now() - timedelta(minutes=15)
            ]
            
            error_rate = len(recent_errors)
            health = 'healthy'
            
            if error_rate > 10:
                health = 'critical'
                health_status['overall_health'] = 'critical'
                health_status['alerts'].append(f"High error rate in {component}")
            elif error_rate > 5:
                health = 'degraded'
                if health_status['overall_health'] == 'healthy':
                    health_status['overall_health'] = 'degraded'
                health_status['alerts'].append(f"Elevated error rate in {component}")
            
            health_status['components'][component] = {
                'health': health,
                'recent_error_count': error_rate,
                'total_errors': len(self.component_errors[component])
            }
        
        # Check circuit breaker states
        for name, breaker in self.circuit_breakers.items():
            if breaker.state == CircuitBreakerState.OPEN:
                health_status['alerts'].append(f"Circuit breaker {name} is OPEN")
                health_status['overall_health'] = 'degraded'
        
        return health_status