"""
Enhanced logging system for Crew-Camufox
Provides structured logging with rich output, file rotation, and monitoring capabilities
"""

import logging
import logging.handlers
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
from rich.logging import RichHandler
from rich.console import Console

from ..config import get_settings, LogLevel


class ResearchLogger:
    """
    Enhanced logger for research operations with structured output and metrics
    """

    def __init__(self, name: str, console: Optional[Console] = None):
        self.name = name
        self.console = console or Console()
        self.settings = get_settings()

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, self.settings.log_level))

        # Clear existing handlers
        self.logger.handlers.clear()

        # Setup handlers
        self._setup_console_handler()
        if self.settings.log_to_file:
            self._setup_file_handler()

    def _setup_console_handler(self):
        """Setup rich console handler"""
        handler = RichHandler(
            console=self.console,
            rich_tracebacks=True,
            show_time=True,
            show_path=True,
            enable_link_path=True,
        )
        handler.setFormatter(logging.Formatter(fmt="[%(name)s] %(message)s", datefmt="[%X]"))
        self.logger.addHandler(handler)

    def _setup_file_handler(self):
        """Setup file handler with rotation"""
        log_file = Path(self.settings.logs_dir) / f"{self.name}.log"

        if self.settings.log_rotation:
            # Rotating file handler - 10MB max, keep 5 backups
            handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"  # 10MB
            )
        else:
            handler = logging.FileHandler(log_file, encoding="utf-8")

        # Detailed format for file logs
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def debug(self, message: str, **context):
        """Log debug message with context"""
        self.logger.debug(message, extra=self._format_context(context))

    def info(self, message: str, **context):
        """Log info message with context"""
        self.logger.info(message, extra=self._format_context(context))

    def warning(self, message: str, **context):
        """Log warning message with context"""
        self.logger.warning(message, extra=self._format_context(context))

    def error(self, message: str, **context):
        """Log error message with context"""
        self.logger.error(message, extra=self._format_context(context))

    def critical(self, message: str, **context):
        """Log critical message with context"""
        self.logger.critical(message, extra=self._format_context(context))

    def log_agent_action(self, agent_name: str, action: str, **context):
        """Log agent-specific actions"""
        self.info(f"{agent_name}: {action}", agent=agent_name, action=action, **context)

    def log_research_metrics(self, mission_id: str, metrics: Dict[str, Any]):
        """Log research performance metrics"""
        self.info(
            f"Research metrics for mission {mission_id}",
            mission_id=mission_id,
            metrics=metrics,
            event_type="research_metrics",
        )

    def log_search_operation(self, engine: str, query: str, results_count: int, duration: float):
        """Log search operation details"""
        self.info(
            f"Search completed: {engine} returned {results_count} results in {duration:.2f}s",
            engine=engine,
            query=query,
            results_count=results_count,
            duration=duration,
            event_type="search_operation",
        )

    def log_browser_action(
        self, action: str, url: Optional[str] = None, success: bool = True, **context
    ):
        """Log browser automation actions"""
        status = "SUCCESS" if success else "FAILED"
        message = f"Browser {action}: {status}"
        if url:
            message += f" - {url}"

        self.info(
            message,
            browser_action=action,
            url=url,
            success=success,
            event_type="browser_action",
            **context,
        )

    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log error with full context information"""
        self.error(
            f"Error occurred: {str(error)}",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context,
            event_type="error",
        )

    def log_performance_warning(self, operation: str, duration: float, threshold: float):
        """Log performance warnings for slow operations"""
        self.warning(
            f"Performance warning: {operation} took {duration:.2f}s (threshold: {threshold:.2f}s)",
            operation=operation,
            duration=duration,
            threshold=threshold,
            event_type="performance_warning",
        )

    def _format_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Format context for structured logging"""
        formatted = {}
        for key, value in context.items():
            try:
                # Ensure all values are JSON serializable
                json.dumps(value)
                formatted[key] = value
            except (TypeError, ValueError):
                formatted[key] = str(value)

        # Add timestamp
        formatted["timestamp"] = datetime.now().isoformat()

        return formatted


class LoggerManager:
    """
    Centralized logger management
    """

    _loggers: Dict[str, ResearchLogger] = {}
    _console: Optional[Console] = None

    @classmethod
    def get_logger(cls, name: str) -> ResearchLogger:
        """Get or create logger instance"""
        if name not in cls._loggers:
            if cls._console is None:
                cls._console = Console()
            cls._loggers[name] = ResearchLogger(name, cls._console)
        return cls._loggers[name]

    @classmethod
    def set_console(cls, console: Console):
        """Set shared console instance"""
        cls._console = console
        # Update existing loggers
        for logger in cls._loggers.values():
            logger.console = console
            logger._setup_console_handler()

    @classmethod
    def shutdown(cls):
        """Shutdown all loggers"""
        for logger in cls._loggers.values():
            for handler in logger.logger.handlers:
                handler.close()
        cls._loggers.clear()


# Convenience functions
def get_logger(name: str) -> ResearchLogger:
    """Get logger instance"""
    return LoggerManager.get_logger(name)


def set_log_level(level: Union[str, LogLevel]):
    """Set global log level"""
    if isinstance(level, str):
        level = LogLevel(level.upper())

    log_level = getattr(logging, level.value)
    logging.getLogger().setLevel(log_level)

    # Update all existing loggers
    for logger in LoggerManager._loggers.values():
        logger.logger.setLevel(log_level)


def configure_logging(console: Optional[Console] = None):
    """Configure global logging settings"""
    if console:
        LoggerManager.set_console(console)

    # Set root logger level
    settings = get_settings()
    logging.getLogger().setLevel(getattr(logging, settings.log_level.value))


# Create main application logger
main_logger = get_logger("crew-camufox")


class PerformanceMonitor:
    """
    Performance monitoring utilities
    """

    def __init__(self, logger: Optional[ResearchLogger] = None):
        self.logger = logger or get_logger("performance")

    def log_operation_time(self, operation: str, duration: float, metadata: Optional[Dict] = None):
        """Log operation timing"""
        self.logger.info(
            f"Operation '{operation}' completed in {duration:.2f}s",
            operation=operation,
            duration=duration,
            metadata=metadata or {},
            event_type="operation_timing",
        )

    def log_memory_usage(self, operation: str, memory_mb: float):
        """Log memory usage"""
        self.logger.info(
            f"Memory usage for '{operation}': {memory_mb:.2f} MB",
            operation=operation,
            memory_mb=memory_mb,
            event_type="memory_usage",
        )

    def log_resource_usage(self, cpu_percent: float, memory_mb: float, disk_usage_mb: float):
        """Log system resource usage"""
        self.logger.info(
            f"Resource usage - CPU: {cpu_percent:.1f}%, Memory: {memory_mb:.1f}MB, Disk: {disk_usage_mb:.1f}MB",
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            disk_usage_mb=disk_usage_mb,
            event_type="resource_usage",
        )


# Global performance monitor
performance_monitor = PerformanceMonitor()
