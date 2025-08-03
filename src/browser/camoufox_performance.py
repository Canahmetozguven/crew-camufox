#!/usr/bin/env python3
"""
Advanced Camoufox Performance Optimization
Browser automation performance features and optimization techniques
"""

import asyncio
import psutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import logging
from collections import defaultdict, deque

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

class PerformanceLevel(Enum):
    """Performance optimization levels"""
    MINIMAL = "minimal"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

class ResourceType(Enum):
    """System resource types"""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    DISK = "disk"

@dataclass
class PerformanceMetrics:
    """Browser performance metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    page_load_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    network_requests: int = 0
    dom_nodes: int = 0
    javascript_heap_size: float = 0.0
    response_time: float = 0.0
    errors_count: int = 0

@dataclass
class OptimizationConfig:
    """Browser optimization configuration"""
    performance_level: PerformanceLevel = PerformanceLevel.BALANCED
    enable_resource_blocking: bool = True
    enable_image_optimization: bool = True
    enable_javascript_optimization: bool = True
    enable_css_optimization: bool = True
    enable_caching: bool = True
    enable_compression: bool = True
    max_concurrent_tabs: int = 5
    memory_limit_mb: int = 2048
    timeout_seconds: int = 30
    user_agent_rotation: bool = True
    proxy_rotation: bool = False

@dataclass
class ResourceBlockingRule:
    """Resource blocking rule configuration"""
    resource_type: str
    pattern: str
    action: str = "block"  # block, allow, modify
    priority: int = 1

class CamoufoxPerformanceOptimizer:
    """
    Advanced performance optimization for Camoufox browser automation
    """
    
    def __init__(
        self,
        optimization_config: Optional[OptimizationConfig] = None,
        metrics_retention_hours: int = 24,
        enable_real_time_monitoring: bool = True
    ):
        self.config = optimization_config or OptimizationConfig()
        self.metrics_retention_hours = metrics_retention_hours
        self.enable_real_time_monitoring = enable_real_time_monitoring
        
        # Performance tracking
        self.metrics_history: deque = deque(maxlen=10000)
        self.session_metrics: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.resource_usage_history: deque = deque(maxlen=1000)
        
        # Optimization features
        self.resource_blocking_rules: List[ResourceBlockingRule] = []
        self.cached_resources: Dict[str, Any] = {}
        self.optimized_user_agents: List[str] = []
        self.performance_profiles: Dict[str, OptimizationConfig] = {}
        
        # Monitoring
        self.active_sessions: Dict[str, Dict] = {}
        self.performance_alerts: List[Dict] = []
        self.optimization_stats: Dict[str, Any] = defaultdict(int)
        
        # Initialize default optimizations
        self._setup_default_optimizations()
        self._setup_logging()
        
        console.print(f"[green]🚀 Camoufox Performance Optimizer initialized[/green]")
        console.print(f"[cyan]   • Performance level: {self.config.performance_level.value}[/cyan]")
        console.print(f"[cyan]   • Real-time monitoring: {enable_real_time_monitoring}[/cyan]")
        console.print(f"[cyan]   • Memory limit: {self.config.memory_limit_mb}MB[/cyan]")
    
    def _setup_logging(self):
        """Setup performance logging"""
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        self.logger = logging.getLogger('CamoufoxPerformance')
    
    def _setup_default_optimizations(self):
        """Setup default performance optimizations"""
        
        # Default resource blocking rules
        default_blocking_rules = [
            ResourceBlockingRule("image", r"\.(gif|png|jpg|jpeg|webp)$", "block", 1),
            ResourceBlockingRule("font", r"\.(woff|woff2|ttf|otf)$", "block", 2),
            ResourceBlockingRule("analytics", r"google-analytics|googletagmanager", "block", 3),
            ResourceBlockingRule("ads", r"doubleclick|googlesyndication|amazon-adsystem", "block", 3),
            ResourceBlockingRule("social", r"facebook\.com|twitter\.com|linkedin\.com", "block", 2)
        ]
        
        # Apply blocking rules based on performance level
        if self.config.performance_level in [PerformanceLevel.AGGRESSIVE, PerformanceLevel.MAXIMUM]:
            self.resource_blocking_rules.extend(default_blocking_rules)
        elif self.config.performance_level == PerformanceLevel.BALANCED:
            # Only block ads and analytics
            self.resource_blocking_rules.extend([
                rule for rule in default_blocking_rules 
                if rule.resource_type in ["analytics", "ads"]
            ])
        
        # Default optimized user agents
        self.optimized_user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]
        
        # Performance profiles
        self.performance_profiles = {
            "speed_focused": OptimizationConfig(
                performance_level=PerformanceLevel.MAXIMUM,
                enable_resource_blocking=True,
                enable_image_optimization=True,
                max_concurrent_tabs=3,
                memory_limit_mb=1024,
                timeout_seconds=15
            ),
            "memory_efficient": OptimizationConfig(
                performance_level=PerformanceLevel.AGGRESSIVE,
                enable_resource_blocking=True,
                max_concurrent_tabs=2,
                memory_limit_mb=512,
                timeout_seconds=20
            ),
            "balanced_performance": OptimizationConfig(
                performance_level=PerformanceLevel.BALANCED,
                max_concurrent_tabs=5,
                memory_limit_mb=2048,
                timeout_seconds=30
            )
        }
    
    async def optimize_browser_launch(self, browser_args: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize browser launch parameters"""
        
        optimized_args = browser_args.copy()
        
        # Performance-based optimizations
        performance_flags = []
        
        if self.config.performance_level in [PerformanceLevel.AGGRESSIVE, PerformanceLevel.MAXIMUM]:
            performance_flags.extend([
                "--disable-extensions",
                "--disable-plugins",
                "--disable-images",
                "--disable-javascript",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding"
            ])
        
        if self.config.performance_level == PerformanceLevel.MAXIMUM:
            performance_flags.extend([
                "--memory-pressure-off",
                "--max_old_space_size=4096",
                "--aggressive-cache-discard"
            ])
        
        # Memory optimizations
        if self.config.memory_limit_mb < 1024:
            performance_flags.extend([
                "--memory-pressure-off",
                "--aggressive-cache-discard",
                "--purge-memory-button"
            ])
        
        # Add performance flags to browser arguments
        if "args" in optimized_args:
            optimized_args["args"].extend(performance_flags)
        else:
            optimized_args["args"] = performance_flags
        
        console.print(f"[green]⚡ Browser launch optimized with {len(performance_flags)} performance flags[/green]")
        
        return optimized_args
    
    async def apply_page_optimizations(self, page, url: str) -> Dict[str, Any]:
        """Apply page-level optimizations"""
        
        optimization_results = {
            "url": url,
            "optimizations_applied": [],
            "blocked_resources": 0,
            "performance_score": 0.0
        }
        
        try:
            # Resource blocking
            if self.config.enable_resource_blocking:
                await self._setup_resource_blocking(page)
                optimization_results["optimizations_applied"].append("resource_blocking")
            
            # JavaScript optimization
            if self.config.enable_javascript_optimization:
                await self._optimize_javascript(page)
                optimization_results["optimizations_applied"].append("javascript_optimization")
            
            # CSS optimization
            if self.config.enable_css_optimization:
                await self._optimize_css(page)
                optimization_results["optimizations_applied"].append("css_optimization")
            
            # Image optimization
            if self.config.enable_image_optimization:
                await self._optimize_images(page)
                optimization_results["optimizations_applied"].append("image_optimization")
            
            # Caching optimization
            if self.config.enable_caching:
                await self._setup_caching(page)
                optimization_results["optimizations_applied"].append("caching")
            
            # Set timeouts
            page.set_default_timeout(self.config.timeout_seconds * 1000)
            
            console.print(f"[green]🎯 Applied {len(optimization_results['optimizations_applied'])} optimizations to {url}[/green]")
            
        except Exception as e:
            self.logger.error(f"Error applying page optimizations: {e}")
            optimization_results["error"] = str(e)
        
        return optimization_results
    
    async def _setup_resource_blocking(self, page):
        """Setup resource blocking for performance"""
        
        blocked_count = 0
        
        async def handle_request(request):
            nonlocal blocked_count
            
            url = request.url
            resource_type = request.resource_type
            
            # Check blocking rules
            should_block = False
            for rule in self.resource_blocking_rules:
                if self._match_blocking_rule(url, resource_type, rule):
                    should_block = True
                    break
            
            if should_block:
                await request.abort()
                blocked_count += 1
            else:
                await request.continue_()
        
        await page.route("**/*", handle_request)
        
        self.optimization_stats["blocked_resources"] += blocked_count
    
    def _match_blocking_rule(self, url: str, resource_type: str, rule: ResourceBlockingRule) -> bool:
        """Check if URL matches blocking rule"""
        
        import re
        
        # Check resource type match
        if rule.resource_type != "all" and rule.resource_type != resource_type:
            return False
        
        # Check pattern match
        try:
            return bool(re.search(rule.pattern, url, re.IGNORECASE))
        except re.error:
            return False
    
    async def _optimize_javascript(self, page):
        """Optimize JavaScript execution"""
        
        # Disable heavy JavaScript features for performance
        if self.config.performance_level in [PerformanceLevel.AGGRESSIVE, PerformanceLevel.MAXIMUM]:
            await page.add_init_script("""
                // Disable heavy operations
                window.addEventListener = () => {};
                window.setInterval = () => {};
                window.setTimeout = (fn, delay) => delay < 100 ? fn() : {};
                
                // Optimize DOM operations
                const originalQuerySelectorAll = document.querySelectorAll;
                document.querySelectorAll = function(selector) {
                    const result = originalQuerySelectorAll.call(this, selector);
                    return Array.from(result).slice(0, 100); // Limit results
                };
            """)
        
        self.optimization_stats["javascript_optimizations"] += 1
    
    async def _optimize_css(self, page):
        """Optimize CSS loading and rendering"""
        
        # Remove non-essential CSS for performance
        await page.add_init_script("""
            // Remove heavy CSS animations
            const style = document.createElement('style');
            style.textContent = `
                *, *::before, *::after {
                    animation-duration: 0.01ms !important;
                    animation-delay: -0.01ms !important;
                    transition-duration: 0.01ms !important;
                    transition-delay: -0.01ms !important;
                }
            `;
            document.head.appendChild(style);
        """)
        
        self.optimization_stats["css_optimizations"] += 1
    
    async def _optimize_images(self, page):
        """Optimize image loading"""
        
        # Replace images with placeholders for performance
        if self.config.performance_level == PerformanceLevel.MAXIMUM:
            await page.add_init_script("""
                // Replace images with lightweight placeholders
                setTimeout(() => {
                    const images = document.querySelectorAll('img');
                    images.forEach(img => {
                        img.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEwMCIgZmlsbD0iI2VlZSIvPjwvc3ZnPg==';
                    });
                }, 100);
            """)
        
        self.optimization_stats["image_optimizations"] += 1
    
    async def _setup_caching(self, page):
        """Setup intelligent caching"""
        
        # Enable aggressive caching for static resources
        await page.route("**/*.{js,css,png,jpg,jpeg,gif,woff,woff2}", lambda request: (
            request.continue_() if request.method == "GET" else request.abort()
        ))
        
        self.optimization_stats["caching_setups"] += 1
    
    async def monitor_performance(self, page, session_id: str) -> PerformanceMetrics:
        """Monitor page performance metrics"""
        
        try:
            # Collect performance metrics
            performance_data = await page.evaluate("""
                () => {
                    const perf = performance;
                    const timing = perf.timing;
                    const memory = perf.memory || {};
                    
                    return {
                        loadTime: timing.loadEventEnd - timing.navigationStart,
                        domContentLoaded: timing.domContentLoadedEventEnd - timing.navigationStart,
                        firstPaint: perf.getEntriesByType('paint')[0]?.startTime || 0,
                        memoryUsed: memory.usedJSHeapSize || 0,
                        memoryTotal: memory.totalJSHeapSize || 0,
                        resourceCount: perf.getEntriesByType('resource').length,
                        domNodes: document.querySelectorAll('*').length
                    };
                }
            """)
            
            # System resource usage
            system_memory = psutil.virtual_memory()
            system_cpu = psutil.cpu_percent(interval=0.1)
            
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                page_load_time=performance_data.get("loadTime", 0) / 1000,
                memory_usage=performance_data.get("memoryUsed", 0) / (1024 * 1024),
                cpu_usage=system_cpu,
                network_requests=performance_data.get("resourceCount", 0),
                dom_nodes=performance_data.get("domNodes", 0),
                javascript_heap_size=performance_data.get("memoryTotal", 0) / (1024 * 1024),
                response_time=performance_data.get("firstPaint", 0) / 1000
            )
            
            # Store metrics
            self.metrics_history.append(metrics)
            self.session_metrics[session_id].append(metrics)
            
            # Check for performance alerts
            await self._check_performance_alerts(metrics, session_id)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error monitoring performance: {e}")
            return PerformanceMetrics()
    
    async def _check_performance_alerts(self, metrics: PerformanceMetrics, session_id: str):
        """Check for performance alerts"""
        
        alerts = []
        
        # Memory usage alert
        if metrics.memory_usage > self.config.memory_limit_mb * 0.8:
            alerts.append({
                "type": "memory_warning",
                "message": f"High memory usage: {metrics.memory_usage:.1f}MB",
                "session_id": session_id,
                "timestamp": metrics.timestamp
            })
        
        # Page load time alert
        if metrics.page_load_time > self.config.timeout_seconds * 0.8:
            alerts.append({
                "type": "load_time_warning",
                "message": f"Slow page load: {metrics.page_load_time:.2f}s",
                "session_id": session_id,
                "timestamp": metrics.timestamp
            })
        
        # CPU usage alert
        if metrics.cpu_usage > 80:
            alerts.append({
                "type": "cpu_warning",
                "message": f"High CPU usage: {metrics.cpu_usage:.1f}%",
                "session_id": session_id,
                "timestamp": metrics.timestamp
            })
        
        self.performance_alerts.extend(alerts)
        
        # Log alerts
        for alert in alerts:
            console.print(f"[yellow]⚠️ Performance Alert: {alert['message']}[/yellow]")
    
    def get_performance_analysis(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive performance analysis"""
        
        if session_id and session_id in self.session_metrics:
            metrics_list = self.session_metrics[session_id]
        else:
            metrics_list = list(self.metrics_history)
        
        if not metrics_list:
            return {"error": "No performance data available"}
        
        # Calculate statistics
        load_times = [m.page_load_time for m in metrics_list if m.page_load_time > 0]
        memory_usage = [m.memory_usage for m in metrics_list if m.memory_usage > 0]
        cpu_usage = [m.cpu_usage for m in metrics_list if m.cpu_usage > 0]
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "total_measurements": len(metrics_list),
            "performance_summary": {
                "avg_load_time": sum(load_times) / len(load_times) if load_times else 0,
                "max_load_time": max(load_times) if load_times else 0,
                "avg_memory_usage": sum(memory_usage) / len(memory_usage) if memory_usage else 0,
                "peak_memory_usage": max(memory_usage) if memory_usage else 0,
                "avg_cpu_usage": sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0,
                "peak_cpu_usage": max(cpu_usage) if cpu_usage else 0
            },
            "optimization_effectiveness": {
                "blocked_resources": self.optimization_stats["blocked_resources"],
                "optimizations_applied": sum([
                    self.optimization_stats.get("javascript_optimizations", 0),
                    self.optimization_stats.get("css_optimizations", 0),
                    self.optimization_stats.get("image_optimizations", 0),
                    self.optimization_stats.get("caching_setups", 0)
                ])
            },
            "alerts_summary": {
                "total_alerts": len(self.performance_alerts),
                "recent_alerts": len([
                    a for a in self.performance_alerts 
                    if a["timestamp"] > datetime.now() - timedelta(hours=1)
                ])
            },
            "recommendations": self._generate_performance_recommendations(metrics_list)
        }
        
        return analysis
    
    def _generate_performance_recommendations(self, metrics_list: List[PerformanceMetrics]) -> List[str]:
        """Generate performance optimization recommendations"""
        
        recommendations = []
        
        if not metrics_list:
            return recommendations
        
        avg_load_time = sum(m.page_load_time for m in metrics_list) / len(metrics_list)
        avg_memory = sum(m.memory_usage for m in metrics_list) / len(metrics_list)
        
        # Load time recommendations
        if avg_load_time > 10:
            recommendations.append("Consider increasing performance level to AGGRESSIVE or MAXIMUM")
            recommendations.append("Enable more aggressive resource blocking")
        elif avg_load_time > 5:
            recommendations.append("Enable image optimization to reduce load times")
            recommendations.append("Consider blocking non-essential resources")
        
        # Memory recommendations
        if avg_memory > self.config.memory_limit_mb * 0.7:
            recommendations.append("Reduce max_concurrent_tabs to conserve memory")
            recommendations.append("Enable aggressive cache discard")
            recommendations.append("Consider using memory_efficient performance profile")
        
        # General recommendations
        if self.optimization_stats["blocked_resources"] < 10:
            recommendations.append("Enable resource blocking to improve performance")
        
        if len(self.performance_alerts) > 10:
            recommendations.append("Review and adjust performance thresholds")
        
        return recommendations
    
    async def auto_optimize_session(self, session_id: str) -> Dict[str, Any]:
        """Automatically optimize browser session based on performance data"""
        
        if session_id not in self.session_metrics:
            return {"error": "No performance data for session"}
        
        metrics = self.session_metrics[session_id]
        recent_metrics = [m for m in metrics if m.timestamp > datetime.now() - timedelta(minutes=5)]
        
        if not recent_metrics:
            return {"message": "No recent performance data"}
        
        optimization_changes = []
        
        # Analyze recent performance
        avg_load_time = sum(m.page_load_time for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        
        # Auto-adjust based on performance
        if avg_load_time > 10 and self.config.performance_level != PerformanceLevel.MAXIMUM:
            self.config.performance_level = PerformanceLevel.MAXIMUM
            optimization_changes.append("Increased performance level to MAXIMUM")
        
        if avg_memory > self.config.memory_limit_mb * 0.8:
            self.config.max_concurrent_tabs = max(1, self.config.max_concurrent_tabs - 1)
            optimization_changes.append(f"Reduced max concurrent tabs to {self.config.max_concurrent_tabs}")
        
        if avg_cpu > 80:
            self.config.enable_javascript_optimization = True
            optimization_changes.append("Enabled aggressive JavaScript optimization")
        
        console.print(f"[green]🔧 Auto-optimization applied {len(optimization_changes)} changes for session {session_id}[/green]")
        
        return {
            "session_id": session_id,
            "optimization_changes": optimization_changes,
            "new_config": {
                "performance_level": self.config.performance_level.value,
                "max_concurrent_tabs": self.config.max_concurrent_tabs,
                "enable_javascript_optimization": self.config.enable_javascript_optimization
            }
        }
    
    def apply_performance_profile(self, profile_name: str) -> bool:
        """Apply predefined performance profile"""
        
        if profile_name not in self.performance_profiles:
            console.print(f"[red]❌ Unknown performance profile: {profile_name}[/red]")
            return False
        
        self.config = self.performance_profiles[profile_name]
        console.print(f"[green]✅ Applied performance profile: {profile_name}[/green]")
        return True
    
    def add_custom_blocking_rule(self, rule: ResourceBlockingRule):
        """Add custom resource blocking rule"""
        
        self.resource_blocking_rules.append(rule)
        self.resource_blocking_rules.sort(key=lambda r: r.priority, reverse=True)
        
        console.print(f"[green]🚫 Added blocking rule for {rule.resource_type}: {rule.pattern}[/green]")
    
    async def cleanup_performance_data(self):
        """Clean up old performance data"""
        
        cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
        
        # Clean metrics history
        self.metrics_history = deque([
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time
        ], maxlen=10000)
        
        # Clean session metrics
        for session_id in list(self.session_metrics.keys()):
            self.session_metrics[session_id] = [
                m for m in self.session_metrics[session_id]
                if m.timestamp > cutoff_time
            ]
            
            if not self.session_metrics[session_id]:
                del self.session_metrics[session_id]
        
        # Clean alerts
        self.performance_alerts = [
            a for a in self.performance_alerts
            if a["timestamp"] > cutoff_time
        ]
        
        console.print(f"[green]🧹 Cleaned performance data older than {self.metrics_retention_hours} hours[/green]")
    
    async def export_performance_report(self, output_path: str) -> str:
        """Export comprehensive performance report"""
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "performance_level": self.config.performance_level.value,
                "memory_limit_mb": self.config.memory_limit_mb,
                "max_concurrent_tabs": self.config.max_concurrent_tabs,
                "timeout_seconds": self.config.timeout_seconds
            },
            "optimization_stats": dict(self.optimization_stats),
            "performance_analysis": self.get_performance_analysis(),
            "active_sessions": len(self.active_sessions),
            "total_alerts": len(self.performance_alerts),
            "blocking_rules": len(self.resource_blocking_rules)
        }
        
        try:
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            console.print(f"[green]📊 Performance report exported to {output_path}[/green]")
            return output_path
            
        except Exception as e:
            console.print(f"[red]❌ Failed to export report: {e}[/red]")
            return ""

# Performance optimization presets
PERFORMANCE_PRESETS = {
    "speed_demon": OptimizationConfig(
        performance_level=PerformanceLevel.MAXIMUM,
        enable_resource_blocking=True,
        enable_image_optimization=True,
        enable_javascript_optimization=True,
        enable_css_optimization=True,
        max_concurrent_tabs=2,
        memory_limit_mb=1024,
        timeout_seconds=10
    ),
    "memory_saver": OptimizationConfig(
        performance_level=PerformanceLevel.AGGRESSIVE,
        enable_resource_blocking=True,
        max_concurrent_tabs=3,
        memory_limit_mb=512,
        timeout_seconds=15
    ),
    "balanced_power": OptimizationConfig(
        performance_level=PerformanceLevel.BALANCED,
        enable_caching=True,
        max_concurrent_tabs=5,
        memory_limit_mb=2048,
        timeout_seconds=30
    )
}