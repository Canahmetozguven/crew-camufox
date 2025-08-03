#!/usr/bin/env python3
"""
CrewAI Monitoring and Observability System
Advanced monitoring capabilities for CrewAI agents, tasks, and workflows
"""

import asyncio
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    console = Console()
except ImportError:
    class MockConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = MockConsole()
    Table = None
    Panel = None
    Progress = None
    SpinnerColumn = None
    TextColumn = None
    BarColumn = None

@dataclass
class AgentMetrics:
    """Metrics for individual agent performance"""
    agent_id: str
    agent_name: str = ""
    role: str = ""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_task_duration: float = 0.0
    total_execution_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    last_activity: Optional[datetime] = None
    error_count: int = 0
    success_rate: float = 0.0
    performance_score: float = 0.0
    status: str = "idle"
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskMetrics:
    """Metrics for task execution"""
    task_id: str
    task_description: str = ""
    agent_id: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: float = 0.0
    status: str = "pending"
    priority: int = 0
    result_quality: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    error_messages: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    output_size: int = 0
    tools_used: List[str] = field(default_factory=list)

@dataclass
class WorkflowMetrics:
    """Metrics for workflow execution"""
    workflow_id: str
    workflow_name: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration: float = 0.0
    agent_count: int = 0
    task_count: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    success_rate: float = 0.0
    throughput: float = 0.0  # tasks per minute
    bottlenecks: List[str] = field(default_factory=list)
    critical_path: List[str] = field(default_factory=list)
    resource_efficiency: float = 0.0

@dataclass
class SystemMetrics:
    """Overall system performance metrics"""
    timestamp: datetime
    total_agents: int = 0
    active_agents: int = 0
    total_workflows: int = 0
    active_workflows: int = 0
    total_tasks: int = 0
    tasks_per_minute: float = 0.0
    avg_response_time: float = 0.0
    system_cpu_percent: float = 0.0
    system_memory_mb: float = 0.0
    disk_usage_mb: float = 0.0
    network_io_mb: float = 0.0
    error_rate: float = 0.0
    uptime_seconds: float = 0.0

class CrewAIMonitor:
    """
    Comprehensive monitoring system for CrewAI operations
    """
    
    def __init__(
        self,
        enable_real_time: bool = True,
        metrics_retention_hours: int = 24,
        alert_thresholds: Optional[Dict[str, float]] = None,
        export_directory: str = "monitoring_data"
    ):
        self.enable_real_time = enable_real_time
        self.metrics_retention_hours = metrics_retention_hours
        self.export_directory = Path(export_directory)
        self.export_directory.mkdir(exist_ok=True)
        
        # Metrics storage
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.task_metrics: Dict[str, TaskMetrics] = {}
        self.workflow_metrics: Dict[str, WorkflowMetrics] = {}
        self.system_metrics_history: deque = deque(maxlen=1000)
        
        # Performance tracking
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.alert_history: List[Dict[str, Any]] = []
        self.event_log: deque = deque(maxlen=1000)
        
        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'cpu_usage': 80.0,
            'memory_usage': 80.0,
            'error_rate': 10.0,
            'response_time': 30.0,
            'task_failure_rate': 20.0
        }
        
        # Monitoring state
        self.start_time = datetime.now()
        self.monitoring_active = False
        self.monitoring_task = None
        self.alert_callbacks: List[Callable] = []
        
        console.print(f"[green]📊 CrewAI Monitor initialized[/green]")
        console.print(f"[cyan]   • Real-time monitoring: {enable_real_time}[/cyan]")
        console.print(f"[cyan]   • Retention period: {metrics_retention_hours} hours[/cyan]")
        console.print(f"[cyan]   • Export directory: {export_directory}[/cyan]")
    
    async def start_monitoring(self) -> None:
        """Start the monitoring system"""
        
        if self.monitoring_active:
            console.print("[yellow]⚠️ Monitoring already active[/yellow]")
            return
        
        self.monitoring_active = True
        
        if self.enable_real_time:
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        self._log_event("system", "Monitoring started")
        console.print("[green]✅ CrewAI monitoring started[/green]")
    
    async def stop_monitoring(self) -> None:
        """Stop the monitoring system"""
        
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self._log_event("system", "Monitoring stopped")
        console.print("[green]🛑 CrewAI monitoring stopped[/green]")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for real-time metrics collection"""
        
        while self.monitoring_active:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Update performance calculations
                await self._update_performance_metrics()
                
                # Check alert conditions
                await self._check_alerts()
                
                # Clean up old metrics
                await self._cleanup_old_metrics()
                
                # Wait before next collection
                await asyncio.sleep(5.0)  # 5-second intervals
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                console.print(f"[red]❌ Monitoring loop error: {e}[/red]")
                await asyncio.sleep(1.0)
    
    async def track_agent_start(
        self,
        agent_id: str,
        agent_name: str = "",
        role: str = ""
    ) -> None:
        """Track when an agent starts"""
        
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = AgentMetrics(
                agent_id=agent_id,
                agent_name=agent_name,
                role=role,
                last_activity=datetime.now()
            )
        
        self.agent_metrics[agent_id].status = "active"
        self.agent_metrics[agent_id].last_activity = datetime.now()
        
        self._log_event("agent", f"Agent {agent_id} started", {"agent_name": agent_name, "role": role})
    
    async def track_agent_stop(self, agent_id: str) -> None:
        """Track when an agent stops"""
        
        if agent_id in self.agent_metrics:
            self.agent_metrics[agent_id].status = "stopped"
            self.agent_metrics[agent_id].last_activity = datetime.now()
        
        self._log_event("agent", f"Agent {agent_id} stopped")
    
    async def track_task_start(
        self,
        task_id: str,
        agent_id: str,
        task_description: str = "",
        priority: int = 0
    ) -> None:
        """Track when a task starts execution"""
        
        task_metrics = TaskMetrics(
            task_id=task_id,
            task_description=task_description,
            agent_id=agent_id,
            start_time=datetime.now(),
            status="running",
            priority=priority
        )
        
        self.task_metrics[task_id] = task_metrics
        
        # Update agent metrics
        if agent_id in self.agent_metrics:
            self.agent_metrics[agent_id].total_tasks += 1
            self.agent_metrics[agent_id].last_activity = datetime.now()
            self.agent_metrics[agent_id].status = "busy"
        
        self._log_event("task", f"Task {task_id} started", {
            "agent_id": agent_id,
            "description": task_description,
            "priority": priority
        })
    
    async def track_task_completion(
        self,
        task_id: str,
        success: bool = True,
        result_quality: float = 1.0,
        error_message: str = "",
        tools_used: List[str] = None,
        output_size: int = 0
    ) -> None:
        """Track when a task completes"""
        
        if task_id not in self.task_metrics:
            return
        
        task_metrics = self.task_metrics[task_id]
        task_metrics.end_time = datetime.now()
        task_metrics.duration = (task_metrics.end_time - task_metrics.start_time).total_seconds()
        task_metrics.status = "completed" if success else "failed"
        task_metrics.result_quality = result_quality
        task_metrics.output_size = output_size
        
        if tools_used:
            task_metrics.tools_used = tools_used
        
        if error_message:
            task_metrics.error_messages.append(error_message)
        
        # Update agent metrics
        agent_id = task_metrics.agent_id
        if agent_id in self.agent_metrics:
            agent_metrics = self.agent_metrics[agent_id]
            
            if success:
                agent_metrics.completed_tasks += 1
            else:
                agent_metrics.failed_tasks += 1
                agent_metrics.error_count += 1
            
            # Update average task duration
            total_completed = agent_metrics.completed_tasks + agent_metrics.failed_tasks
            if total_completed > 0:
                agent_metrics.avg_task_duration = (
                    (agent_metrics.avg_task_duration * (total_completed - 1) + task_metrics.duration)
                    / total_completed
                )
                agent_metrics.success_rate = (agent_metrics.completed_tasks / total_completed) * 100
            
            agent_metrics.total_execution_time += task_metrics.duration
            agent_metrics.last_activity = datetime.now()
            agent_metrics.status = "idle"
        
        self._log_event("task", f"Task {task_id} {'completed' if success else 'failed'}", {
            "agent_id": agent_id,
            "duration": task_metrics.duration,
            "success": success,
            "quality": result_quality
        })
    
    async def track_workflow_start(
        self,
        workflow_id: str,
        workflow_name: str = "",
        agent_ids: List[str] = None
    ) -> None:
        """Track when a workflow starts"""
        
        workflow_metrics = WorkflowMetrics(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            start_time=datetime.now(),
            agent_count=len(agent_ids) if agent_ids else 0
        )
        
        self.workflow_metrics[workflow_id] = workflow_metrics
        
        self._log_event("workflow", f"Workflow {workflow_id} started", {
            "name": workflow_name,
            "agents": agent_ids or []
        })
    
    async def track_workflow_completion(
        self,
        workflow_id: str,
        success: bool = True,
        completed_tasks: int = 0,
        failed_tasks: int = 0
    ) -> None:
        """Track when a workflow completes"""
        
        if workflow_id not in self.workflow_metrics:
            return
        
        workflow_metrics = self.workflow_metrics[workflow_id]
        workflow_metrics.end_time = datetime.now()
        workflow_metrics.total_duration = (
            workflow_metrics.end_time - workflow_metrics.start_time
        ).total_seconds()
        workflow_metrics.completed_tasks = completed_tasks
        workflow_metrics.failed_tasks = failed_tasks
        workflow_metrics.task_count = completed_tasks + failed_tasks
        
        if workflow_metrics.task_count > 0:
            workflow_metrics.success_rate = (completed_tasks / workflow_metrics.task_count) * 100
            workflow_metrics.throughput = workflow_metrics.task_count / (workflow_metrics.total_duration / 60)
        
        self._log_event("workflow", f"Workflow {workflow_id} completed", {
            "duration": workflow_metrics.total_duration,
            "success": success,
            "tasks": workflow_metrics.task_count
        })
    
    async def record_error(
        self,
        component: str,
        error_type: str,
        error_message: str,
        context: Dict[str, Any] = None
    ) -> None:
        """Record an error event"""
        
        error_data = {
            "component": component,
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self._log_event("error", f"{component} error: {error_type}", error_data)
        
        # Update error counts
        if component in self.agent_metrics:
            self.agent_metrics[component].error_count += 1
    
    async def _collect_system_metrics(self) -> None:
        """Collect system-level performance metrics"""
        
        try:
            # Basic system metrics (simplified implementation)
            system_metrics = SystemMetrics(
                timestamp=datetime.now(),
                total_agents=len(self.agent_metrics),
                active_agents=len([a for a in self.agent_metrics.values() if a.status == "active"]),
                total_workflows=len(self.workflow_metrics),
                active_workflows=len([w for w in self.workflow_metrics.values() if w.end_time is None]),
                total_tasks=len(self.task_metrics),
                uptime_seconds=(datetime.now() - self.start_time).total_seconds()
            )
            
            # Calculate tasks per minute
            recent_tasks = [
                t for t in self.task_metrics.values()
                if t.start_time and t.start_time > datetime.now() - timedelta(minutes=5)
            ]
            system_metrics.tasks_per_minute = len(recent_tasks) / 5.0
            
            # Calculate average response time
            completed_tasks = [t for t in self.task_metrics.values() if t.end_time]
            if completed_tasks:
                system_metrics.avg_response_time = sum(t.duration for t in completed_tasks) / len(completed_tasks)
            
            # Calculate error rate
            recent_errors = len([
                e for e in self.event_log
                if e.get('event_type') == 'error' and 
                   datetime.fromisoformat(e.get('timestamp', '1970-01-01')) > datetime.now() - timedelta(minutes=5)
            ])
            system_metrics.error_rate = (recent_errors / max(len(recent_tasks), 1)) * 100
            
            self.system_metrics_history.append(system_metrics)
            
        except Exception as e:
            console.print(f"[yellow]⚠️ Failed to collect system metrics: {e}[/yellow]")
    
    async def _update_performance_metrics(self) -> None:
        """Update performance calculations and trends"""
        
        # Update agent performance scores
        for agent_id, metrics in self.agent_metrics.items():
            if metrics.total_tasks > 0:
                # Calculate performance score based on multiple factors
                success_factor = metrics.success_rate / 100
                speed_factor = max(0, 1 - (metrics.avg_task_duration / 60))  # Prefer faster tasks
                activity_factor = 1.0 if metrics.status == "active" else 0.8
                
                metrics.performance_score = (
                    success_factor * 0.5 +
                    speed_factor * 0.3 +
                    activity_factor * 0.2
                ) * 100
            
            # Store performance history
            self.performance_history[f"agent_{agent_id}_performance"].append(metrics.performance_score)
    
    async def _check_alerts(self) -> None:
        """Check for alert conditions and trigger notifications"""
        
        current_time = datetime.now()
        
        # Check system-level alerts
        if self.system_metrics_history:
            latest_metrics = self.system_metrics_history[-1]
            
            # High error rate alert
            if latest_metrics.error_rate > self.alert_thresholds.get('error_rate', 10.0):
                await self._trigger_alert(
                    "high_error_rate",
                    f"Error rate is {latest_metrics.error_rate:.1f}%",
                    {"current_rate": latest_metrics.error_rate}
                )
            
            # High response time alert
            if latest_metrics.avg_response_time > self.alert_thresholds.get('response_time', 30.0):
                await self._trigger_alert(
                    "high_response_time",
                    f"Average response time is {latest_metrics.avg_response_time:.1f}s",
                    {"current_time": latest_metrics.avg_response_time}
                )
        
        # Check agent-level alerts
        for agent_id, metrics in self.agent_metrics.items():
            # Task failure rate alert
            if metrics.total_tasks > 5:  # Only alert after sufficient data
                failure_rate = (metrics.failed_tasks / metrics.total_tasks) * 100
                if failure_rate > self.alert_thresholds.get('task_failure_rate', 20.0):
                    await self._trigger_alert(
                        "high_task_failure_rate",
                        f"Agent {agent_id} has {failure_rate:.1f}% task failure rate",
                        {"agent_id": agent_id, "failure_rate": failure_rate}
                    )
            
            # Agent inactivity alert
            if metrics.last_activity:
                inactive_minutes = (current_time - metrics.last_activity).total_seconds() / 60
                if inactive_minutes > 30 and metrics.status == "active":  # 30 minutes
                    await self._trigger_alert(
                        "agent_inactive",
                        f"Agent {agent_id} has been inactive for {inactive_minutes:.1f} minutes",
                        {"agent_id": agent_id, "inactive_minutes": inactive_minutes}
                    )
    
    async def _trigger_alert(
        self,
        alert_type: str,
        message: str,
        context: Dict[str, Any]
    ) -> None:
        """Trigger an alert and notify registered callbacks"""
        
        alert = {
            "type": alert_type,
            "message": message,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "severity": self._get_alert_severity(alert_type)
        }
        
        self.alert_history.append(alert)
        
        # Log the alert
        self._log_event("alert", message, context)
        
        # Display alert
        console.print(f"[red]🚨 ALERT: {message}[/red]")
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                console.print(f"[yellow]⚠️ Alert callback failed: {e}[/yellow]")
    
    def _get_alert_severity(self, alert_type: str) -> str:
        """Determine alert severity level"""
        
        high_severity = ["high_error_rate", "system_failure", "agent_crash"]
        medium_severity = ["high_response_time", "high_task_failure_rate", "resource_exhaustion"]
        
        if alert_type in high_severity:
            return "high"
        elif alert_type in medium_severity:
            return "medium"
        else:
            return "low"
    
    async def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics based on retention policy"""
        
        cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
        
        # Clean up task metrics
        expired_tasks = [
            task_id for task_id, metrics in self.task_metrics.items()
            if metrics.start_time and metrics.start_time < cutoff_time
        ]
        
        for task_id in expired_tasks:
            del self.task_metrics[task_id]
        
        # Clean up workflow metrics
        expired_workflows = [
            workflow_id for workflow_id, metrics in self.workflow_metrics.items()
            if metrics.start_time and metrics.start_time < cutoff_time
        ]
        
        for workflow_id in expired_workflows:
            del self.workflow_metrics[workflow_id]
        
        # Clean up event log
        self.event_log = deque([
            event for event in self.event_log
            if datetime.fromisoformat(event.get('timestamp', '1970-01-01')) >= cutoff_time
        ], maxlen=1000)
    
    def _log_event(
        self,
        event_type: str,
        message: str,
        context: Dict[str, Any] = None
    ) -> None:
        """Log an event to the event history"""
        
        event = {
            "event_type": event_type,
            "message": message,
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.event_log.append(event)
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Add a callback function for alert notifications"""
        self.alert_callbacks.append(callback)
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        
        # Agent summary
        agent_summary = {
            "total": len(self.agent_metrics),
            "active": len([a for a in self.agent_metrics.values() if a.status == "active"]),
            "idle": len([a for a in self.agent_metrics.values() if a.status == "idle"]),
            "avg_performance": sum(a.performance_score for a in self.agent_metrics.values()) / max(len(self.agent_metrics), 1)
        }
        
        # Task summary
        task_summary = {
            "total": len(self.task_metrics),
            "completed": len([t for t in self.task_metrics.values() if t.status == "completed"]),
            "failed": len([t for t in self.task_metrics.values() if t.status == "failed"]),
            "running": len([t for t in self.task_metrics.values() if t.status == "running"]),
            "avg_duration": sum(t.duration for t in self.task_metrics.values() if t.duration > 0) / max(len([t for t in self.task_metrics.values() if t.duration > 0]), 1)
        }
        
        # Workflow summary
        workflow_summary = {
            "total": len(self.workflow_metrics),
            "active": len([w for w in self.workflow_metrics.values() if w.end_time is None]),
            "completed": len([w for w in self.workflow_metrics.values() if w.end_time is not None]),
            "avg_success_rate": sum(w.success_rate for w in self.workflow_metrics.values()) / max(len(self.workflow_metrics), 1)
        }
        
        # Recent alerts
        recent_alerts = [
            alert for alert in self.alert_history[-10:]  # Last 10 alerts
        ]
        
        # Performance trends
        performance_trends = {
            metric_name: list(history)[-20:]  # Last 20 data points
            for metric_name, history in self.performance_history.items()
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "agent_summary": agent_summary,
            "task_summary": task_summary,
            "workflow_summary": workflow_summary,
            "recent_alerts": recent_alerts,
            "performance_trends": performance_trends,
            "system_metrics": self.system_metrics_history[-1].__dict__ if self.system_metrics_history else {}
        }
    
    async def export_metrics(self, export_path: str = None) -> str:
        """Export all metrics to a JSON file"""
        
        if not export_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = self.export_directory / f"metrics_export_{timestamp}.json"
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "monitoring_period": {
                "start": self.start_time.isoformat(),
                "duration_hours": (datetime.now() - self.start_time).total_seconds() / 3600
            },
            "agent_metrics": {
                agent_id: {
                    "agent_id": metrics.agent_id,
                    "agent_name": metrics.agent_name,
                    "role": metrics.role,
                    "total_tasks": metrics.total_tasks,
                    "completed_tasks": metrics.completed_tasks,
                    "failed_tasks": metrics.failed_tasks,
                    "avg_task_duration": metrics.avg_task_duration,
                    "success_rate": metrics.success_rate,
                    "performance_score": metrics.performance_score,
                    "status": metrics.status
                }
                for agent_id, metrics in self.agent_metrics.items()
            },
            "task_metrics": {
                task_id: {
                    "task_id": metrics.task_id,
                    "agent_id": metrics.agent_id,
                    "duration": metrics.duration,
                    "status": metrics.status,
                    "result_quality": metrics.result_quality,
                    "tools_used": metrics.tools_used,
                    "start_time": metrics.start_time.isoformat() if metrics.start_time else None,
                    "end_time": metrics.end_time.isoformat() if metrics.end_time else None
                }
                for task_id, metrics in self.task_metrics.items()
            },
            "workflow_metrics": {
                workflow_id: {
                    "workflow_id": metrics.workflow_id,
                    "workflow_name": metrics.workflow_name,
                    "total_duration": metrics.total_duration,
                    "task_count": metrics.task_count,
                    "success_rate": metrics.success_rate,
                    "throughput": metrics.throughput,
                    "start_time": metrics.start_time.isoformat() if metrics.start_time else None,
                    "end_time": metrics.end_time.isoformat() if metrics.end_time else None
                }
                for workflow_id, metrics in self.workflow_metrics.items()
            },
            "alert_history": self.alert_history,
            "event_log": list(self.event_log)
        }
        
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            console.print(f"[green]✅ Metrics exported to {export_path}[/green]")
            return str(export_path)
            
        except Exception as e:
            console.print(f"[red]❌ Failed to export metrics: {e}[/red]")
            return ""
    
    def display_live_dashboard(self) -> None:
        """Display a live dashboard of current metrics"""
        
        if not Table:
            console.print("Rich library not available for dashboard display")
            return
        
        try:
            # Agent metrics table
            agent_table = Table(title="Agent Metrics")
            agent_table.add_column("Agent ID", style="cyan")
            agent_table.add_column("Role", style="magenta")
            agent_table.add_column("Status", style="green")
            agent_table.add_column("Tasks", justify="right")
            agent_table.add_column("Success Rate", justify="right")
            agent_table.add_column("Performance", justify="right")
            
            for metrics in self.agent_metrics.values():
                status_color = "green" if metrics.status == "active" else "yellow"
                agent_table.add_row(
                    metrics.agent_id[:20],
                    metrics.role[:15],
                    f"[{status_color}]{metrics.status}[/{status_color}]",
                    str(metrics.total_tasks),
                    f"{metrics.success_rate:.1f}%",
                    f"{metrics.performance_score:.1f}"
                )
            
            console.print(agent_table)
            
            # System overview
            if self.system_metrics_history:
                latest = self.system_metrics_history[-1]
                overview_table = Table(title="System Overview")
                overview_table.add_column("Metric", style="cyan")
                overview_table.add_column("Value", style="green")
                
                overview_table.add_row("Active Agents", str(latest.active_agents))
                overview_table.add_row("Active Workflows", str(latest.active_workflows))
                overview_table.add_row("Tasks/Min", f"{latest.tasks_per_minute:.1f}")
                overview_table.add_row("Avg Response Time", f"{latest.avg_response_time:.1f}s")
                overview_table.add_row("Error Rate", f"{latest.error_rate:.1f}%")
                overview_table.add_row("Uptime", f"{latest.uptime_seconds/3600:.1f}h")
                
                console.print(overview_table)
            
        except Exception as e:
            console.print(f"[red]❌ Dashboard display error: {e}[/red]")