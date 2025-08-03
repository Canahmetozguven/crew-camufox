#!/usr/bin/env python3
"""
Hierarchical Agent Management System for CrewAI
Advanced agent coordination, hierarchy management, and task delegation
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Union
from pathlib import Path
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

class AgentRole(Enum):
    """Agent roles in the hierarchy"""
    COORDINATOR = "coordinator"
    SUPERVISOR = "supervisor"
    SPECIALIST = "specialist"
    WORKER = "worker"
    ANALYST = "analyst"

class AgentStatus(Enum):
    """Agent status states"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    SUSPENDED = "suspended"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    URGENT = 5

@dataclass
class AgentCapability:
    """Agent capability definition"""
    name: str
    description: str
    proficiency_level: float  # 0.0 to 1.0
    tags: Set[str] = field(default_factory=set)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HierarchicalTask:
    """Task in the hierarchical system"""
    task_id: str
    title: str
    description: str
    priority: TaskPriority
    required_capabilities: List[str]
    assigned_agent: Optional[str] = None
    parent_task: Optional[str] = None
    subtasks: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    progress: float = 0.0
    result: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentProfile:
    """Comprehensive agent profile"""
    agent_id: str
    name: str
    role: AgentRole
    status: AgentStatus
    capabilities: List[AgentCapability]
    supervisor: Optional[str] = None
    subordinates: List[str] = field(default_factory=list)
    current_tasks: List[str] = field(default_factory=list)
    completed_tasks: int = 0
    success_rate: float = 1.0
    average_completion_time: float = 0.0
    specializations: Set[str] = field(default_factory=set)
    max_concurrent_tasks: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

class HierarchicalAgentManager:
    """
    Advanced hierarchical agent management system for CrewAI
    """
    
    def __init__(
        self,
        enable_auto_assignment: bool = True,
        enable_load_balancing: bool = True,
        enable_performance_tracking: bool = True,
        data_directory: str = "agent_management"
    ):
        self.enable_auto_assignment = enable_auto_assignment
        self.enable_load_balancing = enable_load_balancing
        self.enable_performance_tracking = enable_performance_tracking
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(exist_ok=True)
        
        # Agent management
        self.agents: Dict[str, AgentProfile] = {}
        self.agent_hierarchy: Dict[str, List[str]] = defaultdict(list)  # supervisor -> subordinates
        self.reverse_hierarchy: Dict[str, str] = {}  # subordinate -> supervisor
        
        # Task management
        self.tasks: Dict[str, HierarchicalTask] = {}
        self.task_queue: deque = deque()
        self.completed_tasks: List[str] = []
        
        # Coordination and communication
        self.message_queues: Dict[str, deque] = defaultdict(deque)
        self.agent_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Performance tracking
        self.performance_history: Dict[str, List[Dict]] = defaultdict(list)
        self.system_metrics: Dict[str, Any] = {
            "total_agents": 0,
            "active_agents": 0,
            "total_tasks": 0,
            "completed_tasks": 0,
            "average_task_completion_time": 0.0,
            "system_efficiency": 0.0
        }
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        self._setup_logging()
        
        console.print(f"[green]🏗️ Hierarchical Agent Manager initialized[/green]")
        console.print(f"[cyan]   • Auto assignment: {enable_auto_assignment}[/cyan]")
        console.print(f"[cyan]   • Load balancing: {enable_load_balancing}[/cyan]")
        console.print(f"[cyan]   • Performance tracking: {enable_performance_tracking}[/cyan]")
    
    def _setup_logging(self):
        """Setup agent management logging"""
        
        log_file = self.data_directory / "agent_management.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('HierarchicalAgentManager')
    
    def register_agent(
        self,
        agent_id: str,
        name: str,
        role: AgentRole,
        capabilities: List[AgentCapability],
        supervisor: Optional[str] = None,
        specializations: Optional[Set[str]] = None
    ) -> bool:
        """Register a new agent in the hierarchy"""
        
        if agent_id in self.agents:
            console.print(f"[yellow]⚠️ Agent {agent_id} already registered[/yellow]")
            return False
        
        # Create agent profile
        agent_profile = AgentProfile(
            agent_id=agent_id,
            name=name,
            role=role,
            status=AgentStatus.IDLE,
            capabilities=capabilities,
            supervisor=supervisor,
            specializations=specializations or set()
        )
        
        self.agents[agent_id] = agent_profile
        
        # Update hierarchy
        if supervisor and supervisor in self.agents:
            self.agent_hierarchy[supervisor].append(agent_id)
            self.reverse_hierarchy[agent_id] = supervisor
            self.agents[supervisor].subordinates.append(agent_id)
        
        # Update system metrics
        self.system_metrics["total_agents"] += 1
        self.system_metrics["active_agents"] += 1
        
        # Trigger event
        self._trigger_event("agent_registered", {
            "agent_id": agent_id,
            "role": role.value,
            "supervisor": supervisor
        })
        
        console.print(f"[green]👤 Registered agent: {name} ({role.value})[/green]")
        
        return True
    
    def create_task(
        self,
        task_id: str,
        title: str,
        description: str,
        priority: TaskPriority,
        required_capabilities: List[str],
        parent_task: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        deadline: Optional[datetime] = None
    ) -> bool:
        """Create a new hierarchical task"""
        
        if task_id in self.tasks:
            console.print(f"[yellow]⚠️ Task {task_id} already exists[/yellow]")
            return False
        
        task = HierarchicalTask(
            task_id=task_id,
            title=title,
            description=description,
            priority=priority,
            required_capabilities=required_capabilities,
            parent_task=parent_task,
            dependencies=dependencies or [],
            deadline=deadline
        )
        
        self.tasks[task_id] = task
        
        # Update parent task if specified
        if parent_task and parent_task in self.tasks:
            self.tasks[parent_task].subtasks.append(task_id)
        
        # Add to task queue for assignment
        self.task_queue.append(task_id)
        
        # Update system metrics
        self.system_metrics["total_tasks"] += 1
        
        # Auto-assign if enabled
        if self.enable_auto_assignment:
            asyncio.create_task(self._auto_assign_task(task_id))
        
        console.print(f"[green]📋 Created task: {title} (Priority: {priority.name})[/green]")
        
        return True
    
    async def _auto_assign_task(self, task_id: str) -> Optional[str]:
        """Automatically assign task to best available agent"""
        
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        
        # Check dependencies
        if not self._are_dependencies_completed(task_id):
            console.print(f"[yellow]⏳ Task {task_id} waiting for dependencies[/yellow]")
            return None
        
        # Find best agent for the task
        best_agent = self._find_best_agent_for_task(task)
        
        if best_agent:
            return await self.assign_task(task_id, best_agent)
        else:
            console.print(f"[yellow]⚠️ No suitable agent found for task {task_id}[/yellow]")
            return None
    
    def _are_dependencies_completed(self, task_id: str) -> bool:
        """Check if all task dependencies are completed"""
        
        task = self.tasks[task_id]
        
        for dep_id in task.dependencies:
            if dep_id in self.tasks and self.tasks[dep_id].status != "completed":
                return False
        
        return True
    
    def _find_best_agent_for_task(self, task: HierarchicalTask) -> Optional[str]:
        """Find the best agent for a given task"""
        
        suitable_agents = []
        
        for agent_id, agent in self.agents.items():
            # Check if agent is available
            if agent.status != AgentStatus.IDLE:
                continue
            
            # Check if agent has capacity
            if len(agent.current_tasks) >= agent.max_concurrent_tasks:
                continue
            
            # Check capabilities
            agent_capability_names = {cap.name for cap in agent.capabilities}
            required_caps = set(task.required_capabilities)
            
            if not required_caps.issubset(agent_capability_names):
                continue
            
            # Calculate suitability score
            capability_scores = []
            for req_cap in task.required_capabilities:
                for agent_cap in agent.capabilities:
                    if agent_cap.name == req_cap:
                        capability_scores.append(agent_cap.proficiency_level)
                        break
            
            avg_capability_score = sum(capability_scores) / len(capability_scores) if capability_scores else 0
            
            # Factor in success rate and workload
            workload_factor = 1.0 - (len(agent.current_tasks) / agent.max_concurrent_tasks)
            
            suitability_score = (
                avg_capability_score * 0.5 +
                agent.success_rate * 0.3 +
                workload_factor * 0.2
            )
            
            suitable_agents.append((agent_id, suitability_score))
        
        # Sort by suitability score and return best agent
        if suitable_agents:
            suitable_agents.sort(key=lambda x: x[1], reverse=True)
            return suitable_agents[0][0]
        
        return None
    
    async def assign_task(self, task_id: str, agent_id: str) -> Optional[str]:
        """Assign a task to a specific agent"""
        
        if task_id not in self.tasks or agent_id not in self.agents:
            return None
        
        task = self.tasks[task_id]
        agent = self.agents[agent_id]
        
        # Check if agent can handle the task
        if len(agent.current_tasks) >= agent.max_concurrent_tasks:
            console.print(f"[red]❌ Agent {agent_id} at capacity[/red]")
            return None
        
        # Assign task
        task.assigned_agent = agent_id
        task.status = "assigned"
        agent.current_tasks.append(task_id)
        agent.status = AgentStatus.BUSY
        agent.last_active = datetime.now()
        
        # Send task to agent
        await self._send_message_to_agent(agent_id, {
            "type": "task_assignment",
            "task_id": task_id,
            "task": task.__dict__,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trigger event
        self._trigger_event("task_assigned", {
            "task_id": task_id,
            "agent_id": agent_id,
            "priority": task.priority.name
        })
        
        console.print(f"[green]✅ Assigned task {task_id} to agent {agent_id}[/green]")
        
        return agent_id
    
    async def _send_message_to_agent(self, agent_id: str, message: Dict[str, Any]):
        """Send message to an agent"""
        
        self.message_queues[agent_id].append(message)
        
        # Trigger callbacks for the agent
        for callback in self.agent_callbacks[agent_id]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as e:
                self.logger.error(f"Error in agent callback for {agent_id}: {e}")
    
    def register_agent_callback(self, agent_id: str, callback: Callable):
        """Register callback for agent messages"""
        
        self.agent_callbacks[agent_id].append(callback)
    
    async def complete_task(
        self,
        task_id: str,
        result: Any,
        agent_id: Optional[str] = None
    ) -> bool:
        """Mark task as completed"""
        
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if agent_id and agent_id != task.assigned_agent:
            console.print(f"[red]❌ Agent {agent_id} not assigned to task {task_id}[/red]")
            return False
        
        # Update task
        task.status = "completed"
        task.progress = 1.0
        task.result = result
        completion_time = datetime.now()
        task.metadata["completed_at"] = completion_time.isoformat()
        
        # Update agent
        if task.assigned_agent:
            agent = self.agents[task.assigned_agent]
            agent.current_tasks.remove(task_id)
            agent.completed_tasks += 1
            
            # Calculate completion time
            start_time = task.created_at
            completion_duration = (completion_time - start_time).total_seconds()
            
            # Update average completion time
            if agent.average_completion_time == 0:
                agent.average_completion_time = completion_duration
            else:
                agent.average_completion_time = (
                    agent.average_completion_time * 0.8 + completion_duration * 0.2
                )
            
            # Update agent status
            if not agent.current_tasks:
                agent.status = AgentStatus.IDLE
            
            # Track performance
            if self.enable_performance_tracking:
                self._track_agent_performance(task.assigned_agent, task, completion_duration)
        
        # Move to completed tasks
        self.completed_tasks.append(task_id)
        
        # Update system metrics
        self.system_metrics["completed_tasks"] += 1
        
        # Check and auto-assign dependent tasks
        await self._check_dependent_tasks(task_id)
        
        # Trigger event
        self._trigger_event("task_completed", {
            "task_id": task_id,
            "agent_id": task.assigned_agent,
            "completion_time": completion_duration if task.assigned_agent else 0
        })
        
        console.print(f"[green]✅ Task {task_id} completed successfully[/green]")
        
        return True
    
    async def _check_dependent_tasks(self, completed_task_id: str):
        """Check and potentially assign tasks that were waiting for this one"""
        
        for task_id, task in self.tasks.items():
            if (completed_task_id in task.dependencies and 
                task.status == "pending" and 
                self._are_dependencies_completed(task_id)):
                
                if self.enable_auto_assignment:
                    await self._auto_assign_task(task_id)
    
    def _track_agent_performance(self, agent_id: str, task: HierarchicalTask, duration: float):
        """Track agent performance metrics"""
        
        performance_record = {
            "task_id": task.task_id,
            "task_priority": task.priority.name,
            "completion_time": duration,
            "timestamp": datetime.now().isoformat(),
            "required_capabilities": task.required_capabilities
        }
        
        self.performance_history[agent_id].append(performance_record)
        
        # Keep only recent performance records
        if len(self.performance_history[agent_id]) > 100:
            self.performance_history[agent_id] = self.performance_history[agent_id][-100:]
    
    def get_agent_hierarchy(self) -> Dict[str, Any]:
        """Get complete agent hierarchy"""
        
        hierarchy = {}
        
        # Find root agents (no supervisors)
        root_agents = [
            agent_id for agent_id, agent in self.agents.items()
            if agent.supervisor is None
        ]
        
        def build_hierarchy_node(agent_id: str) -> Dict[str, Any]:
            agent = self.agents[agent_id]
            node = {
                "agent_id": agent_id,
                "name": agent.name,
                "role": agent.role.value,
                "status": agent.status.value,
                "current_tasks": len(agent.current_tasks),
                "completed_tasks": agent.completed_tasks,
                "success_rate": agent.success_rate,
                "subordinates": []
            }
            
            # Add subordinates recursively
            for subordinate_id in agent.subordinates:
                node["subordinates"].append(build_hierarchy_node(subordinate_id))
            
            return node
        
        for root_agent in root_agents:
            hierarchy[root_agent] = build_hierarchy_node(root_agent)
        
        return hierarchy
    
    def get_task_overview(self) -> Dict[str, Any]:
        """Get comprehensive task overview"""
        
        task_stats = {
            "total_tasks": len(self.tasks),
            "pending_tasks": 0,
            "assigned_tasks": 0,
            "completed_tasks": len(self.completed_tasks),
            "task_queue_length": len(self.task_queue),
            "tasks_by_priority": defaultdict(int),
            "tasks_by_status": defaultdict(int),
            "average_completion_time": 0.0
        }
        
        completion_times = []
        
        for task in self.tasks.values():
            task_stats["tasks_by_priority"][task.priority.name] += 1
            task_stats["tasks_by_status"][task.status] += 1
            
            if task.status == "pending":
                task_stats["pending_tasks"] += 1
            elif task.status == "assigned":
                task_stats["assigned_tasks"] += 1
            
            # Calculate completion time if completed
            if (task.status == "completed" and 
                "completed_at" in task.metadata):
                start_time = task.created_at
                end_time = datetime.fromisoformat(task.metadata["completed_at"])
                completion_times.append((end_time - start_time).total_seconds())
        
        if completion_times:
            task_stats["average_completion_time"] = sum(completion_times) / len(completion_times)
        
        return dict(task_stats)
    
    def get_agent_performance_report(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed performance report for an agent"""
        
        if agent_id not in self.agents:
            return None
        
        agent = self.agents[agent_id]
        performance_data = self.performance_history.get(agent_id, [])
        
        report = {
            "agent_id": agent_id,
            "name": agent.name,
            "role": agent.role.value,
            "status": agent.status.value,
            "basic_metrics": {
                "completed_tasks": agent.completed_tasks,
                "current_tasks": len(agent.current_tasks),
                "success_rate": agent.success_rate,
                "average_completion_time": agent.average_completion_time,
                "max_concurrent_tasks": agent.max_concurrent_tasks
            },
            "capabilities": [
                {
                    "name": cap.name,
                    "proficiency": cap.proficiency_level,
                    "tags": list(cap.tags)
                }
                for cap in agent.capabilities
            ],
            "hierarchy": {
                "supervisor": agent.supervisor,
                "subordinates": agent.subordinates
            },
            "recent_performance": []
        }
        
        # Analyze recent performance
        if performance_data:
            recent_tasks = performance_data[-10:]  # Last 10 tasks
            
            report["recent_performance"] = {
                "tasks_completed": len(recent_tasks),
                "average_time": sum(t["completion_time"] for t in recent_tasks) / len(recent_tasks),
                "fastest_completion": min(t["completion_time"] for t in recent_tasks),
                "slowest_completion": max(t["completion_time"] for t in recent_tasks),
                "priority_breakdown": defaultdict(int)
            }
            
            for task_data in recent_tasks:
                report["recent_performance"]["priority_breakdown"][task_data["task_priority"]] += 1
        
        return report
    
    def rebalance_workload(self) -> Dict[str, Any]:
        """Rebalance workload across agents"""
        
        if not self.enable_load_balancing:
            return {"message": "Load balancing disabled"}
        
        rebalancing_actions = []
        
        # Find overloaded and underloaded agents
        overloaded_agents = []
        underloaded_agents = []
        
        for agent_id, agent in self.agents.items():
            if agent.status == AgentStatus.OFFLINE:
                continue
            
            current_load = len(agent.current_tasks)
            capacity_ratio = current_load / agent.max_concurrent_tasks
            
            if capacity_ratio > 0.8:  # More than 80% capacity
                overloaded_agents.append((agent_id, current_load, capacity_ratio))
            elif capacity_ratio < 0.3:  # Less than 30% capacity
                underloaded_agents.append((agent_id, current_load, capacity_ratio))
        
        # Sort by load
        overloaded_agents.sort(key=lambda x: x[2], reverse=True)
        underloaded_agents.sort(key=lambda x: x[2])
        
        # Attempt to redistribute tasks
        for overloaded_id, load, ratio in overloaded_agents:
            if not underloaded_agents:
                break
            
            # Find tasks that can be reassigned
            reassignable_tasks = []
            for task_id in self.agents[overloaded_id].current_tasks:
                task = self.tasks[task_id]
                if task.status == "assigned" and task.progress < 0.5:  # Less than 50% progress
                    reassignable_tasks.append(task_id)
            
            # Try to reassign tasks
            for task_id in reassignable_tasks[:2]:  # Max 2 tasks per rebalancing
                task = self.tasks[task_id]
                
                # Find suitable underloaded agent
                for underloaded_id, _, _ in underloaded_agents:
                    if self._can_agent_handle_task(underloaded_id, task):
                        # Reassign task
                        self._reassign_task(task_id, overloaded_id, underloaded_id)
                        rebalancing_actions.append({
                            "action": "reassign",
                            "task_id": task_id,
                            "from_agent": overloaded_id,
                            "to_agent": underloaded_id
                        })
                        break
        
        return {
            "timestamp": datetime.now().isoformat(),
            "actions_taken": len(rebalancing_actions),
            "details": rebalancing_actions,
            "overloaded_agents": len(overloaded_agents),
            "underloaded_agents": len(underloaded_agents)
        }
    
    def _can_agent_handle_task(self, agent_id: str, task: HierarchicalTask) -> bool:
        """Check if agent can handle a specific task"""
        
        agent = self.agents[agent_id]
        
        # Check capacity
        if len(agent.current_tasks) >= agent.max_concurrent_tasks:
            return False
        
        # Check capabilities
        agent_capabilities = {cap.name for cap in agent.capabilities}
        required_capabilities = set(task.required_capabilities)
        
        return required_capabilities.issubset(agent_capabilities)
    
    def _reassign_task(self, task_id: str, from_agent: str, to_agent: str):
        """Reassign task from one agent to another"""
        
        # Remove from old agent
        self.agents[from_agent].current_tasks.remove(task_id)
        if not self.agents[from_agent].current_tasks:
            self.agents[from_agent].status = AgentStatus.IDLE
        
        # Assign to new agent
        self.agents[to_agent].current_tasks.append(task_id)
        self.agents[to_agent].status = AgentStatus.BUSY
        
        # Update task
        self.tasks[task_id].assigned_agent = to_agent
        self.tasks[task_id].metadata["reassigned_at"] = datetime.now().isoformat()
        self.tasks[task_id].metadata["previous_agent"] = from_agent
    
    def _trigger_event(self, event_type: str, data: Dict[str, Any]):
        """Trigger event handlers"""
        
        for handler in self.event_handlers[event_type]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(data))
            except Exception as e:
                self.logger.error(f"Error in event handler for {event_type}: {e}")
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register event handler"""
        
        self.event_handlers[event_type].append(handler)
    
    async def export_management_report(self, output_path: str) -> str:
        """Export comprehensive management report"""
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "system_overview": {
                "total_agents": len(self.agents),
                "active_agents": len([a for a in self.agents.values() if a.status != AgentStatus.OFFLINE]),
                "total_tasks": len(self.tasks),
                "completed_tasks": len(self.completed_tasks),
                "pending_tasks": len([t for t in self.tasks.values() if t.status == "pending"])
            },
            "agent_hierarchy": self.get_agent_hierarchy(),
            "task_overview": self.get_task_overview(),
            "performance_summary": {
                agent_id: self.get_agent_performance_report(agent_id)
                for agent_id in self.agents.keys()
            },
            "system_metrics": self.system_metrics
        }
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            console.print(f"[green]📊 Management report exported to {output_path}[/green]")
            return output_path
            
        except Exception as e:
            console.print(f"[red]❌ Failed to export report: {e}[/red]")
            return ""