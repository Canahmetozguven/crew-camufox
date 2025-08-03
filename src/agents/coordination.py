#!/usr/bin/env python3
"""
Agent Coordination and Communication System
Advanced coordination mechanisms for hierarchical agents
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Union
from pathlib import Path
from collections import defaultdict, deque

class MessageType(Enum):
    """Types of inter-agent messages"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    COORDINATION_REQUEST = "coordination_request"
    RESOURCE_REQUEST = "resource_request"
    KNOWLEDGE_SHARE = "knowledge_share"
    ESCALATION = "escalation"
    BROADCAST = "broadcast"

class CoordinationStrategy(Enum):
    """Coordination strategies"""
    HIERARCHICAL = "hierarchical"
    COLLABORATIVE = "collaborative"
    COMPETITIVE = "competitive"
    CONSENSUS = "consensus"

@dataclass
class AgentMessage:
    """Inter-agent message structure"""
    message_id: str
    sender_id: str
    recipient_id: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 1
    requires_response: bool = False
    response_deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CoordinationContext:
    """Context for agent coordination"""
    coordination_id: str
    strategy: CoordinationStrategy
    participating_agents: List[str]
    objective: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    shared_resources: List[str] = field(default_factory=list)
    coordination_state: str = "active"
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None

class AgentCoordinator:
    """
    Advanced agent coordination system for managing multi-agent interactions
    """
    
    def __init__(
        self,
        enable_real_time_coordination: bool = True,
        max_message_queue_size: int = 1000,
        message_timeout: int = 300,  # 5 minutes
        data_directory: str = "coordination_data",
        max_history_size: int = 5000,  # Limit history growth
        cleanup_interval: int = 3600  # Cleanup every hour
    ):
        self.enable_real_time_coordination = enable_real_time_coordination
        self.max_message_queue_size = max_message_queue_size
        self.message_timeout = message_timeout
        self.max_history_size = max_history_size
        self.cleanup_interval = cleanup_interval
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(exist_ok=True)
        
        # Message management with size limits
        self.message_queues: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_message_queue_size))
        self.pending_responses: Dict[str, AgentMessage] = {}
        self.message_history: List[AgentMessage] = []
        
        # Coordination contexts
        self.active_coordinations: Dict[str, CoordinationContext] = {}
        self.coordination_history: List[CoordinationContext] = []
        
        # Agent status tracking
        self.agent_statuses: Dict[str, Dict[str, Any]] = {}
        self.agent_capabilities: Dict[str, Set[str]] = {}
        self.agent_workload: Dict[str, float] = defaultdict(float)
        
        # Communication patterns with size limits
        self.communication_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))  # Limit response time history
        
        # Event handling
        self.message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self.coordination_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Cleanup tracking
        self._last_cleanup = datetime.now()
        self._shutdown_event = asyncio.Event()
        self._cleanup_task = None
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup coordination logging"""
        
        log_file = self.data_directory / "coordination.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('AgentCoordinator')
        
        # Start background cleanup task
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of old data"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup task error: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old messages and coordination data"""
        now = datetime.now()
        cutoff_time = now - timedelta(seconds=self.message_timeout * 2)
        
        # Clean up old message history
        if len(self.message_history) > self.max_history_size:
            self.message_history = self.message_history[-self.max_history_size//2:]
            self.logger.info(f"Cleaned message history, kept {len(self.message_history)} messages")
        
        # Clean up old coordination history
        if len(self.coordination_history) > self.max_history_size:
            self.coordination_history = self.coordination_history[-self.max_history_size//2:]
            self.logger.info(f"Cleaned coordination history, kept {len(self.coordination_history)} contexts")
        
        # Clean up expired pending responses
        expired_responses = [
            msg_id for msg_id, msg in self.pending_responses.items()
            if msg.timestamp < cutoff_time
        ]
        for msg_id in expired_responses:
            del self.pending_responses[msg_id]
        
        if expired_responses:
            self.logger.info(f"Cleaned {len(expired_responses)} expired pending responses")
        
        # Clean up finished coordination contexts
        finished_contexts = [
            coord_id for coord_id, context in self.active_coordinations.items()
            if context.coordination_state in ['completed', 'failed', 'cancelled'] and
            context.created_at < cutoff_time
        ]
        for coord_id in finished_contexts:
            self.coordination_history.append(self.active_coordinations[coord_id])
            del self.active_coordinations[coord_id]
        
        if finished_contexts:
            self.logger.info(f"Moved {len(finished_contexts)} finished contexts to history")
        
        self._last_cleanup = now
    
    async def shutdown(self):
        """Properly shutdown the coordinator"""
        self.logger.info("Shutting down AgentCoordinator...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Final cleanup
        await self._cleanup_old_data()
        
        # Clear all data structures
        self.message_queues.clear()
        self.pending_responses.clear()
        self.active_coordinations.clear()
        self.agent_statuses.clear()
        self.agent_capabilities.clear()
        self.agent_workload.clear()
        
        self.logger.info("AgentCoordinator shutdown complete")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        if hasattr(self, '_cleanup_task') and self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
    
    async def send_message(
        self,
        sender_id: str,
        recipient_id: str,
        message_type: MessageType,
        content: Dict[str, Any],
        priority: int = 1,
        requires_response: bool = False,
        response_timeout: Optional[int] = None
    ) -> str:
        """Send message between agents with validation and error handling"""
        
        # Input validation
        if not sender_id or not isinstance(sender_id, str):
            raise ValueError("sender_id must be a non-empty string")
        if not recipient_id or not isinstance(recipient_id, str):
            raise ValueError("recipient_id must be a non-empty string")
        if not isinstance(content, dict):
            raise ValueError("content must be a dictionary")
        if priority < 1 or priority > 10:
            raise ValueError("priority must be between 1 and 10")
        
        # Sanitize content to prevent injection attacks
        sanitized_content = self._sanitize_content(content)
        
        message_id = f"msg_{datetime.now().timestamp()}_{sender_id}_{recipient_id}"
        
        try:
            message = AgentMessage(
                message_id=message_id,
                sender_id=sender_id,
                recipient_id=recipient_id,
                message_type=message_type,
                content=sanitized_content,
                priority=priority,
                requires_response=requires_response,
                response_deadline=datetime.now() + timedelta(seconds=response_timeout) if response_timeout else None
            )
            
            # Add to recipient's queue
            self.message_queues[recipient_id].append(message)
            
            # Maintain queue size
            if len(self.message_queues[recipient_id]) > self.max_message_queue_size:
                self.message_queues[recipient_id].popleft()
            
            # Track pending response if required
            if requires_response:
                self.pending_responses[message_id] = message
            
            # Update communication patterns
            self.communication_patterns[sender_id][recipient_id] += 1
            
            # Add to message history
            self.message_history.append(message)
            
            # Trigger message handlers
            await self._trigger_message_handlers(message)
            
            self.logger.info(f"Message sent: {sender_id} -> {recipient_id} ({message_type.value})")
            return message_id
            
        except Exception as e:
            self.logger.error(f"Failed to send message {message_id}: {e}")
            raise
    
    def _sanitize_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize message content to prevent injection attacks"""
        sanitized = {}
        
        for key, value in content.items():
            # Sanitize string keys and values
            if isinstance(key, str):
                # Remove potentially dangerous characters
                clean_key = re.sub(r'[<>"\';]', '', key)
                if clean_key != key:
                    self.logger.warning(f"Sanitized message key: {key} -> {clean_key}")
                key = clean_key
            
            if isinstance(value, str):
                # Remove potentially dangerous characters and limit length
                clean_value = re.sub(r'[<>]', '', value)[:10000]  # Limit to 10KB
                if clean_value != value:
                    self.logger.warning(f"Sanitized message value for key {key}")
                value = clean_value
            elif isinstance(value, dict):
                value = self._sanitize_content(value)  # Recursive sanitization
            elif isinstance(value, list):
                value = [self._sanitize_content(item) if isinstance(item, dict) else item for item in value[:100]]  # Limit list size
            
            sanitized[key] = value
        
        return sanitized
        
        self.logger.info(f"Message sent: {sender_id} -> {recipient_id} ({message_type.value})")
        
        return message_id
    
    async def receive_messages(self, agent_id: str, limit: Optional[int] = None) -> List[AgentMessage]:
        """Receive messages for an agent"""
        
        messages = []
        agent_queue = self.message_queues[agent_id]
        
        count = 0
        while agent_queue and (limit is None or count < limit):
            message = agent_queue.popleft()
            messages.append(message)
            count += 1
        
        # Sort by priority and timestamp
        messages.sort(key=lambda m: (-m.priority, m.timestamp))
        
        return messages
    
    async def send_response(
        self,
        original_message_id: str,
        sender_id: str,
        content: Dict[str, Any]
    ) -> bool:
        """Send response to a message"""
        
        if original_message_id not in self.pending_responses:
            return False
        
        original_message = self.pending_responses[original_message_id]
        
        # Calculate response time
        response_time = (datetime.now() - original_message.timestamp).total_seconds()
        self.response_times[sender_id].append(response_time)
        
        # Keep only recent response times (deque handles this automatically with maxlen)
        
        # Send response
        response_id = await self.send_message(
            sender_id=sender_id,
            recipient_id=original_message.sender_id,
            message_type=MessageType.TASK_RESPONSE,
            content={
                "response_to": original_message_id,
                "response_time": response_time,
                **content
            }
        )
        
        # Remove from pending responses
        del self.pending_responses[original_message_id]
        
        return True
    
    async def broadcast_message(
        self,
        sender_id: str,
        recipients: List[str],
        content: Dict[str, Any],
        message_type: MessageType = MessageType.BROADCAST
    ) -> List[str]:
        """Broadcast message to multiple agents"""
        
        message_ids = []
        
        for recipient_id in recipients:
            message_id = await self.send_message(
                sender_id=sender_id,
                recipient_id=recipient_id,
                message_type=message_type,
                content=content
            )
            message_ids.append(message_id)
        
        return message_ids
    
    def create_coordination_context(
        self,
        coordination_id: str,
        strategy: CoordinationStrategy,
        participating_agents: List[str],
        objective: str,
        constraints: Optional[Dict[str, Any]] = None,
        deadline: Optional[datetime] = None
    ) -> CoordinationContext:
        """Create new coordination context"""
        
        context = CoordinationContext(
            coordination_id=coordination_id,
            strategy=strategy,
            participating_agents=participating_agents,
            objective=objective,
            constraints=constraints or {},
            deadline=deadline
        )
        
        self.active_coordinations[coordination_id] = context
        
        self.logger.info(f"Created coordination context: {coordination_id} with {len(participating_agents)} agents")
        
        return context
    
    async def coordinate_task_execution(
        self,
        coordination_id: str,
        task_data: Dict[str, Any],
        coordination_strategy: CoordinationStrategy = CoordinationStrategy.HIERARCHICAL
    ) -> Dict[str, Any]:
        """Coordinate task execution among agents"""
        
        if coordination_id not in self.active_coordinations:
            return {"error": "Coordination context not found"}
        
        context = self.active_coordinations[coordination_id]
        
        coordination_result = {
            "coordination_id": coordination_id,
            "strategy": coordination_strategy.value,
            "participants": context.participating_agents,
            "started_at": datetime.now().isoformat(),
            "task_assignments": {},
            "coordination_messages": []
        }
        
        if coordination_strategy == CoordinationStrategy.HIERARCHICAL:
            result = await self._coordinate_hierarchical(context, task_data)
        elif coordination_strategy == CoordinationStrategy.COLLABORATIVE:
            result = await self._coordinate_collaborative(context, task_data)
        elif coordination_strategy == CoordinationStrategy.CONSENSUS:
            result = await self._coordinate_consensus(context, task_data)
        else:
            result = await self._coordinate_competitive(context, task_data)
        
        coordination_result.update(result)
        coordination_result["completed_at"] = datetime.now().isoformat()
        
        return coordination_result
    
    async def _coordinate_hierarchical(
        self,
        context: CoordinationContext,
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implement hierarchical coordination strategy"""
        
        # Find the highest-ranking agent (coordinator)
        coordinator = context.participating_agents[0]  # Simplified selection
        subordinates = context.participating_agents[1:]
        
        # Send coordination request to coordinator
        coord_message_id = await self.send_message(
            sender_id="system",
            recipient_id=coordinator,
            message_type=MessageType.COORDINATION_REQUEST,
            content={
                "coordination_type": "hierarchical",
                "role": "coordinator",
                "subordinates": subordinates,
                "task_data": task_data,
                "context": context.__dict__
            },
            requires_response=True,
            response_timeout=self.message_timeout
        )
        
        # Notify subordinates of the coordination
        subordinate_messages = []
        for subordinate in subordinates:
            msg_id = await self.send_message(
                sender_id="system",
                recipient_id=subordinate,
                message_type=MessageType.COORDINATION_REQUEST,
                content={
                    "coordination_type": "hierarchical",
                    "role": "subordinate",
                    "coordinator": coordinator,
                    "task_data": task_data,
                    "context": context.__dict__
                }
            )
            subordinate_messages.append(msg_id)
        
        return {
            "coordinator": coordinator,
            "subordinates": subordinates,
            "coordinator_message": coord_message_id,
            "subordinate_messages": subordinate_messages
        }
    
    async def _coordinate_collaborative(
        self,
        context: CoordinationContext,
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implement collaborative coordination strategy"""
        
        # Send collaboration request to all agents
        collaboration_messages = []
        
        for agent_id in context.participating_agents:
            msg_id = await self.send_message(
                sender_id="system",
                recipient_id=agent_id,
                message_type=MessageType.COORDINATION_REQUEST,
                content={
                    "coordination_type": "collaborative",
                    "participants": context.participating_agents,
                    "task_data": task_data,
                    "context": context.__dict__
                },
                requires_response=True,
                response_timeout=self.message_timeout
            )
            collaboration_messages.append(msg_id)
        
        return {
            "collaboration_messages": collaboration_messages,
            "participants": context.participating_agents
        }
    
    async def _coordinate_consensus(
        self,
        context: CoordinationContext,
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implement consensus-based coordination strategy"""
        
        # Start consensus round
        consensus_round_id = f"consensus_{datetime.now().timestamp()}"
        
        consensus_messages = []
        for agent_id in context.participating_agents:
            msg_id = await self.send_message(
                sender_id="system",
                recipient_id=agent_id,
                message_type=MessageType.COORDINATION_REQUEST,
                content={
                    "coordination_type": "consensus",
                    "consensus_round": consensus_round_id,
                    "participants": context.participating_agents,
                    "task_data": task_data,
                    "voting_deadline": (datetime.now() + timedelta(minutes=5)).isoformat(),
                    "context": context.__dict__
                },
                requires_response=True,
                response_timeout=self.message_timeout
            )
            consensus_messages.append(msg_id)
        
        return {
            "consensus_round_id": consensus_round_id,
            "consensus_messages": consensus_messages,
            "participants": context.participating_agents
        }
    
    async def _coordinate_competitive(
        self,
        context: CoordinationContext,
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Implement competitive coordination strategy"""
        
        # Send competition announcement to all agents
        competition_id = f"comp_{datetime.now().timestamp()}"
        
        competition_messages = []
        for agent_id in context.participating_agents:
            msg_id = await self.send_message(
                sender_id="system",
                recipient_id=agent_id,
                message_type=MessageType.COORDINATION_REQUEST,
                content={
                    "coordination_type": "competitive",
                    "competition_id": competition_id,
                    "competitors": context.participating_agents,
                    "task_data": task_data,
                    "evaluation_criteria": task_data.get("evaluation_criteria", {}),
                    "submission_deadline": (datetime.now() + timedelta(minutes=10)).isoformat(),
                    "context": context.__dict__
                }
            )
            competition_messages.append(msg_id)
        
        return {
            "competition_id": competition_id,
            "competition_messages": competition_messages,
            "competitors": context.participating_agents
        }
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register handler for specific message type"""
        
        self.message_handlers[message_type].append(handler)
    
    async def _trigger_message_handlers(self, message: AgentMessage):
        """Trigger registered message handlers"""
        
        for handler in self.message_handlers[message.message_type]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                self.logger.error(f"Error in message handler: {e}")
    
    def update_agent_status(
        self,
        agent_id: str,
        status_data: Dict[str, Any]
    ):
        """Update agent status information"""
        
        self.agent_statuses[agent_id] = {
            **status_data,
            "last_updated": datetime.now().isoformat()
        }
        
        # Update workload if provided
        if "workload" in status_data:
            self.agent_workload[agent_id] = status_data["workload"]
    
    def get_agent_communication_stats(self, agent_id: str) -> Dict[str, Any]:
        """Get communication statistics for an agent"""
        
        sent_messages = sum(self.communication_patterns[agent_id].values())
        received_messages = sum(
            count for sender_patterns in self.communication_patterns.values()
            for recipient, count in sender_patterns.items()
            if recipient == agent_id
        )
        
        avg_response_time = 0.0
        if agent_id in self.response_times and self.response_times[agent_id]:
            avg_response_time = sum(self.response_times[agent_id]) / len(self.response_times[agent_id])
        
        return {
            "agent_id": agent_id,
            "messages_sent": sent_messages,
            "messages_received": received_messages,
            "average_response_time": avg_response_time,
            "pending_responses": len([
                msg for msg in self.pending_responses.values()
                if msg.recipient_id == agent_id
            ]),
            "queue_length": len(self.message_queues[agent_id]),
            "communication_partners": list(self.communication_patterns[agent_id].keys())
        }
    
    def get_coordination_status(self, coordination_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a coordination context"""
        
        if coordination_id not in self.active_coordinations:
            return None
        
        context = self.active_coordinations[coordination_id]
        
        # Calculate progress
        total_messages = len([
            msg for msg in self.message_history
            if any(agent in [msg.sender_id, msg.recipient_id] 
                  for agent in context.participating_agents)
        ])
        
        return {
            "coordination_id": coordination_id,
            "strategy": context.strategy.value,
            "participants": context.participating_agents,
            "objective": context.objective,
            "state": context.coordination_state,
            "created_at": context.created_at.isoformat(),
            "deadline": context.deadline.isoformat() if context.deadline else None,
            "total_messages": total_messages,
            "constraints": context.constraints
        }
    
    async def finalize_coordination(self, coordination_id: str) -> bool:
        """Finalize and archive coordination context"""
        
        if coordination_id not in self.active_coordinations:
            return False
        
        context = self.active_coordinations[coordination_id]
        context.coordination_state = "completed"
        
        # Move to history
        self.coordination_history.append(context)
        del self.active_coordinations[coordination_id]
        
        # Notify participants
        await self.broadcast_message(
            sender_id="system",
            recipients=context.participating_agents,
            content={
                "coordination_completed": coordination_id,
                "final_status": "completed"
            }
        )
        
        self.logger.info(f"Coordination {coordination_id} finalized")
        
        return True
    
    def export_coordination_report(self, output_path: str) -> str:
        """Export comprehensive coordination report"""
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "active_coordinations": len(self.active_coordinations),
            "total_messages": len(self.message_history),
            "agent_communication_stats": {
                agent_id: self.get_agent_communication_stats(agent_id)
                for agent_id in self.agent_statuses.keys()
            },
            "coordination_contexts": {
                coord_id: self.get_coordination_status(coord_id)
                for coord_id in self.active_coordinations.keys()
            },
            "message_type_distribution": {},
            "average_response_times": {
                agent_id: sum(times) / len(times) if times else 0
                for agent_id, times in self.response_times.items()
            }
        }
        
        # Calculate message type distribution
        type_counts = defaultdict(int)
        for message in self.message_history:
            type_counts[message.message_type.value] += 1
        
        report_data["message_type_distribution"] = dict(type_counts)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to export coordination report: {e}")
            return ""