#!/usr/bin/env python3
"""
Agent Memory System for CrewAI Integration
Provides memory-enabled agents with context retention capabilities
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

try:
    from crewai import Agent
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    # Mock Agent class for development
    class Agent:
        def __init__(self, **kwargs):
            pass

try:
    from rich.console import Console
    console = Console()
except ImportError:
    class MockConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = MockConsole()

from .memory_manager import (
    EnhancedMemoryManager, 
    MemoryType, 
    MemoryEntry, 
    MemoryContext, 
    MemoryRetrieval
)

@dataclass
class AgentMemoryConfig:
    """Configuration for agent memory capabilities"""
    enable_conversation_memory: bool = True
    enable_task_memory: bool = True
    enable_entity_memory: bool = True
    enable_knowledge_memory: bool = True
    
    max_context_window: int = 20
    memory_importance_threshold: float = 0.3
    auto_consolidation: bool = True
    consolidation_threshold: int = 100
    
    # Memory retention settings
    conversation_ttl_hours: Optional[int] = 24
    task_ttl_hours: Optional[int] = 168  # 1 week
    entity_ttl_hours: Optional[int] = None  # Permanent
    knowledge_ttl_hours: Optional[int] = None  # Permanent

class AgentMemorySystem:
    """
    Agent Memory System for CrewAI Integration
    
    Provides comprehensive memory capabilities for CrewAI agents including:
    - Conversation memory for dialogue context
    - Task memory for work history
    - Entity memory for people/concepts
    - Knowledge memory for learned facts
    - Automatic memory consolidation
    - Context-aware retrieval
    """
    
    def __init__(
        self,
        agent_id: str,
        memory_manager: Optional[EnhancedMemoryManager] = None,
        config: Optional[AgentMemoryConfig] = None,
    ):
        self.agent_id = agent_id
        self.config = config or AgentMemoryConfig()
        
        # Initialize memory manager
        if memory_manager:
            self.memory_manager = memory_manager
        else:
            self.memory_manager = EnhancedMemoryManager()
        
        # Current session tracking
        self.current_session_id = str(uuid.uuid4())
        self.current_task_id: Optional[str] = None
        self.current_conversation_id: Optional[str] = None
        
        # Memory stats
        self.stats = {
            "memories_stored": 0,
            "memories_retrieved": 0,
            "conversations_tracked": 0,
            "tasks_completed": 0,
            "entities_learned": 0,
        }
        
        console.print(f"[green]🧠 Agent Memory System initialized for agent: {agent_id}[/green]")
        console.print(f"[cyan]   • Session ID: {self.current_session_id}[/cyan]")
        console.print(f"[cyan]   • Configuration: {self.config}[/cyan]")

    async def start_conversation(self, conversation_id: Optional[str] = None) -> str:
        """Start a new conversation and return conversation ID"""
        
        self.current_conversation_id = conversation_id or f"conv_{int(time.time())}"
        self.stats["conversations_tracked"] += 1
        
        # Store conversation start event
        if self.config.enable_conversation_memory:
            await self._store_memory(
                content=f"Started conversation: {self.current_conversation_id}",
                memory_type=MemoryType.CONVERSATION,
                importance=0.3,
                tags=["conversation_start"],
                ttl_hours=self.config.conversation_ttl_hours,
            )
        
        console.print(f"[cyan]💬 Started conversation: {self.current_conversation_id}[/cyan]")
        return self.current_conversation_id

    async def add_conversation_turn(
        self,
        role: str,
        content: str,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a conversation turn to memory"""
        
        if not self.config.enable_conversation_memory:
            return ""
        
        # Format conversation content
        conversation_content = f"[{role}]: {content}"
        
        # Prepare metadata
        turn_metadata = {
            "role": role,
            "turn_timestamp": datetime.now().isoformat(),
            **(metadata or {})
        }
        
        memory_id = await self._store_memory(
            content=conversation_content,
            memory_type=MemoryType.CONVERSATION,
            importance=importance,
            tags=["conversation", role.lower()],
            metadata=turn_metadata,
            ttl_hours=self.config.conversation_ttl_hours,
        )
        
        console.print(f"[blue]💬 Stored conversation turn: [{role}] {content[:50]}...[/blue]")
        return memory_id

    async def start_task(self, task_description: str, task_id: Optional[str] = None) -> str:
        """Start a new task and return task ID"""
        
        self.current_task_id = task_id or f"task_{int(time.time())}"
        self.stats["tasks_completed"] += 1
        
        # Store task start event
        if self.config.enable_task_memory:
            await self._store_memory(
                content=f"Started task: {task_description}",
                memory_type=MemoryType.TASK_RESULT,
                importance=0.7,
                tags=["task_start", "task"],
                metadata={"task_description": task_description},
                ttl_hours=self.config.task_ttl_hours,
            )
        
        console.print(f"[yellow]📋 Started task: {self.current_task_id} - {task_description}[/yellow]")
        return self.current_task_id

    async def complete_task(
        self,
        result: str,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Complete the current task and store result"""
        
        if not self.config.enable_task_memory or not self.current_task_id:
            return ""
        
        # Format task completion content
        status = "completed successfully" if success else "failed"
        task_content = f"Task {self.current_task_id} {status}: {result}"
        
        # Prepare metadata
        completion_metadata = {
            "task_id": self.current_task_id,
            "success": success,
            "completion_timestamp": datetime.now().isoformat(),
            **(metadata or {})
        }
        
        memory_id = await self._store_memory(
            content=task_content,
            memory_type=MemoryType.TASK_RESULT,
            importance=0.8 if success else 0.6,
            tags=["task_completion", "task", "success" if success else "failure"],
            metadata=completion_metadata,
            ttl_hours=self.config.task_ttl_hours,
        )
        
        console.print(f"[green]✅ Completed task: {self.current_task_id}[/green]")
        self.current_task_id = None
        return memory_id

    async def learn_entity(
        self,
        entity_name: str,
        entity_info: str,
        entity_type: str = "general",
        importance: float = 0.6,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Learn information about an entity (person, concept, etc.)"""
        
        if not self.config.enable_entity_memory:
            return ""
        
        # Format entity content
        entity_content = f"Entity '{entity_name}' ({entity_type}): {entity_info}"
        
        # Prepare metadata
        entity_metadata = {
            "entity_name": entity_name,
            "entity_type": entity_type,
            "learned_timestamp": datetime.now().isoformat(),
            **(metadata or {})
        }
        
        memory_id = await self._store_memory(
            content=entity_content,
            memory_type=MemoryType.ENTITY,
            importance=importance,
            tags=["entity", entity_type, entity_name.lower()],
            metadata=entity_metadata,
            ttl_hours=self.config.entity_ttl_hours,
        )
        
        self.stats["entities_learned"] += 1
        console.print(f"[magenta]👤 Learned about entity: {entity_name}[/magenta]")
        return memory_id

    async def store_knowledge(
        self,
        knowledge: str,
        topic: str = "general",
        importance: float = 0.7,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store general knowledge or facts"""
        
        if not self.config.enable_knowledge_memory:
            return ""
        
        # Format knowledge content
        knowledge_content = f"Knowledge about {topic}: {knowledge}"
        
        # Prepare metadata
        knowledge_metadata = {
            "topic": topic,
            "knowledge_timestamp": datetime.now().isoformat(),
            **(metadata or {})
        }
        
        memory_id = await self._store_memory(
            content=knowledge_content,
            memory_type=MemoryType.KNOWLEDGE,
            importance=importance,
            tags=["knowledge", topic.lower()],
            metadata=knowledge_metadata,
            ttl_hours=self.config.knowledge_ttl_hours,
        )
        
        console.print(f"[cyan]🧠 Stored knowledge: {topic}[/cyan]")
        return memory_id

    async def get_context(
        self,
        query: str = "",
        context_types: Optional[List[MemoryType]] = None,
        limit: Optional[int] = None,
    ) -> List[MemoryEntry]:
        """Get relevant context for current situation"""
        
        # Use config limit if not specified
        if limit is None:
            limit = self.config.max_context_window
        
        # Create memory context
        context = MemoryContext(
            agent_id=self.agent_id,
            session_id=self.current_session_id,
            task_id=self.current_task_id,
            conversation_id=self.current_conversation_id,
            context_window=limit,
            relevance_threshold=self.config.memory_importance_threshold,
        )
        
        # Retrieve relevant memories
        retrieval = await self.memory_manager.retrieve_memories(
            query=query,
            context=context,
            memory_types=context_types,
            limit=limit,
        )
        
        self.stats["memories_retrieved"] += len(retrieval.entries)
        
        console.print(f"[cyan]🔍 Retrieved {len(retrieval.entries)} context memories[/cyan]")
        return retrieval.entries

    async def get_conversation_history(self, limit: int = 20) -> List[MemoryEntry]:
        """Get recent conversation history"""
        
        context = MemoryContext(
            agent_id=self.agent_id,
            session_id=self.current_session_id,
            conversation_id=self.current_conversation_id,
        )
        
        history = await self.memory_manager.get_conversation_history(context, limit)
        console.print(f"[blue]💬 Retrieved {len(history)} conversation entries[/blue]")
        return history

    async def get_entity_info(self, entity_name: str, limit: int = 10) -> List[MemoryEntry]:
        """Get information about a specific entity"""
        
        context = MemoryContext(
            agent_id=self.agent_id,
            session_id=self.current_session_id,
        )
        
        entity_memories = await self.memory_manager.get_entity_memories(
            entity_name, context, limit
        )
        
        console.print(f"[magenta]👤 Retrieved {len(entity_memories)} memories about {entity_name}[/magenta]")
        return entity_memories

    async def consolidate_memories(self) -> Dict[str, Any]:
        """Trigger memory consolidation if enabled and threshold is met"""
        
        if not self.config.auto_consolidation:
            return {"message": "Auto-consolidation disabled"}
        
        result = await self.memory_manager.consolidate_memories(
            agent_id=self.agent_id,
            consolidation_threshold=self.config.consolidation_threshold,
        )
        
        console.print(f"[yellow]🔄 Memory consolidation result: {result}[/yellow]")
        return result

    async def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of the agent's memory state"""
        
        # Get overall memory stats
        memory_stats = self.memory_manager.get_memory_stats()
        
        # Get agent-specific stats
        agent_memory_count = len(self.memory_manager.agent_memories.get(self.agent_id, []))
        
        summary = {
            "agent_id": self.agent_id,
            "current_session": self.current_session_id,
            "current_task": self.current_task_id,
            "current_conversation": self.current_conversation_id,
            "agent_memory_count": agent_memory_count,
            "agent_stats": self.stats,
            "system_stats": memory_stats,
            "config": self.config,
        }
        
        return summary

    # Private methods
    
    async def _store_memory(
        self,
        content: str,
        memory_type: MemoryType,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_hours: Optional[int] = None,
    ) -> str:
        """Store a memory entry"""
        
        # Create memory context
        context = MemoryContext(
            agent_id=self.agent_id,
            session_id=self.current_session_id,
            task_id=self.current_task_id,
            conversation_id=self.current_conversation_id,
        )
        
        # Store memory
        memory_id = await self.memory_manager.store_memory(
            content=content,
            memory_type=memory_type,
            context=context,
            importance=importance,
            tags=tags,
            metadata=metadata,
            ttl_hours=ttl_hours,
        )
        
        self.stats["memories_stored"] += 1
        return memory_id

class MemoryEnabledAgent(Agent):
    """
    CrewAI Agent with integrated memory capabilities
    
    Extends the standard CrewAI Agent with automatic memory management
    for conversations, tasks, entities, and knowledge.
    """
    
    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        memory_config: Optional[AgentMemoryConfig] = None,
        memory_manager: Optional[EnhancedMemoryManager] = None,
        agent_id: Optional[str] = None,
        **kwargs
    ):
        # Initialize CrewAI Agent
        super().__init__(role=role, goal=goal, backstory=backstory, **kwargs)
        
        # Initialize memory system
        self.agent_id = agent_id or f"agent_{int(time.time())}"
        self.memory_system = AgentMemorySystem(
            agent_id=self.agent_id,
            memory_manager=memory_manager,
            config=memory_config,
        )
        
        console.print(f"[green]🤖 Memory-enabled agent created: {role}[/green]")
        console.print(f"[cyan]   • Agent ID: {self.agent_id}[/cyan]")

    async def remember_conversation(self, role: str, content: str, importance: float = 0.5) -> str:
        """Remember a conversation turn"""
        return await self.memory_system.add_conversation_turn(role, content, importance)

    async def remember_task_start(self, task_description: str) -> str:
        """Remember starting a task"""
        return await self.memory_system.start_task(task_description)

    async def remember_task_completion(self, result: str, success: bool = True) -> str:
        """Remember completing a task"""
        return await self.memory_system.complete_task(result, success)

    async def learn_about(self, entity_name: str, info: str, entity_type: str = "general") -> str:
        """Learn about an entity"""
        return await self.memory_system.learn_entity(entity_name, info, entity_type)

    async def remember_fact(self, knowledge: str, topic: str = "general") -> str:
        """Remember a fact or piece of knowledge"""
        return await self.memory_system.store_knowledge(knowledge, topic)

    async def recall_context(self, query: str = "", limit: int = 10) -> List[MemoryEntry]:
        """Recall relevant context"""
        return await self.memory_system.get_context(query, limit=limit)

    async def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory summary"""
        return await self.memory_system.get_memory_summary()

class SharedMemoryPool:
    """
    Shared Memory Pool for Multi-Agent Coordination
    
    Allows multiple agents to share certain types of memories
    for better coordination and knowledge sharing.
    """
    
    def __init__(
        self,
        pool_id: str,
        memory_manager: Optional[EnhancedMemoryManager] = None,
        shared_memory_types: Optional[List[MemoryType]] = None,
    ):
        self.pool_id = pool_id
        self.memory_manager = memory_manager or EnhancedMemoryManager()
        
        # Default shared memory types
        self.shared_memory_types = shared_memory_types or [
            MemoryType.KNOWLEDGE,
            MemoryType.ENTITY,
        ]
        
        # Track participating agents
        self.agents: Dict[str, AgentMemorySystem] = {}
        
        console.print(f"[green]🤝 Shared Memory Pool created: {pool_id}[/green]")
        console.print(f"[cyan]   • Shared types: {[t.value for t in self.shared_memory_types]}[/cyan]")

    def add_agent(self, agent_memory_system: AgentMemorySystem) -> None:
        """Add an agent to the shared memory pool"""
        
        self.agents[agent_memory_system.agent_id] = agent_memory_system
        console.print(f"[cyan]👥 Added agent {agent_memory_system.agent_id} to shared pool[/cyan]")

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from the shared memory pool"""
        
        if agent_id in self.agents:
            del self.agents[agent_id]
            console.print(f"[yellow]👥 Removed agent {agent_id} from shared pool[/yellow]")

    async def share_knowledge(
        self,
        source_agent_id: str,
        knowledge: str,
        topic: str = "shared",
        importance: float = 0.7,
    ) -> List[str]:
        """Share knowledge from one agent to all others in the pool"""
        
        if source_agent_id not in self.agents:
            return []
        
        memory_ids = []
        
        # Share with all other agents
        for agent_id, agent_memory in self.agents.items():
            if agent_id != source_agent_id:
                memory_id = await agent_memory.store_knowledge(
                    knowledge=f"[Shared from {source_agent_id}] {knowledge}",
                    topic=topic,
                    importance=importance,
                    metadata={"source_agent": source_agent_id, "shared": True},
                )
                memory_ids.append(memory_id)
        
        console.print(f"[green]🤝 Shared knowledge from {source_agent_id} to {len(memory_ids)} agents[/green]")
        return memory_ids

    async def share_entity_info(
        self,
        source_agent_id: str,
        entity_name: str,
        entity_info: str,
        entity_type: str = "shared",
        importance: float = 0.6,
    ) -> List[str]:
        """Share entity information from one agent to all others in the pool"""
        
        if source_agent_id not in self.agents:
            return []
        
        memory_ids = []
        
        # Share with all other agents
        for agent_id, agent_memory in self.agents.items():
            if agent_id != source_agent_id:
                memory_id = await agent_memory.learn_entity(
                    entity_name=entity_name,
                    entity_info=f"[Shared from {source_agent_id}] {entity_info}",
                    entity_type=entity_type,
                    importance=importance,
                    metadata={"source_agent": source_agent_id, "shared": True},
                )
                memory_ids.append(memory_id)
        
        console.print(f"[magenta]🤝 Shared entity '{entity_name}' from {source_agent_id} to {len(memory_ids)} agents[/magenta]")
        return memory_ids

    async def get_shared_context(
        self,
        query: str,
        requesting_agent_id: str,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Get shared context from all agents in the pool"""
        
        all_memories = []
        
        for agent_id, agent_memory in self.agents.items():
            if agent_id != requesting_agent_id:
                # Get memories from other agents
                memories = await agent_memory.get_context(
                    query=query,
                    context_types=self.shared_memory_types,
                    limit=limit // len(self.agents),
                )
                all_memories.extend(memories)
        
        # Sort by importance and recency
        all_memories.sort(
            key=lambda m: (m.importance, m.timestamp),
            reverse=True
        )
        
        console.print(f"[cyan]🤝 Retrieved {len(all_memories[:limit])} shared context memories[/cyan]")
        return all_memories[:limit]

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics about the shared memory pool"""
        
        return {
            "pool_id": self.pool_id,
            "agent_count": len(self.agents),
            "agents": list(self.agents.keys()),
            "shared_memory_types": [t.value for t in self.shared_memory_types],
        }