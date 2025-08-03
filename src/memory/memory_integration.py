#!/usr/bin/env python3
"""
Memory Integration Layer for CrewAI Agent Memory Systems
Provides seamless integration between memory systems and CrewAI agents
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

try:
    from rich.console import Console
    console = Console()
except ImportError:
    class MockConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = MockConsole()

from .memory_manager import EnhancedMemoryManager, MemoryType, MemoryEntry
from .agent_memory import AgentMemorySystem, SharedMemoryPool
from .memory_storage import (
    VectorMemoryStorage, 
    ConversationMemory, 
    EntityMemory, 
    LongTermMemory
)

class MemoryIntegrationManager:
    """
    Central integration manager for all memory systems
    Coordinates between different memory types and storage backends
    """
    
    def __init__(
        self,
        storage_directory: str = "memory_data",
        enable_vector_storage: bool = True,
        enable_conversation_memory: bool = True,
        enable_entity_memory: bool = True,
        enable_longterm_memory: bool = True
    ):
        self.storage_directory = storage_directory
        self.storage_backends = {}
        self.agent_memory_systems = {}
        self.shared_pools = {}
        
        # Initialize storage backends
        self._initialize_storage_backends(
            enable_vector_storage,
            enable_conversation_memory,
            enable_entity_memory,
            enable_longterm_memory
        )
        
        # Initialize core memory manager
        self.memory_manager = EnhancedMemoryManager(
            storage_path=f"{storage_directory}/core_memory.json"
        )
        
        console.print(f"[green]🧠 Memory Integration Manager initialized[/green]")
        console.print(f"[cyan]   • Storage directory: {storage_directory}[/cyan]")
        console.print(f"[cyan]   • Active backends: {list(self.storage_backends.keys())}[/cyan]")
    
    def _initialize_storage_backends(
        self,
        enable_vector: bool,
        enable_conversation: bool,
        enable_entity: bool,
        enable_longterm: bool
    ):
        """Initialize storage backends based on configuration"""
        
        if enable_vector:
            self.storage_backends['vector'] = VectorMemoryStorage(
                f"{self.storage_directory}/vector_memory.db"
            )
        
        if enable_conversation:
            self.storage_backends['conversation'] = ConversationMemory(
                f"{self.storage_directory}/conversation_memory.db"
            )
        
        if enable_entity:
            self.storage_backends['entity'] = EntityMemory(
                f"{self.storage_directory}/entity_memory.db"
            )
        
        if enable_longterm:
            self.storage_backends['longterm'] = LongTermMemory(
                f"{self.storage_directory}/longterm_memory.db"
            )
    
    async def create_agent_memory_system(
        self,
        agent_id: str,
        agent_name: str = "",
        memory_capacity: int = 1000,
        enable_shared_memory: bool = True
    ) -> AgentMemorySystem:
        """Create a memory system for a specific agent"""
        
        if agent_id in self.agent_memory_systems:
            console.print(f"[yellow]⚠️ Agent memory system already exists for {agent_id}[/yellow]")
            return self.agent_memory_systems[agent_id]
        
        # Create agent memory system
        agent_memory = AgentMemorySystem(
            agent_id=agent_id,
            agent_name=agent_name or agent_id,
            memory_manager=self.memory_manager,
            memory_capacity=memory_capacity
        )
        
        self.agent_memory_systems[agent_id] = agent_memory
        
        # Create shared memory pool if enabled
        if enable_shared_memory:
            shared_pool = SharedMemoryPool(
                pool_id=f"shared_{agent_id}",
                memory_manager=self.memory_manager
            )
            self.shared_pools[agent_id] = shared_pool
            agent_memory.shared_pool = shared_pool
        
        console.print(f"[green]✅ Agent memory system created for {agent_id}[/green]")
        return agent_memory
    
    async def get_agent_memory_system(self, agent_id: str) -> Optional[AgentMemorySystem]:
        """Get existing agent memory system"""
        return self.agent_memory_systems.get(agent_id)
    
    async def store_memory_across_backends(
        self,
        memory: MemoryEntry,
        use_vector_storage: bool = True,
        use_specialized_storage: bool = True
    ) -> Dict[str, bool]:
        """Store memory across multiple storage backends"""
        
        results = {}
        
        # Store in core memory manager
        core_result = await self.memory_manager.store_memory(memory)
        results['core'] = core_result
        
        # Store in vector storage
        if use_vector_storage and 'vector' in self.storage_backends:
            vector_result = await self.storage_backends['vector'].store(memory)
            results['vector'] = vector_result
        
        # Store in specialized storage based on memory type
        if use_specialized_storage:
            if memory.memory_type == MemoryType.CONVERSATION and 'conversation' in self.storage_backends:
                conv_result = await self.storage_backends['conversation'].store(memory)
                results['conversation'] = conv_result
            
            elif memory.memory_type == MemoryType.ENTITY and 'entity' in self.storage_backends:
                entity_result = await self.storage_backends['entity'].store(memory)
                results['entity'] = entity_result
            
            elif memory.memory_type in [MemoryType.KNOWLEDGE, MemoryType.EXPERIENCE] and 'longterm' in self.storage_backends:
                lt_result = await self.storage_backends['longterm'].store(memory)
                results['longterm'] = lt_result
        
        return results
    
    async def search_across_backends(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 20,
        use_semantic_search: bool = True
    ) -> Dict[str, List[MemoryEntry]]:
        """Search for memories across all storage backends"""
        
        results = {}
        
        # Search in core memory manager
        core_memories = await self.memory_manager.search_memories(
            query, memory_type, limit
        )
        results['core'] = core_memories
        
        # Search in vector storage with semantic search
        if use_semantic_search and 'vector' in self.storage_backends:
            vector_memories = await self.storage_backends['vector'].search(
                query, memory_type, limit
            )
            results['vector'] = vector_memories
        
        # Search in specialized storages
        if memory_type == MemoryType.CONVERSATION and 'conversation' in self.storage_backends:
            conv_memories = await self.storage_backends['conversation'].search(
                query, memory_type, limit
            )
            results['conversation'] = conv_memories
        
        elif memory_type == MemoryType.ENTITY and 'entity' in self.storage_backends:
            entity_memories = await self.storage_backends['entity'].search(
                query, memory_type, limit
            )
            results['entity'] = entity_memories
        
        elif memory_type in [MemoryType.KNOWLEDGE, MemoryType.EXPERIENCE] and 'longterm' in self.storage_backends:
            lt_memories = await self.storage_backends['longterm'].search(
                query, memory_type, limit
            )
            results['longterm'] = lt_memories
        
        return results
    
    async def get_conversation_history(
        self,
        conversation_id: str,
        agent_id: Optional[str] = None,
        limit: int = 50
    ) -> List[MemoryEntry]:
        """Get conversation history from specialized storage"""
        
        if 'conversation' not in self.storage_backends:
            console.print("[yellow]⚠️ Conversation storage not available[/yellow]")
            return []
        
        return await self.storage_backends['conversation'].get_conversation_history(
            conversation_id, agent_id, limit
        )
    
    async def get_entity_information(
        self,
        entity_name: str,
        agent_id: Optional[str] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Get entity information from specialized storage"""
        
        if 'entity' not in self.storage_backends:
            console.print("[yellow]⚠️ Entity storage not available[/yellow]")
            return []
        
        return await self.storage_backends['entity'].get_entity_info(
            entity_name, agent_id, limit
        )
    
    async def consolidate_memories(
        self,
        similarity_threshold: float = 0.8,
        max_consolidations: int = 100
    ) -> Dict[str, int]:
        """Consolidate memories across all systems"""
        
        results = {}
        
        # Consolidate in core memory manager
        core_consolidated = await self.memory_manager.consolidate_memories(
            similarity_threshold, max_consolidations
        )
        results['core'] = core_consolidated
        
        # Consolidate in agent memory systems
        agent_consolidated = 0
        for agent_id, agent_memory in self.agent_memory_systems.items():
            agent_count = await agent_memory.consolidate_memories(
                similarity_threshold, max_consolidations // len(self.agent_memory_systems)
            )
            agent_consolidated += agent_count
        results['agents'] = agent_consolidated
        
        console.print(f"[green]✅ Memory consolidation completed[/green]")
        console.print(f"[cyan]   • Core: {core_consolidated} consolidations[/cyan]")
        console.print(f"[cyan]   • Agents: {agent_consolidated} consolidations[/cyan]")
        
        return results
    
    async def cleanup_expired_memories(self) -> Dict[str, int]:
        """Clean up expired memories across all storage backends"""
        
        results = {}
        
        # Cleanup core memory manager
        core_cleaned = await self.memory_manager.cleanup_expired_memories()
        results['core'] = core_cleaned
        
        # Cleanup storage backends
        for backend_name, backend in self.storage_backends.items():
            backend_cleaned = await backend.cleanup_expired()
            results[backend_name] = backend_cleaned
        
        # Cleanup agent memory systems
        agent_cleaned = 0
        for agent_memory in self.agent_memory_systems.values():
            agent_count = await agent_memory.cleanup_old_memories()
            agent_cleaned += agent_count
        results['agents'] = agent_cleaned
        
        total_cleaned = sum(results.values())
        console.print(f"[green]🧹 Memory cleanup completed: {total_cleaned} memories removed[/green]")
        
        return results
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'storage_backends': list(self.storage_backends.keys()),
            'active_agents': len(self.agent_memory_systems),
            'shared_pools': len(self.shared_pools)
        }
        
        # Core memory manager stats
        core_stats = await self.memory_manager.get_memory_statistics()
        stats['core_memory'] = core_stats
        
        # Agent memory stats
        agent_stats = {}
        for agent_id, agent_memory in self.agent_memory_systems.items():
            agent_stats[agent_id] = await agent_memory.get_memory_summary()
        stats['agent_memories'] = agent_stats
        
        # Shared pool stats
        pool_stats = {}
        for pool_id, pool in self.shared_pools.items():
            pool_stats[pool_id] = await pool.get_pool_statistics()
        stats['shared_pools'] = pool_stats
        
        return stats
    
    async def export_memories(
        self,
        export_path: str,
        agent_id: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        include_metadata: bool = True
    ) -> bool:
        """Export memories to external format"""
        
        try:
            import json
            from pathlib import Path
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'agent_id': agent_id,
                'memory_type': memory_type.value if memory_type else None,
                'memories': []
            }
            
            # Get memories from core manager
            if agent_id and agent_id in self.agent_memory_systems:
                agent_memory = self.agent_memory_systems[agent_id]
                memories = await agent_memory.get_all_memories(memory_type)
            else:
                memories = await self.memory_manager.get_all_memories(memory_type)
            
            # Convert memories to exportable format
            for memory in memories:
                memory_dict = {
                    'id': memory.id,
                    'type': memory.memory_type.value,
                    'content': memory.content,
                    'timestamp': memory.timestamp.isoformat(),
                    'importance': memory.importance,
                    'tags': memory.tags
                }
                
                if include_metadata:
                    memory_dict['metadata'] = memory.metadata
                    memory_dict['access_count'] = memory.access_count
                    memory_dict['last_accessed'] = memory.last_accessed.isoformat() if memory.last_accessed else None
                
                export_data['memories'].append(memory_dict)
            
            # Write to file
            Path(export_path).parent.mkdir(parents=True, exist_ok=True)
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            console.print(f"[green]✅ Exported {len(export_data['memories'])} memories to {export_path}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to export memories: {e}[/red]")
            return False
    
    async def import_memories(
        self,
        import_path: str,
        agent_id: Optional[str] = None,
        overwrite_existing: bool = False
    ) -> bool:
        """Import memories from external format"""
        
        try:
            import json
            
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            imported_count = 0
            
            for memory_dict in import_data.get('memories', []):
                # Create memory entry
                memory = MemoryEntry(
                    id=memory_dict['id'],
                    memory_type=MemoryType(memory_dict['type']),
                    content=memory_dict['content'],
                    metadata=memory_dict.get('metadata', {}),
                    timestamp=datetime.fromisoformat(memory_dict['timestamp']),
                    importance=memory_dict['importance'],
                    tags=memory_dict.get('tags', [])
                )
                
                # Store memory
                if agent_id and agent_id in self.agent_memory_systems:
                    agent_memory = self.agent_memory_systems[agent_id]
                    success = await agent_memory.store_memory(memory.content, memory.memory_type, memory.importance)
                else:
                    success = await self.memory_manager.store_memory(memory)
                
                if success:
                    imported_count += 1
            
            console.print(f"[green]✅ Imported {imported_count} memories from {import_path}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to import memories: {e}[/red]")
            return False

class MemoryMiddleware:
    """
    Middleware for integrating memory systems with CrewAI agents
    Provides transparent memory functionality to existing agents
    """
    
    def __init__(self, integration_manager: MemoryIntegrationManager):
        self.integration_manager = integration_manager
        self.active_contexts = {}
        
        console.print("[green]🔗 Memory Middleware initialized[/green]")
    
    async def wrap_agent_with_memory(
        self,
        agent,
        agent_id: Optional[str] = None,
        memory_capacity: int = 1000,
        auto_store_conversations: bool = True,
        auto_extract_entities: bool = True
    ):
        """Wrap an existing CrewAI agent with memory capabilities"""
        
        if not agent_id:
            agent_id = getattr(agent, 'id', f"agent_{id(agent)}")
        
        # Create agent memory system
        agent_memory = await self.integration_manager.create_agent_memory_system(
            agent_id=agent_id,
            agent_name=getattr(agent, 'role', agent_id),
            memory_capacity=memory_capacity
        )
        
        # Store reference for middleware operations
        self.active_contexts[agent_id] = {
            'agent': agent,
            'memory_system': agent_memory,
            'auto_store_conversations': auto_store_conversations,
            'auto_extract_entities': auto_extract_entities
        }
        
        console.print(f"[green]🤖 Agent {agent_id} wrapped with memory capabilities[/green]")
        return agent_memory
    
    async def process_agent_interaction(
        self,
        agent_id: str,
        interaction_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Process agent interaction and store relevant memories"""
        
        if agent_id not in self.active_contexts:
            console.print(f"[yellow]⚠️ No memory context for agent {agent_id}[/yellow]")
            return False
        
        context = self.active_contexts[agent_id]
        agent_memory = context['memory_system']
        
        # Store conversation memory
        if context['auto_store_conversations'] and interaction_type in ['input', 'output', 'thought']:
            await agent_memory.store_conversation_memory(
                content=content,
                role=interaction_type,
                metadata=metadata or {}
            )
        
        # Extract and store entity information
        if context['auto_extract_entities'] and len(content) > 50:
            # Simple entity extraction (can be enhanced with NLP)
            entities = self._extract_simple_entities(content)
            for entity_name, entity_type in entities:
                await agent_memory.store_entity_memory(
                    entity_name=entity_name,
                    content=f"Mentioned in {interaction_type}: {content[:100]}...",
                    entity_type=entity_type
                )
        
        return True
    
    def _extract_simple_entities(self, text: str) -> List[tuple]:
        """Simple entity extraction (placeholder for more sophisticated NLP)"""
        
        entities = []
        
        # Simple patterns for common entities
        import re
        
        # URLs
        urls = re.findall(r'https?://[^\s]+', text)
        for url in urls:
            entities.append((url, 'url'))
        
        # Email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        for email in emails:
            entities.append((email, 'email'))
        
        # Simple capitalized words (potential names/places)
        names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for name in names[:5]:  # Limit to first 5 to avoid noise
            if len(name) > 3:  # Filter short words
                entities.append((name, 'name'))
        
        return entities
    
    async def get_agent_context(
        self,
        agent_id: str,
        query: str,
        context_size: int = 5
    ) -> Dict[str, Any]:
        """Get relevant context for an agent based on query"""
        
        if agent_id not in self.active_contexts:
            return {}
        
        agent_memory = self.active_contexts[agent_id]['memory_system']
        
        # Get relevant memories
        relevant_memories = await agent_memory.get_relevant_memories(
            query, limit=context_size
        )
        
        # Get recent conversation history
        recent_conversations = await agent_memory.get_recent_conversations(
            limit=context_size
        )
        
        return {
            'relevant_memories': [
                {
                    'content': mem.content,
                    'type': mem.memory_type.value,
                    'importance': mem.importance,
                    'timestamp': mem.timestamp.isoformat()
                }
                for mem in relevant_memories
            ],
            'recent_conversations': [
                {
                    'content': conv.content,
                    'role': conv.metadata.get('role', 'unknown'),
                    'timestamp': conv.timestamp.isoformat()
                }
                for conv in recent_conversations
            ]
        }