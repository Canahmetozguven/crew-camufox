#!/usr/bin/env python3
"""
Enhanced Memory Manager for CrewAI Agent Memory Systems
Provides comprehensive memory capabilities for context retention and intelligent coordination
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from rich.console import Console
    console = Console()
except ImportError:
    class MockConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = MockConsole()

console = Console()

class MemoryType(Enum):
    """Types of memory entries"""
    CONVERSATION = "conversation"
    ENTITY = "entity"
    TASK_RESULT = "task_result"
    KNOWLEDGE = "knowledge"
    EXPERIENCE = "experience"
    CONTEXT = "context"
    METADATA = "metadata"

@dataclass
class MemoryEntry:
    """Individual memory entry with rich metadata"""
    id: str
    memory_type: MemoryType
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    related_entries: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    ttl: Optional[datetime] = None  # Time to live
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['memory_type'] = self.memory_type.value
        data['timestamp'] = self.timestamp.isoformat()
        data['last_accessed'] = self.last_accessed.isoformat() if self.last_accessed else None
        data['ttl'] = self.ttl.isoformat() if self.ttl else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary"""
        data['memory_type'] = MemoryType(data['memory_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['last_accessed'] = datetime.fromisoformat(data['last_accessed']) if data['last_accessed'] else None
        data['ttl'] = datetime.fromisoformat(data['ttl']) if data['ttl'] else None
        return cls(**data)

@dataclass
class MemoryContext:
    """Context for memory operations"""
    agent_id: str
    session_id: str
    task_id: Optional[str] = None
    conversation_id: Optional[str] = None
    context_window: int = 10
    relevance_threshold: float = 0.3
    include_metadata: bool = True

@dataclass
class MemoryRetrieval:
    """Memory retrieval result"""
    entries: List[MemoryEntry]
    total_found: int
    search_time: float
    context: MemoryContext
    similarity_scores: Optional[List[float]] = None

class EnhancedMemoryManager:
    """
    Enhanced Memory Manager for CrewAI Agents
    
    Provides comprehensive memory capabilities including:
    - Multiple memory types (conversation, entity, task, knowledge)
    - Vector similarity search for semantic retrieval
    - Importance-based memory retention
    - Temporal decay and TTL support
    - Memory consolidation and cleanup
    - Cross-agent memory sharing
    """
    
    def __init__(
        self,
        storage_path: str = "memory_storage",
        max_memory_entries: int = 10000,
        cleanup_interval: int = 3600,  # 1 hour
        enable_vector_search: bool = True,
        embedding_model: Optional[str] = None,
    ):
        self.storage_path = Path(storage_path)
        self.max_memory_entries = max_memory_entries
        self.cleanup_interval = cleanup_interval
        self.enable_vector_search = enable_vector_search
        self.embedding_model = embedding_model
        
        # Memory storage
        self.memories: Dict[str, MemoryEntry] = {}
        self.agent_memories: Dict[str, List[str]] = {}  # agent_id -> memory_ids
        self.session_memories: Dict[str, List[str]] = {}  # session_id -> memory_ids
        self.tag_index: Dict[str, List[str]] = {}  # tag -> memory_ids
        
        # Performance tracking
        self.stats = {
            "total_memories": 0,
            "successful_retrievals": 0,
            "failed_retrievals": 0,
            "cleanup_runs": 0,
            "last_cleanup": datetime.now(),
        }
        
        # Initialize storage
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._load_memories()
        
        # Start background cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
        
        console.print(f"[green]✅ Enhanced Memory Manager initialized[/green]")
        console.print(f"[cyan]   • Storage path: {self.storage_path}[/cyan]")
        console.print(f"[cyan]   • Max entries: {self.max_memory_entries}[/cyan]")
        console.print(f"[cyan]   • Vector search: {'Enabled' if self.enable_vector_search else 'Disabled'}[/cyan]")
        console.print(f"[cyan]   • Loaded memories: {len(self.memories)}[/cyan]")

    async def store_memory(
        self,
        content: str,
        memory_type: MemoryType,
        context: MemoryContext,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_hours: Optional[int] = None,
    ) -> str:
        """Store a new memory entry"""
        
        memory_id = str(uuid.uuid4())
        now = datetime.now()
        
        # Calculate TTL
        ttl = now + timedelta(hours=ttl_hours) if ttl_hours else None
        
        # Create memory entry
        memory = MemoryEntry(
            id=memory_id,
            memory_type=memory_type,
            content=content,
            metadata=metadata or {},
            timestamp=now,
            importance=importance,
            tags=tags or [],
            ttl=ttl,
        )
        
        # Add context metadata
        memory.metadata.update({
            "agent_id": context.agent_id,
            "session_id": context.session_id,
            "task_id": context.task_id,
            "conversation_id": context.conversation_id,
        })
        
        # Generate embedding if vector search is enabled
        if self.enable_vector_search:
            memory.embedding = await self._generate_embedding(content)
        
        # Store memory
        self.memories[memory_id] = memory
        
        # Update indices
        self._update_indices(memory)
        
        # Update stats
        self.stats["total_memories"] += 1
        
        # Save to disk
        await self._save_memory(memory)
        
        # Check if we need to clean up old memories
        if len(self.memories) > self.max_memory_entries:
            await self._cleanup_old_memories()
        
        console.print(f"[green]💾 Stored memory: {memory_type.value} - {content[:50]}...[/green]")
        return memory_id

    async def retrieve_memories(
        self,
        query: str,
        context: MemoryContext,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10,
        include_similarity: bool = True,
    ) -> MemoryRetrieval:
        """Retrieve relevant memories based on query and context"""
        
        start_time = time.time()
        
        try:
            # Filter memories by context
            candidate_memories = self._filter_by_context(context, memory_types)
            
            if not candidate_memories:
                return MemoryRetrieval(
                    entries=[],
                    total_found=0,
                    search_time=time.time() - start_time,
                    context=context,
                )
            
            # Perform search
            if self.enable_vector_search and query:
                results = await self._vector_search(query, candidate_memories, limit)
            else:
                results = self._keyword_search(query, candidate_memories, limit)
            
            # Update access tracking
            for memory in results:
                memory.access_count += 1
                memory.last_accessed = datetime.now()
            
            # Calculate similarity scores if requested
            similarity_scores = None
            if include_similarity and self.enable_vector_search:
                query_embedding = await self._generate_embedding(query)
                similarity_scores = [
                    self._calculate_similarity(query_embedding, memory.embedding)
                    for memory in results
                    if memory.embedding is not None
                ]
            
            search_time = time.time() - start_time
            self.stats["successful_retrievals"] += 1
            
            console.print(f"[cyan]🔍 Retrieved {len(results)} memories in {search_time:.2f}s[/cyan]")
            
            return MemoryRetrieval(
                entries=results,
                total_found=len(candidate_memories),
                search_time=search_time,
                context=context,
                similarity_scores=similarity_scores,
            )
            
        except Exception as e:
            self.stats["failed_retrievals"] += 1
            console.print(f"[red]❌ Memory retrieval failed: {e}[/red]")
            return MemoryRetrieval(
                entries=[],
                total_found=0,
                search_time=time.time() - start_time,
                context=context,
            )

    async def get_conversation_history(
        self,
        context: MemoryContext,
        limit: int = 50,
    ) -> List[MemoryEntry]:
        """Get conversation history for a specific context"""
        
        memories = []
        session_memory_ids = self.session_memories.get(context.session_id, [])
        
        for memory_id in session_memory_ids:
            if memory_id in self.memories:
                memory = self.memories[memory_id]
                if memory.memory_type == MemoryType.CONVERSATION:
                    memories.append(memory)
        
        # Sort by timestamp and limit
        memories.sort(key=lambda m: m.timestamp, reverse=True)
        return memories[:limit]

    async def get_entity_memories(
        self,
        entity_name: str,
        context: MemoryContext,
        limit: int = 20,
    ) -> List[MemoryEntry]:
        """Get memories related to a specific entity"""
        
        memories = []
        
        for memory in self.memories.values():
            if (memory.memory_type == MemoryType.ENTITY and 
                entity_name.lower() in memory.content.lower() and
                memory.metadata.get("agent_id") == context.agent_id):
                memories.append(memory)
        
        # Sort by importance and recency
        memories.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
        return memories[:limit]

    async def consolidate_memories(
        self,
        agent_id: str,
        consolidation_threshold: int = 100,
    ) -> Dict[str, Any]:
        """Consolidate related memories to reduce redundancy"""
        
        console.print(f"[yellow]🔄 Starting memory consolidation for agent {agent_id}[/yellow]")
        
        agent_memory_ids = self.agent_memories.get(agent_id, [])
        if len(agent_memory_ids) < consolidation_threshold:
            return {"message": "Not enough memories to consolidate", "consolidated": 0}
        
        # Group similar memories
        consolidation_groups = await self._group_similar_memories(agent_memory_ids)
        
        consolidated_count = 0
        for group in consolidation_groups:
            if len(group) > 1:
                consolidated_memory = await self._merge_memories(group)
                if consolidated_memory:
                    # Remove old memories
                    for memory_id in group:
                        await self._remove_memory(memory_id)
                    
                    # Store consolidated memory
                    await self._store_consolidated_memory(consolidated_memory, agent_id)
                    consolidated_count += 1
        
        console.print(f"[green]✅ Consolidated {consolidated_count} memory groups[/green]")
        
        return {
            "consolidated_groups": consolidated_count,
            "memories_processed": len(agent_memory_ids),
            "total_memories_after": len(self.agent_memories.get(agent_id, [])),
        }

    async def cleanup_expired_memories(self) -> Dict[str, Any]:
        """Clean up expired memories and low-importance entries"""
        
        console.print(f"[yellow]🧹 Starting memory cleanup[/yellow]")
        
        now = datetime.now()
        expired_count = 0
        low_importance_count = 0
        
        expired_memories = []
        
        for memory_id, memory in self.memories.items():
            # Check TTL expiration
            if memory.ttl and now > memory.ttl:
                expired_memories.append(memory_id)
                expired_count += 1
                continue
            
            # Check low importance with age
            age_days = (now - memory.timestamp).days
            if (memory.importance < 0.2 and age_days > 7) or \
               (memory.importance < 0.1 and age_days > 3):
                expired_memories.append(memory_id)
                low_importance_count += 1
        
        # Remove expired memories
        for memory_id in expired_memories:
            await self._remove_memory(memory_id)
        
        self.stats["cleanup_runs"] += 1
        self.stats["last_cleanup"] = now
        
        cleanup_result = {
            "expired_memories": expired_count,
            "low_importance_memories": low_importance_count,
            "total_removed": len(expired_memories),
            "remaining_memories": len(self.memories),
        }
        
        console.print(f"[green]✅ Cleanup complete: {len(expired_memories)} memories removed[/green]")
        
        return cleanup_result

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        
        # Calculate memory distribution
        type_distribution = {}
        importance_distribution = {"high": 0, "medium": 0, "low": 0}
        agent_distribution = {}
        
        for memory in self.memories.values():
            # Type distribution
            type_key = memory.memory_type.value
            type_distribution[type_key] = type_distribution.get(type_key, 0) + 1
            
            # Importance distribution
            if memory.importance > 0.7:
                importance_distribution["high"] += 1
            elif memory.importance > 0.4:
                importance_distribution["medium"] += 1
            else:
                importance_distribution["low"] += 1
            
            # Agent distribution
            agent_id = memory.metadata.get("agent_id", "unknown")
            agent_distribution[agent_id] = agent_distribution.get(agent_id, 0) + 1
        
        return {
            "total_memories": len(self.memories),
            "total_agents": len(self.agent_memories),
            "total_sessions": len(self.session_memories),
            "type_distribution": type_distribution,
            "importance_distribution": importance_distribution,
            "agent_distribution": agent_distribution,
            "performance_stats": self.stats,
            "storage_path": str(self.storage_path),
            "vector_search_enabled": self.enable_vector_search,
        }

    # Private methods
    
    def _update_indices(self, memory: MemoryEntry) -> None:
        """Update memory indices"""
        
        # Agent index
        agent_id = memory.metadata.get("agent_id")
        if agent_id:
            if agent_id not in self.agent_memories:
                self.agent_memories[agent_id] = []
            self.agent_memories[agent_id].append(memory.id)
        
        # Session index
        session_id = memory.metadata.get("session_id")
        if session_id:
            if session_id not in self.session_memories:
                self.session_memories[session_id] = []
            self.session_memories[session_id].append(memory.id)
        
        # Tag index
        for tag in memory.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = []
            self.tag_index[tag].append(memory.id)

    def _filter_by_context(
        self,
        context: MemoryContext,
        memory_types: Optional[List[MemoryType]] = None,
    ) -> List[MemoryEntry]:
        """Filter memories by context criteria"""
        
        candidate_memories = []
        
        # Get memories for the agent
        agent_memory_ids = self.agent_memories.get(context.agent_id, [])
        
        for memory_id in agent_memory_ids:
            if memory_id not in self.memories:
                continue
                
            memory = self.memories[memory_id]
            
            # Filter by memory type
            if memory_types and memory.memory_type not in memory_types:
                continue
            
            # Filter by session if specified
            if (context.session_id and 
                memory.metadata.get("session_id") != context.session_id):
                continue
            
            # Filter by task if specified
            if (context.task_id and 
                memory.metadata.get("task_id") != context.task_id):
                continue
            
            candidate_memories.append(memory)
        
        return candidate_memories

    async def _vector_search(
        self,
        query: str,
        candidate_memories: List[MemoryEntry],
        limit: int,
    ) -> List[MemoryEntry]:
        """Perform vector similarity search"""
        
        if not query:
            return candidate_memories[:limit]
        
        query_embedding = await self._generate_embedding(query)
        
        # Calculate similarities
        scored_memories = []
        for memory in candidate_memories:
            if memory.embedding:
                similarity = self._calculate_similarity(query_embedding, memory.embedding)
                scored_memories.append((memory, similarity))
        
        # Sort by similarity and importance
        scored_memories.sort(
            key=lambda x: (x[1] * 0.7 + x[0].importance * 0.3),
            reverse=True
        )
        
        return [memory for memory, _ in scored_memories[:limit]]

    def _keyword_search(
        self,
        query: str,
        candidate_memories: List[MemoryEntry],
        limit: int,
    ) -> List[MemoryEntry]:
        """Perform keyword-based search"""
        
        if not query:
            return sorted(
                candidate_memories,
                key=lambda m: (m.importance, m.timestamp),
                reverse=True
            )[:limit]
        
        query_words = query.lower().split()
        scored_memories = []
        
        for memory in candidate_memories:
            score = 0.0
            content_lower = memory.content.lower()
            
            # Calculate keyword match score
            for word in query_words:
                if word in content_lower:
                    score += 1.0
                
                # Check tags
                for tag in memory.tags:
                    if word in tag.lower():
                        score += 0.5
            
            # Normalize by number of query words
            score = score / len(query_words) if query_words else 0.0
            
            # Combine with importance
            final_score = score * 0.7 + memory.importance * 0.3
            
            if score > 0:  # Only include memories with keyword matches
                scored_memories.append((memory, final_score))
        
        # Sort by combined score
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        return [memory for memory, _ in scored_memories[:limit]]

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text (mock implementation)"""
        
        # In a real implementation, this would use a proper embedding model
        # For now, we'll create a simple hash-based embedding
        import hashlib
        
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to a vector of floats
        embedding = []
        for i in range(0, len(text_hash), 2):
            hex_chunk = text_hash[i:i+2]
            embedding.append(int(hex_chunk, 16) / 255.0)
        
        # Pad or truncate to fixed size
        embedding_size = 128
        if len(embedding) < embedding_size:
            embedding.extend([0.0] * (embedding_size - len(embedding)))
        else:
            embedding = embedding[:embedding_size]
        
        return embedding

    def _calculate_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """Calculate cosine similarity between two embeddings"""
        
        if not embedding1 or not embedding2:
            return 0.0
        
        if NUMPY_AVAILABLE:
            # Use numpy for efficient calculation
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, similarity)  # Ensure non-negative
        else:
            # Fallback implementation without numpy
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
            norm1 = sum(a * a for a in embedding1) ** 0.5
            norm2 = sum(b * b for b in embedding2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return max(0.0, similarity)  # Ensure non-negative

    async def _group_similar_memories(
        self,
        memory_ids: List[str],
    ) -> List[List[str]]:
        """Group similar memories for consolidation"""
        
        # Simple grouping based on content similarity
        groups = []
        processed = set()
        
        for memory_id in memory_ids:
            if memory_id in processed or memory_id not in self.memories:
                continue
            
            memory = self.memories[memory_id]
            group = [memory_id]
            processed.add(memory_id)
            
            # Find similar memories
            for other_id in memory_ids:
                if (other_id in processed or other_id not in self.memories or
                    other_id == memory_id):
                    continue
                
                other_memory = self.memories[other_id]
                
                # Check if memories are similar
                if (memory.memory_type == other_memory.memory_type and
                    self._are_memories_similar(memory, other_memory)):
                    group.append(other_id)
                    processed.add(other_id)
            
            if len(group) > 1:  # Only add groups with multiple memories
                groups.append(group)
        
        return groups

    def _are_memories_similar(
        self,
        memory1: MemoryEntry,
        memory2: MemoryEntry,
    ) -> bool:
        """Check if two memories are similar enough to consolidate"""
        
        # Check embedding similarity if available
        if memory1.embedding and memory2.embedding:
            similarity = self._calculate_similarity(memory1.embedding, memory2.embedding)
            return similarity > 0.7
        
        # Fallback to keyword similarity
        words1 = set(memory1.content.lower().split())
        words2 = set(memory2.content.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard_similarity = len(intersection) / len(union)
        return jaccard_similarity > 0.5

    async def _merge_memories(self, memory_ids: List[str]) -> Optional[MemoryEntry]:
        """Merge multiple memories into one consolidated memory"""
        
        memories = [self.memories[mid] for mid in memory_ids if mid in self.memories]
        
        if not memories:
            return None
        
        # Combine content
        combined_content = "\n".join([m.content for m in memories])
        
        # Calculate average importance
        avg_importance = sum(m.importance for m in memories) / len(memories)
        
        # Combine tags
        all_tags = set()
        for memory in memories:
            all_tags.update(memory.tags)
        
        # Combine metadata
        combined_metadata = {}
        for memory in memories:
            combined_metadata.update(memory.metadata)
        
        combined_metadata["consolidated_from"] = memory_ids
        combined_metadata["consolidation_timestamp"] = datetime.now().isoformat()
        
        # Create consolidated memory
        consolidated = MemoryEntry(
            id=str(uuid.uuid4()),
            memory_type=memories[0].memory_type,
            content=combined_content,
            metadata=combined_metadata,
            timestamp=max(m.timestamp for m in memories),
            importance=avg_importance,
            tags=list(all_tags),
        )
        
        # Generate new embedding
        if self.enable_vector_search:
            consolidated.embedding = await self._generate_embedding(combined_content)
        
        return consolidated

    async def _store_consolidated_memory(
        self,
        memory: MemoryEntry,
        agent_id: str,
    ) -> None:
        """Store a consolidated memory"""
        
        self.memories[memory.id] = memory
        self._update_indices(memory)
        await self._save_memory(memory)

    async def _remove_memory(self, memory_id: str) -> None:
        """Remove a memory and update indices"""
        
        if memory_id not in self.memories:
            return
        
        memory = self.memories[memory_id]
        
        # Remove from main storage
        del self.memories[memory_id]
        
        # Remove from indices
        agent_id = memory.metadata.get("agent_id")
        if agent_id and agent_id in self.agent_memories:
            if memory_id in self.agent_memories[agent_id]:
                self.agent_memories[agent_id].remove(memory_id)
        
        session_id = memory.metadata.get("session_id")
        if session_id and session_id in self.session_memories:
            if memory_id in self.session_memories[session_id]:
                self.session_memories[session_id].remove(memory_id)
        
        for tag in memory.tags:
            if tag in self.tag_index and memory_id in self.tag_index[tag]:
                self.tag_index[tag].remove(memory_id)
        
        # Remove from disk
        memory_file = self.storage_path / f"{memory_id}.json"
        if memory_file.exists():
            memory_file.unlink()

    async def _cleanup_old_memories(self) -> None:
        """Clean up old memories when storage is full"""
        
        # Sort memories by importance and age
        sorted_memories = sorted(
            self.memories.values(),
            key=lambda m: (m.importance, m.timestamp),
            reverse=False,  # Least important and oldest first
        )
        
        # Remove 10% of memories
        remove_count = max(1, len(sorted_memories) // 10)
        memories_to_remove = sorted_memories[:remove_count]
        
        for memory in memories_to_remove:
            await self._remove_memory(memory.id)
        
        console.print(f"[yellow]🧹 Cleaned up {remove_count} old memories[/yellow]")

    async def _save_memory(self, memory: MemoryEntry) -> None:
        """Save memory to disk"""
        
        memory_file = self.storage_path / f"{memory.id}.json"
        
        try:
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            console.print(f"[red]❌ Failed to save memory {memory.id}: {e}[/red]")

    def _load_memories(self) -> None:
        """Load memories from disk"""
        
        try:
            for memory_file in self.storage_path.glob("*.json"):
                try:
                    with open(memory_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    memory = MemoryEntry.from_dict(data)
                    self.memories[memory.id] = memory
                    self._update_indices(memory)
                    
                except Exception as e:
                    console.print(f"[yellow]⚠️ Failed to load memory from {memory_file}: {e}[/yellow]")
                    continue
            
            console.print(f"[green]✅ Loaded {len(self.memories)} memories from disk[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to load memories: {e}[/red]")

    def _start_cleanup_task(self) -> None:
        """Start background cleanup task"""
        
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval)
                    await self.cleanup_expired_memories()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    console.print(f"[red]❌ Cleanup task error: {e}[/red]")
        
        # Cancel existing task if running
        if hasattr(self, '_cleanup_task') and self._cleanup_task:
            self._cleanup_task.cancel()
        
        try:
            # Only create task if there's a running event loop
            loop = asyncio.get_running_loop()
            self._cleanup_task = asyncio.create_task(cleanup_loop())
        except RuntimeError:
            # No running event loop, cleanup will be manual
            self._cleanup_task = None
            raise

    def __del__(self):
        """Cleanup when manager is destroyed"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()