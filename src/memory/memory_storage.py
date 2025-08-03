#!/usr/bin/env python3
"""
Memory Storage Systems for CrewAI Agent Memory
Provides specialized storage backends for different memory types
"""

import json
import sqlite3
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from rich.console import Console
    console = Console()
except ImportError:
    class MockConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = MockConsole()

from .memory_manager import MemoryEntry, MemoryType

class MemoryStorage(ABC):
    """Abstract base class for memory storage backends"""
    
    @abstractmethod
    async def store(self, memory: MemoryEntry) -> bool:
        """Store a memory entry"""
        pass
    
    @abstractmethod
    async def retrieve(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID"""
        pass
    
    @abstractmethod
    async def search(
        self, 
        query: str, 
        memory_type: Optional[MemoryType] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Search memories"""
        pass
    
    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory entry"""
        pass
    
    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Clean up expired memories"""
        pass

class VectorMemoryStorage(MemoryStorage):
    """
    Vector-based memory storage using embeddings for semantic search
    Specialized for semantic similarity and content-based retrieval
    """
    
    def __init__(
        self,
        storage_path: str = "vector_memory.db",
        embedding_dimension: int = 128,
    ):
        self.storage_path = Path(storage_path)
        self.embedding_dimension = embedding_dimension
        self._init_database()
        
        console.print(f"[green]✅ Vector Memory Storage initialized[/green]")
        console.print(f"[cyan]   • Database: {self.storage_path}[/cyan]")
        console.print(f"[cyan]   • Embedding dimension: {embedding_dimension}[/cyan]")
    
    def _init_database(self):
        """Initialize SQLite database with vector storage"""
        
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    importance REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    tags TEXT,
                    embedding BLOB,
                    ttl TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ttl ON memories(ttl)")
            
            conn.commit()
    
    async def store(self, memory: MemoryEntry) -> bool:
        """Store a memory entry with vector embedding"""
        
        try:
            # Serialize complex fields
            metadata_json = json.dumps(memory.metadata)
            tags_json = json.dumps(memory.tags)
            embedding_blob = json.dumps(memory.embedding) if memory.embedding else None
            
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memories 
                    (id, memory_type, content, metadata, timestamp, importance, 
                     access_count, last_accessed, tags, embedding, ttl)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.id,
                    memory.memory_type.value,
                    memory.content,
                    metadata_json,
                    memory.timestamp.isoformat(),
                    memory.importance,
                    memory.access_count,
                    memory.last_accessed.isoformat() if memory.last_accessed else None,
                    tags_json,
                    embedding_blob,
                    memory.ttl.isoformat() if memory.ttl else None
                ))
                conn.commit()
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to store memory {memory.id}: {e}[/red]")
            return False
    
    async def retrieve(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by ID"""
        
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM memories WHERE id = ?", (memory_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_memory(row)
                return None
                
        except Exception as e:
            console.print(f"[red]❌ Failed to retrieve memory {memory_id}: {e}[/red]")
            return None
    
    async def search(
        self, 
        query: str, 
        memory_type: Optional[MemoryType] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Search memories using content matching"""
        
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.row_factory = sqlite3.Row
                
                # Build query
                sql = "SELECT * FROM memories WHERE content LIKE ?"
                params = [f"%{query}%"]
                
                if memory_type:
                    sql += " AND memory_type = ?"
                    params.append(memory_type.value)
                
                sql += " ORDER BY importance DESC, timestamp DESC LIMIT ?"
                params.append(str(limit))
                
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()
                
                return [self._row_to_memory(row) for row in rows]
                
        except Exception as e:
            console.print(f"[red]❌ Failed to search memories: {e}[/red]")
            return []
    
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory entry"""
        
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                conn.commit()
                return True
                
        except Exception as e:
            console.print(f"[red]❌ Failed to delete memory {memory_id}: {e}[/red]")
            return False
    
    async def cleanup_expired(self) -> int:
        """Clean up expired memories"""
        
        try:
            now = datetime.now()
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM memories WHERE ttl IS NOT NULL AND ttl < ?",
                    (now.isoformat(),)
                )
                deleted_count = cursor.rowcount
                conn.commit()
                
                console.print(f"[green]✅ Cleaned up {deleted_count} expired memories[/green]")
                return deleted_count
                
        except Exception as e:
            console.print(f"[red]❌ Failed to cleanup expired memories: {e}[/red]")
            return 0
    
    def _row_to_memory(self, row: sqlite3.Row) -> MemoryEntry:
        """Convert database row to MemoryEntry"""
        
        return MemoryEntry(
            id=row['id'],
            memory_type=MemoryType(row['memory_type']),
            content=row['content'],
            metadata=json.loads(row['metadata']),
            timestamp=datetime.fromisoformat(row['timestamp']),
            importance=row['importance'],
            access_count=row['access_count'],
            last_accessed=datetime.fromisoformat(row['last_accessed']) if row['last_accessed'] else None,
            tags=json.loads(row['tags']) if row['tags'] else [],
            embedding=json.loads(row['embedding']) if row['embedding'] else None,
            ttl=datetime.fromisoformat(row['ttl']) if row['ttl'] else None
        )

class ConversationMemory(MemoryStorage):
    """
    Specialized storage for conversation memories
    Optimized for sequential access and dialogue context
    """
    
    def __init__(self, storage_path: str = "conversation_memory.db"):
        self.storage_path = Path(storage_path)
        self._init_database()
        
        console.print(f"[green]💬 Conversation Memory Storage initialized[/green]")
    
    def _init_database(self):
        """Initialize conversation-specific database"""
        
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    importance REAL NOT NULL,
                    metadata TEXT,
                    turn_number INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Indexes for conversation retrieval
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversation_id ON conversations(conversation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_conversation ON conversations(agent_id, conversation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations(timestamp)")
            
            conn.commit()
    
    async def store(self, memory: MemoryEntry) -> bool:
        """Store conversation memory"""
        
        if memory.memory_type != MemoryType.CONVERSATION:
            return False
        
        try:
            metadata = memory.metadata
            
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO conversations 
                    (id, conversation_id, agent_id, role, content, timestamp, 
                     importance, metadata, turn_number)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.id,
                    metadata.get('conversation_id', ''),
                    metadata.get('agent_id', ''),
                    metadata.get('role', 'unknown'),
                    memory.content,
                    memory.timestamp.isoformat(),
                    memory.importance,
                    json.dumps(metadata),
                    metadata.get('turn_number', 0)
                ))
                conn.commit()
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to store conversation memory: {e}[/red]")
            return False
    
    async def get_conversation_history(
        self, 
        conversation_id: str, 
        agent_id: Optional[str] = None,
        limit: int = 50
    ) -> List[MemoryEntry]:
        """Get conversation history in chronological order"""
        
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.row_factory = sqlite3.Row
                
                sql = "SELECT * FROM conversations WHERE conversation_id = ?"
                params = [conversation_id]
                
                if agent_id:
                    sql += " AND agent_id = ?"
                    params.append(agent_id)
                
                sql += " ORDER BY timestamp ASC LIMIT ?"
                params.append(str(limit))
                
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()
                
                return [self._row_to_memory(row) for row in rows]
                
        except Exception as e:
            console.print(f"[red]❌ Failed to get conversation history: {e}[/red]")
            return []
    
    async def retrieve(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve conversation memory by ID"""
        
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM conversations WHERE id = ?", (memory_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_memory(row)
                return None
                
        except Exception as e:
            console.print(f"[red]❌ Failed to retrieve conversation memory: {e}[/red]")
            return None
    
    async def search(
        self, 
        query: str, 
        memory_type: Optional[MemoryType] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Search conversation memories"""
        
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.row_factory = sqlite3.Row
                
                cursor = conn.execute("""
                    SELECT * FROM conversations 
                    WHERE content LIKE ? 
                    ORDER BY timestamp DESC LIMIT ?
                """, (f"%{query}%", limit))
                
                rows = cursor.fetchall()
                return [self._row_to_memory(row) for row in rows]
                
        except Exception as e:
            console.print(f"[red]❌ Failed to search conversation memories: {e}[/red]")
            return []
    
    async def delete(self, memory_id: str) -> bool:
        """Delete conversation memory"""
        
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute("DELETE FROM conversations WHERE id = ?", (memory_id,))
                conn.commit()
                return True
                
        except Exception as e:
            console.print(f"[red]❌ Failed to delete conversation memory: {e}[/red]")
            return False
    
    async def cleanup_expired(self) -> int:
        """Clean up old conversation memories"""
        
        try:
            # Remove conversations older than 30 days
            cutoff = datetime.now() - timedelta(days=30)
            
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM conversations WHERE timestamp < ?",
                    (cutoff.isoformat(),)
                )
                deleted_count = cursor.rowcount
                conn.commit()
                
                console.print(f"[green]💬 Cleaned up {deleted_count} old conversation memories[/green]")
                return deleted_count
                
        except Exception as e:
            console.print(f"[red]❌ Failed to cleanup conversation memories: {e}[/red]")
            return 0
    
    def _row_to_memory(self, row: sqlite3.Row) -> MemoryEntry:
        """Convert database row to MemoryEntry"""
        
        metadata = json.loads(row['metadata']) if row['metadata'] else {}
        
        return MemoryEntry(
            id=row['id'],
            memory_type=MemoryType.CONVERSATION,
            content=row['content'],
            metadata=metadata,
            timestamp=datetime.fromisoformat(row['timestamp']),
            importance=row['importance']
        )

class EntityMemory(MemoryStorage):
    """
    Specialized storage for entity memories
    Optimized for entity-based retrieval and relationship tracking
    """
    
    def __init__(self, storage_path: str = "entity_memory.db"):
        self.storage_path = Path(storage_path)
        self._init_database()
        
        console.print(f"[green]👤 Entity Memory Storage initialized[/green]")
    
    def _init_database(self):
        """Initialize entity-specific database"""
        
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    entity_name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    importance REAL NOT NULL,
                    metadata TEXT,
                    last_updated TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Indexes for entity retrieval
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(entity_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entity_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_entity ON entities(agent_id, entity_name)")
            
            conn.commit()
    
    async def store(self, memory: MemoryEntry) -> bool:
        """Store entity memory"""
        
        if memory.memory_type != MemoryType.ENTITY:
            return False
        
        try:
            metadata = memory.metadata
            
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO entities 
                    (id, entity_name, entity_type, agent_id, content, timestamp, 
                     importance, metadata, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.id,
                    metadata.get('entity_name', ''),
                    metadata.get('entity_type', 'general'),
                    metadata.get('agent_id', ''),
                    memory.content,
                    memory.timestamp.isoformat(),
                    memory.importance,
                    json.dumps(metadata),
                    datetime.now().isoformat()
                ))
                conn.commit()
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to store entity memory: {e}[/red]")
            return False
    
    async def get_entity_info(
        self, 
        entity_name: str, 
        agent_id: Optional[str] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Get information about a specific entity"""
        
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.row_factory = sqlite3.Row
                
                sql = "SELECT * FROM entities WHERE entity_name LIKE ?"
                params = [f"%{entity_name}%"]
                
                if agent_id:
                    sql += " AND agent_id = ?"
                    params.append(agent_id)
                
                sql += " ORDER BY importance DESC, last_updated DESC LIMIT ?"
                params.append(str(limit))
                
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()
                
                return [self._row_to_memory(row) for row in rows]
                
        except Exception as e:
            console.print(f"[red]❌ Failed to get entity info: {e}[/red]")
            return []
    
    async def retrieve(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve entity memory by ID"""
        
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM entities WHERE id = ?", (memory_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_memory(row)
                return None
                
        except Exception as e:
            console.print(f"[red]❌ Failed to retrieve entity memory: {e}[/red]")
            return None
    
    async def search(
        self, 
        query: str, 
        memory_type: Optional[MemoryType] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Search entity memories"""
        
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.row_factory = sqlite3.Row
                
                cursor = conn.execute("""
                    SELECT * FROM entities 
                    WHERE content LIKE ? OR entity_name LIKE ?
                    ORDER BY importance DESC, last_updated DESC LIMIT ?
                """, (f"%{query}%", f"%{query}%", limit))
                
                rows = cursor.fetchall()
                return [self._row_to_memory(row) for row in rows]
                
        except Exception as e:
            console.print(f"[red]❌ Failed to search entity memories: {e}[/red]")
            return []
    
    async def delete(self, memory_id: str) -> bool:
        """Delete entity memory"""
        
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute("DELETE FROM entities WHERE id = ?", (memory_id,))
                conn.commit()
                return True
                
        except Exception as e:
            console.print(f"[red]❌ Failed to delete entity memory: {e}[/red]")
            return False
    
    async def cleanup_expired(self) -> int:
        """Clean up old entity memories"""
        
        try:
            # Remove low-importance entities older than 90 days
            cutoff = datetime.now() - timedelta(days=90)
            
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM entities WHERE importance < 0.3 AND timestamp < ?",
                    (cutoff.isoformat(),)
                )
                deleted_count = cursor.rowcount
                conn.commit()
                
                console.print(f"[green]👤 Cleaned up {deleted_count} old entity memories[/green]")
                return deleted_count
                
        except Exception as e:
            console.print(f"[red]❌ Failed to cleanup entity memories: {e}[/red]")
            return 0
    
    def _row_to_memory(self, row: sqlite3.Row) -> MemoryEntry:
        """Convert database row to MemoryEntry"""
        
        metadata = json.loads(row['metadata']) if row['metadata'] else {}
        
        return MemoryEntry(
            id=row['id'],
            memory_type=MemoryType.ENTITY,
            content=row['content'],
            metadata=metadata,
            timestamp=datetime.fromisoformat(row['timestamp']),
            importance=row['importance']
        )

class LongTermMemory(MemoryStorage):
    """
    Long-term memory storage for persistent knowledge and experiences
    Optimized for durability and efficient long-term retrieval
    """
    
    def __init__(self, storage_path: str = "longterm_memory.db"):
        self.storage_path = Path(storage_path)
        self._init_database()
        
        console.print(f"[green]🧠 Long-term Memory Storage initialized[/green]")
    
    def _init_database(self):
        """Initialize long-term memory database"""
        
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS longterm_memories (
                    id TEXT PRIMARY KEY,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    topic TEXT,
                    timestamp TEXT NOT NULL,
                    importance REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Indexes for long-term retrieval
            conn.execute("CREATE INDEX IF NOT EXISTS idx_topic ON longterm_memories(topic)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON longterm_memories(importance)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_topic ON longterm_memories(agent_id, topic)")
            
            conn.commit()
    
    async def store(self, memory: MemoryEntry) -> bool:
        """Store long-term memory"""
        
        if memory.memory_type not in [MemoryType.KNOWLEDGE, MemoryType.EXPERIENCE]:
            return False
        
        try:
            metadata = memory.metadata
            
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO longterm_memories 
                    (id, memory_type, content, agent_id, topic, timestamp, 
                     importance, access_count, last_accessed, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory.id,
                    memory.memory_type.value,
                    memory.content,
                    metadata.get('agent_id', ''),
                    metadata.get('topic', 'general'),
                    memory.timestamp.isoformat(),
                    memory.importance,
                    memory.access_count,
                    memory.last_accessed.isoformat() if memory.last_accessed else None,
                    json.dumps(metadata)
                ))
                conn.commit()
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to store long-term memory: {e}[/red]")
            return False
    
    async def retrieve(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve long-term memory by ID"""
        
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM longterm_memories WHERE id = ?", (memory_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    # Update access tracking
                    conn.execute(
                        "UPDATE longterm_memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
                        (datetime.now().isoformat(), memory_id)
                    )
                    conn.commit()
                    
                    return self._row_to_memory(row)
                return None
                
        except Exception as e:
            console.print(f"[red]❌ Failed to retrieve long-term memory: {e}[/red]")
            return None
    
    async def search(
        self, 
        query: str, 
        memory_type: Optional[MemoryType] = None,
        limit: int = 10
    ) -> List[MemoryEntry]:
        """Search long-term memories"""
        
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.row_factory = sqlite3.Row
                
                sql = "SELECT * FROM longterm_memories WHERE content LIKE ? OR topic LIKE ?"
                params = [f"%{query}%", f"%{query}%"]
                
                if memory_type:
                    sql += " AND memory_type = ?"
                    params.append(memory_type.value)
                
                sql += " ORDER BY importance DESC, access_count DESC LIMIT ?"
                params.append(str(limit))
                
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()
                
                return [self._row_to_memory(row) for row in rows]
                
        except Exception as e:
            console.print(f"[red]❌ Failed to search long-term memories: {e}[/red]")
            return []
    
    async def delete(self, memory_id: str) -> bool:
        """Delete long-term memory"""
        
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute("DELETE FROM longterm_memories WHERE id = ?", (memory_id,))
                conn.commit()
                return True
                
        except Exception as e:
            console.print(f"[red]❌ Failed to delete long-term memory: {e}[/red]")
            return False
    
    async def cleanup_expired(self) -> int:
        """Clean up unused long-term memories"""
        
        try:
            # Remove very low importance memories not accessed in 6 months
            cutoff = datetime.now() - timedelta(days=180)
            
            with sqlite3.connect(self.storage_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM longterm_memories 
                    WHERE importance < 0.2 
                    AND access_count = 0 
                    AND timestamp < ?
                """, (cutoff.isoformat(),))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                console.print(f"[green]🧠 Cleaned up {deleted_count} unused long-term memories[/green]")
                return deleted_count
                
        except Exception as e:
            console.print(f"[red]❌ Failed to cleanup long-term memories: {e}[/red]")
            return 0
    
    def _row_to_memory(self, row: sqlite3.Row) -> MemoryEntry:
        """Convert database row to MemoryEntry"""
        
        metadata = json.loads(row['metadata']) if row['metadata'] else {}
        
        return MemoryEntry(
            id=row['id'],
            memory_type=MemoryType(row['memory_type']),
            content=row['content'],
            metadata=metadata,
            timestamp=datetime.fromisoformat(row['timestamp']),
            importance=row['importance'],
            access_count=row['access_count'],
            last_accessed=datetime.fromisoformat(row['last_accessed']) if row['last_accessed'] else None
        )