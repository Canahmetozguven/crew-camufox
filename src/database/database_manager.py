"""Database manager for handling all database operations."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Dict, Any

try:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy.orm import Session
    from sqlalchemy import text

    ASYNC_SQLALCHEMY_AVAILABLE = True
except ImportError:
    ASYNC_SQLALCHEMY_AVAILABLE = False

from src.config.settings import get_settings
from src.utils.logging import get_logger
from .models import SQLALCHEMY_AVAILABLE, create_database_engine, create_tables, get_session_maker

logger = get_logger(__name__)


class DatabaseManager:
    """Manages database connections and operations."""

    def __init__(self, database_url: Optional[str] = None):
        """Initialize database manager."""
        self.settings = get_settings()
        self.database_url = database_url or self.settings.database.url
        self.engine = None
        self.async_engine = None
        self.session_maker = None
        self.async_session_maker = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database connections and create tables."""
        if self._initialized:
            logger.warning("Database already initialized")
            return

        try:
            # Create sync engine for table creation
            if SQLALCHEMY_AVAILABLE:
                self.engine = create_database_engine(self.database_url)
                create_tables(self.engine)
                self.session_maker = get_session_maker(self.engine)
                logger.info("Sync database engine initialized")

                # Create async engine if available
                if ASYNC_SQLALCHEMY_AVAILABLE:
                    # Convert sync URL to async for SQLite
                    async_url = self.database_url
                    if async_url.startswith("sqlite:///"):
                        async_url = async_url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
                    elif async_url.startswith("postgresql://"):
                        async_url = async_url.replace("postgresql://", "postgresql+asyncpg://", 1)

                    self.async_engine = create_async_engine(
                        async_url,
                        echo=self.settings.database.echo,
                        pool_size=self.settings.database.pool_size,
                        max_overflow=self.settings.database.max_overflow,
                        pool_timeout=self.settings.database.pool_timeout,
                        pool_recycle=self.settings.database.pool_recycle,
                    )
                    self.async_session_maker = async_sessionmaker(
                        self.async_engine, class_=AsyncSession
                    )
                    logger.info("Async database engine initialized")
                else:
                    logger.warning("Async SQLAlchemy not available, using sync operations only")
            else:
                logger.warning("SQLAlchemy not available, database operations disabled")

            self._initialized = True
            logger.info(f"Database manager initialized with URL: {self.database_url}")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def close(self) -> None:
        """Close database connections."""
        if self.async_engine:
            await self.async_engine.dispose()
            logger.info("Async database engine closed")

        if self.engine:
            self.engine.dispose()
            logger.info("Sync database engine closed")

        self._initialized = False

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session context manager."""
        if not self._initialized:
            await self.initialize()

        if not self.async_session_maker:
            raise RuntimeError("Async database not available")

        async with self.async_session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    def get_sync_session(self) -> Session:
        """Get sync database session."""
        if not self._initialized:
            # Can't await in sync function, so initialize synchronously
            if SQLALCHEMY_AVAILABLE:
                self.engine = create_database_engine(self.database_url)
                create_tables(self.engine)
                self.session_maker = get_session_maker(self.engine)
                self._initialized = True
            else:
                raise RuntimeError("SQLAlchemy not available")

        if not self.session_maker:
            raise RuntimeError("Database not available")

        return self.session_maker()

    async def health_check(self) -> Dict[str, Any]:
        """Check database health."""
        if not self._initialized:
            return {"status": "not_initialized", "available": False}

        try:
            if self.async_session_maker:
                async with self.get_async_session() as session:
                    await session.execute(text("SELECT 1"))
                    return {
                        "status": "healthy",
                        "available": True,
                        "type": "async",
                        "url": self.database_url,
                    }
            elif self.session_maker:
                session = self.get_sync_session()
                try:
                    session.execute(text("SELECT 1"))
                    return {
                        "status": "healthy",
                        "available": True,
                        "type": "sync",
                        "url": self.database_url,
                    }
                finally:
                    session.close()
            else:
                return {"status": "unavailable", "available": False}

        except (RuntimeError, OSError, ConnectionError) as e:
            logger.error(f"Database health check failed: {e}")
            return {"status": "unhealthy", "available": False, "error": str(e)}

    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self._initialized:
            return {"error": "Database not initialized"}

        try:
            stats = {
                "initialized": self._initialized,
                "sync_available": bool(self.engine),
                "async_available": bool(self.async_engine),
                "url": self.database_url,
            }

            # Add connection pool stats if available
            if self.engine and hasattr(self.engine.pool, "size"):
                stats["pool_size"] = self.engine.pool.size()
                stats["checked_in"] = self.engine.pool.checkedin()
                stats["checked_out"] = self.engine.pool.checkedout()

            return stats

        except (RuntimeError, OSError, AttributeError) as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}


# Global database manager instance
_database_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get or create global database manager instance."""
    global _database_manager
    if _database_manager is None:
        _database_manager = DatabaseManager()
    return _database_manager


async def initialize_database() -> DatabaseManager:
    """Initialize the global database manager."""
    manager = get_database_manager()
    await manager.initialize()
    return manager


async def close_database() -> None:
    """Close the global database manager."""
    global _database_manager
    if _database_manager:
        await _database_manager.close()
        _database_manager = None
