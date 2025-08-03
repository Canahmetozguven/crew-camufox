"""Repository layer for database operations."""

from datetime import datetime
from typing import List, Optional, Dict, Any

try:
    from sqlalchemy.orm import Session
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.future import select
    from sqlalchemy import desc, and_, or_

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

from src.utils.logging import get_logger
from .models import (
    ResearchSession,
    SearchResult,
    ResearchPlan,
    ResearchOutput,
    ResearchSessionCreate,
    ResearchSessionUpdate,
    SearchResultCreate,
    ResearchPlanCreate,
    ResearchOutputCreate,
)
from .database_manager import get_database_manager

logger = get_logger(__name__)


class ResearchRepository:
    """Repository for research data operations."""

    def __init__(self):
        """Initialize repository."""
        self.db_manager = get_database_manager()

    # Research Session Operations
    async def create_session(
        self, session_data: ResearchSessionCreate
    ) -> Optional[ResearchSession]:
        """Create a new research session."""
        try:
            if self.db_manager.async_session_maker:
                async with self.db_manager.get_async_session() as db:
                    db_session = ResearchSession(
                        mission_id=session_data.mission_id,
                        query=session_data.query,
                        status=session_data.status,
                        extra_data=session_data.extra_data,
                    )
                    db.add(db_session)
                    await db.commit()
                    await db.refresh(db_session)

                    # Access attributes while session is still active
                    session_id = db_session.id
                    mission_id = db_session.mission_id
                    query = db_session.query
                    status = db_session.status
                    created_at = db_session.created_at
                    completed_at = db_session.completed_at
                    extra_data = db_session.extra_data

                    # Return a simple dict-like object
                    class SessionResult:
                        def __init__(
                            self,
                            id,
                            mission_id,
                            query,
                            status,
                            created_at,
                            completed_at,
                            extra_data,
                        ):
                            self.id = id
                            self.mission_id = mission_id
                            self.query = query
                            self.status = status
                            self.created_at = created_at
                            self.completed_at = completed_at
                            self.extra_data = extra_data

                    return SessionResult(
                        session_id, mission_id, query, status, created_at, completed_at, extra_data
                    )
            else:
                # Fallback to sync
                db = self.db_manager.get_sync_session()
                try:
                    db_session = ResearchSession(
                        mission_id=session_data.mission_id,
                        query=session_data.query,
                        status=session_data.status,
                        extra_data=session_data.extra_data,
                    )
                    db.add(db_session)
                    db.flush()
                    db.refresh(db_session)
                    db.commit()
                    return db_session
                finally:
                    db.close()
        except Exception as e:
            logger.error(f"Failed to create research session: {e}")
            return None

    async def get_session_by_mission_id(self, mission_id: str) -> Optional[ResearchSession]:
        """Get research session by mission ID."""
        try:
            if self.db_manager.async_session_maker:
                async with self.db_manager.get_async_session() as db:
                    result = await db.execute(
                        select(ResearchSession).filter(ResearchSession.mission_id == mission_id)
                    )
                    return result.scalar_one_or_none()
            else:
                db = self.db_manager.get_sync_session()
                try:
                    return (
                        db.query(ResearchSession)
                        .filter(ResearchSession.mission_id == mission_id)
                        .first()
                    )
                finally:
                    db.close()
        except Exception as e:
            logger.error(f"Failed to get session by mission ID {mission_id}: {e}")
            return None

    async def update_session(self, mission_id: str, update_data: ResearchSessionUpdate) -> bool:
        """Update research session."""
        try:
            if self.db_manager.async_session_maker:
                async with self.db_manager.get_async_session() as db:
                    result = await db.execute(
                        select(ResearchSession).filter(ResearchSession.mission_id == mission_id)
                    )
                    db_session = result.scalar_one_or_none()
                    if db_session:
                        for field, value in update_data.model_dump(exclude_unset=True).items():
                            setattr(db_session, field, value)
                        return True
                    return False
            else:
                db = self.db_manager.get_sync_session()
                try:
                    db_session = (
                        db.query(ResearchSession)
                        .filter(ResearchSession.mission_id == mission_id)
                        .first()
                    )
                    if db_session:
                        for field, value in update_data.model_dump(exclude_unset=True).items():
                            setattr(db_session, field, value)
                        db.commit()
                        return True
                    return False
                finally:
                    db.close()
        except Exception as e:
            logger.error(f"Failed to update session {mission_id}: {e}")
            return False

    async def list_sessions(self, limit: int = 100, offset: int = 0) -> List[ResearchSession]:
        """List research sessions."""
        try:
            if self.db_manager.async_session_maker:
                async with self.db_manager.get_async_session() as db:
                    result = await db.execute(
                        select(ResearchSession)
                        .order_by(desc(ResearchSession.created_at))
                        .limit(limit)
                        .offset(offset)
                    )
                    return result.scalars().all()
            else:
                db = self.db_manager.get_sync_session()
                try:
                    return (
                        db.query(ResearchSession)
                        .order_by(desc(ResearchSession.created_at))
                        .limit(limit)
                        .offset(offset)
                        .all()
                    )
                finally:
                    db.close()
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []

    # Search Result Operations
    async def add_search_result(self, result_data: SearchResultCreate) -> Optional[SearchResult]:
        """Add a search result."""
        try:
            if self.db_manager.async_session_maker:
                async with self.db_manager.get_async_session() as db:
                    db_result = SearchResult(
                        session_id=result_data.session_id,
                        url=result_data.url,
                        title=result_data.title,
                        content=result_data.content,
                        source=result_data.source,
                        relevance_score=result_data.relevance_score,
                        extra_data=result_data.extra_data,
                    )
                    db.add(db_result)
                    await db.flush()
                    await db.refresh(db_result)
                    return db_result
            else:
                db = self.db_manager.get_sync_session()
                try:
                    db_result = SearchResult(
                        session_id=result_data.session_id,
                        url=result_data.url,
                        title=result_data.title,
                        content=result_data.content,
                        source=result_data.source,
                        relevance_score=result_data.relevance_score,
                        extra_data=result_data.extra_data,
                    )
                    db.add(db_result)
                    db.flush()
                    db.refresh(db_result)
                    db.commit()
                    return db_result
                finally:
                    db.close()
        except Exception as e:
            logger.error(f"Failed to add search result: {e}")
            return None

    async def get_search_results(self, session_id: int, limit: int = 50) -> List[SearchResult]:
        """Get search results for a session."""
        try:
            if self.db_manager.async_session_maker:
                async with self.db_manager.get_async_session() as db:
                    result = await db.execute(
                        select(SearchResult)
                        .filter(SearchResult.session_id == session_id)
                        .order_by(desc(SearchResult.relevance_score))
                        .limit(limit)
                    )
                    return result.scalars().all()
            else:
                db = self.db_manager.get_sync_session()
                try:
                    return (
                        db.query(SearchResult)
                        .filter(SearchResult.session_id == session_id)
                        .order_by(desc(SearchResult.relevance_score))
                        .limit(limit)
                        .all()
                    )
                finally:
                    db.close()
        except Exception as e:
            logger.error(f"Failed to get search results for session {session_id}: {e}")
            return []

    # Research Plan Operations
    async def save_research_plan(self, plan_data: ResearchPlanCreate) -> Optional[ResearchPlan]:
        """Save a research plan."""
        try:
            if self.db_manager.async_session_maker:
                async with self.db_manager.get_async_session() as db:
                    db_plan = ResearchPlan(
                        session_id=plan_data.session_id,
                        plan_data=plan_data.plan_data,
                        agent_type=plan_data.agent_type,
                        version=plan_data.version,
                    )
                    db.add(db_plan)
                    await db.flush()
                    await db.refresh(db_plan)
                    return db_plan
            else:
                db = self.db_manager.get_sync_session()
                try:
                    db_plan = ResearchPlan(
                        session_id=plan_data.session_id,
                        plan_data=plan_data.plan_data,
                        agent_type=plan_data.agent_type,
                        version=plan_data.version,
                    )
                    db.add(db_plan)
                    db.flush()
                    db.refresh(db_plan)
                    db.commit()
                    return db_plan
                finally:
                    db.close()
        except Exception as e:
            logger.error(f"Failed to save research plan: {e}")
            return None

    async def get_latest_research_plan(
        self, session_id: int, agent_type: Optional[str] = None
    ) -> Optional[ResearchPlan]:
        """Get the latest research plan for a session."""
        try:
            if self.db_manager.async_session_maker:
                async with self.db_manager.get_async_session() as db:
                    query = select(ResearchPlan).filter(ResearchPlan.session_id == session_id)
                    if agent_type:
                        query = query.filter(ResearchPlan.agent_type == agent_type)
                    query = query.order_by(
                        desc(ResearchPlan.version), desc(ResearchPlan.created_at)
                    )
                    result = await db.execute(query)
                    return result.scalar_one_or_none()
            else:
                db = self.db_manager.get_sync_session()
                try:
                    query = db.query(ResearchPlan).filter(ResearchPlan.session_id == session_id)
                    if agent_type:
                        query = query.filter(ResearchPlan.agent_type == agent_type)
                    return query.order_by(
                        desc(ResearchPlan.version), desc(ResearchPlan.created_at)
                    ).first()
                finally:
                    db.close()
        except Exception as e:
            logger.error(f"Failed to get latest research plan for session {session_id}: {e}")
            return None

    # Research Output Operations
    async def save_research_output(
        self, output_data: ResearchOutputCreate
    ) -> Optional[ResearchOutput]:
        """Save research output."""
        try:
            if self.db_manager.async_session_maker:
                async with self.db_manager.get_async_session() as db:
                    db_output = ResearchOutput(
                        session_id=output_data.session_id,
                        output_type=output_data.output_type,
                        content=output_data.content,
                        file_path=output_data.file_path,
                        extra_data=output_data.extra_data,
                    )
                    db.add(db_output)
                    await db.flush()
                    await db.refresh(db_output)
                    return db_output
            else:
                db = self.db_manager.get_sync_session()
                try:
                    db_output = ResearchOutput(
                        session_id=output_data.session_id,
                        output_type=output_data.output_type,
                        content=output_data.content,
                        file_path=output_data.file_path,
                        extra_data=output_data.extra_data,
                    )
                    db.add(db_output)
                    db.flush()
                    db.refresh(db_output)
                    db.commit()
                    return db_output
                finally:
                    db.close()
        except Exception as e:
            logger.error(f"Failed to save research output: {e}")
            return None

    async def get_research_outputs(
        self, session_id: int, output_type: Optional[str] = None
    ) -> List[ResearchOutput]:
        """Get research outputs for a session."""
        try:
            if self.db_manager.async_session_maker:
                async with self.db_manager.get_async_session() as db:
                    query = select(ResearchOutput).filter(ResearchOutput.session_id == session_id)
                    if output_type:
                        query = query.filter(ResearchOutput.output_type == output_type)
                    query = query.order_by(desc(ResearchOutput.created_at))
                    result = await db.execute(query)
                    return result.scalars().all()
            else:
                db = self.db_manager.get_sync_session()
                try:
                    query = db.query(ResearchOutput).filter(ResearchOutput.session_id == session_id)
                    if output_type:
                        query = query.filter(ResearchOutput.output_type == output_type)
                    return query.order_by(desc(ResearchOutput.created_at)).all()
                finally:
                    db.close()
        except Exception as e:
            logger.error(f"Failed to get research outputs for session {session_id}: {e}")
            return []

    async def cleanup_old_sessions(self, days: int = 30) -> int:
        """Clean up old research sessions."""
        try:
            cutoff_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days)

            if self.db_manager.async_session_maker:
                async with self.db_manager.get_async_session() as db:
                    result = await db.execute(
                        select(ResearchSession).filter(ResearchSession.created_at < cutoff_date)
                    )
                    sessions_to_delete = result.scalars().all()
                    count = len(sessions_to_delete)
                    for session in sessions_to_delete:
                        await db.delete(session)
                    return count
            else:
                db = self.db_manager.get_sync_session()
                try:
                    sessions_to_delete = (
                        db.query(ResearchSession)
                        .filter(ResearchSession.created_at < cutoff_date)
                        .all()
                    )
                    count = len(sessions_to_delete)
                    for session in sessions_to_delete:
                        db.delete(session)
                    db.commit()
                    return count
                finally:
                    db.close()
        except Exception as e:
            logger.error(f"Failed to cleanup old sessions: {e}")
            return 0
