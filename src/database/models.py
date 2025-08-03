"""Database models for research data persistence."""

from datetime import datetime
from typing import Dict, Optional, Any

try:
    from sqlalchemy import (
        Column,
        Integer,
        String,
        Text,
        DateTime,
        JSON,
        ForeignKey,
        Float,
        create_engine,
    )
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import relationship, sessionmaker
    from sqlalchemy.sql import func

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

from pydantic import BaseModel
from src.config.settings import get_settings


# Pydantic models for data validation
class ResearchSessionCreate(BaseModel):
    """Create schema for research session."""

    mission_id: str
    query: str
    status: str = "active"
    extra_data: Optional[Dict[str, Any]] = None


class ResearchSessionUpdate(BaseModel):
    """Update schema for research session."""

    status: Optional[str] = None
    completed_at: Optional[datetime] = None
    extra_data: Optional[Dict[str, Any]] = None


class SearchResultCreate(BaseModel):
    """Create schema for search result."""

    session_id: int
    url: str
    title: str
    content: str
    source: str
    relevance_score: Optional[float] = None
    extra_data: Optional[Dict[str, Any]] = None


class ResearchPlanCreate(BaseModel):
    """Create schema for research plan."""

    session_id: int
    plan_data: Dict[str, Any]
    agent_type: str
    version: int = 1


class ResearchOutputCreate(BaseModel):
    """Create schema for research output."""

    session_id: int
    output_type: str
    content: str
    file_path: Optional[str] = None
    extra_data: Optional[Dict[str, Any]] = None


# SQLAlchemy models (if available)
if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()

    class ResearchSessionDB(Base):
        """Research session model."""

        __tablename__ = "research_sessions"

        id = Column(Integer, primary_key=True, index=True)
        mission_id = Column(String(255), unique=True, index=True)
        query = Column(Text, nullable=False)
        status = Column(String(50), default="active")
        created_at = Column(DateTime(timezone=True), server_default=func.now())
        completed_at = Column(DateTime(timezone=True), nullable=True)
        extra_data = Column(JSON, nullable=True)

        # Relationships
        search_results = relationship(
            "SearchResultDB", back_populates="session", cascade="all, delete-orphan"
        )
        research_plans = relationship(
            "ResearchPlanDB", back_populates="session", cascade="all, delete-orphan"
        )
        research_outputs = relationship(
            "ResearchOutputDB", back_populates="session", cascade="all, delete-orphan"
        )

    class SearchResultDB(Base):
        """Search result model."""

        __tablename__ = "search_results"

        id = Column(Integer, primary_key=True, index=True)
        session_id = Column(Integer, ForeignKey("research_sessions.id"))
        url = Column(Text, nullable=False)
        title = Column(Text, nullable=False)
        content = Column(Text, nullable=False)
        source = Column(String(100), nullable=False)
        relevance_score = Column(Float, nullable=True)
        extracted_at = Column(DateTime(timezone=True), server_default=func.now())
        extra_data = Column(JSON, nullable=True)

        # Relationships
        session = relationship("ResearchSessionDB", back_populates="search_results")

    class ResearchPlanDB(Base):
        """Research plan model."""

        __tablename__ = "research_plans"

        id = Column(Integer, primary_key=True, index=True)
        session_id = Column(Integer, ForeignKey("research_sessions.id"))
        plan_data = Column(JSON, nullable=False)
        agent_type = Column(String(100), nullable=False)
        version = Column(Integer, default=1)
        created_at = Column(DateTime(timezone=True), server_default=func.now())

        # Relationships
        session = relationship("ResearchSessionDB", back_populates="research_plans")

    class ResearchOutputDB(Base):
        """Research output model."""

        __tablename__ = "research_outputs"

        id = Column(Integer, primary_key=True, index=True)
        session_id = Column(Integer, ForeignKey("research_sessions.id"))
        output_type = Column(String(50), nullable=False)  # report, summary, analysis
        content = Column(Text, nullable=False)
        file_path = Column(String(500), nullable=True)
        created_at = Column(DateTime(timezone=True), server_default=func.now())
        extra_data = Column(JSON, nullable=True)

        # Relationships
        session = relationship("ResearchSessionDB", back_populates="research_outputs")

    # Aliases for easier use
    ResearchSession = ResearchSessionDB
    SearchResult = SearchResultDB
    ResearchPlan = ResearchPlanDB
    ResearchOutput = ResearchOutputDB

else:
    # Fallback models for when SQLAlchemy is not available
    class ResearchSession(BaseModel):
        """Research session fallback model."""

        id: Optional[int] = None
        mission_id: str
        query: str
        status: str = "active"
        created_at: Optional[datetime] = None
        completed_at: Optional[datetime] = None
        extra_data: Optional[Dict[str, Any]] = None

    class SearchResult(BaseModel):
        """Search result fallback model."""

        id: Optional[int] = None
        session_id: int
        url: str
        title: str
        content: str
        source: str
        relevance_score: Optional[float] = None
        extracted_at: Optional[datetime] = None
        extra_data: Optional[Dict[str, Any]] = None

    class ResearchPlan(BaseModel):
        """Research plan fallback model."""

        id: Optional[int] = None
        session_id: int
        plan_data: Dict[str, Any]
        agent_type: str
        version: int = 1
        created_at: Optional[datetime] = None

    class ResearchOutput(BaseModel):
        """Research output fallback model."""

        id: Optional[int] = None
        session_id: int
        output_type: str
        content: str
        file_path: Optional[str] = None
        created_at: Optional[datetime] = None
        extra_data: Optional[Dict[str, Any]] = None


# Database connection factory
def create_database_engine(database_url: Optional[str] = None):
    """Create database engine based on configuration."""
    if not SQLALCHEMY_AVAILABLE:
        raise ImportError("SQLAlchemy is not available. Install with: uv add sqlalchemy")

    settings = get_settings()
    if database_url is None:
        database_url = settings.database.url

    engine = create_engine(
        database_url,
        echo=settings.database.echo,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_timeout=settings.database.pool_timeout,
        pool_recycle=settings.database.pool_recycle,
    )
    return engine


def create_tables(engine):
    """Create all database tables."""
    if SQLALCHEMY_AVAILABLE:
        Base.metadata.create_all(bind=engine)


def get_session_maker(engine):
    """Get SQLAlchemy session maker."""
    if not SQLALCHEMY_AVAILABLE:
        raise ImportError("SQLAlchemy is not available")

    return sessionmaker(autocommit=False, autoflush=False, bind=engine)
