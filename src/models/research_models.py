from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ResearchDepth(str, Enum):
    SURFACE = "surface"  # 1-2 levels deep
    MEDIUM = "medium"  # 3-4 levels deep
    DEEP = "deep"  # 5+ levels deep


class SourceType(str, Enum):
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    NEWS = "news"
    ACADEMIC = "academic"
    SOCIAL_MEDIA = "social_media"
    FORUM = "forum"
    DOCUMENTATION = "documentation"
    OTHER = "other"


class ResearchSource(BaseModel):
    """Model for a research source"""

    url: str
    title: str
    content: str
    source_type: SourceType
    credibility_score: float = Field(ge=0.0, le=1.0)
    extraction_timestamp: datetime
    word_count: int
    key_topics: List[str] = []
    summary: str = ""


class ResearchQuery(BaseModel):
    """Model for research query parameters"""

    query: str
    focus_areas: List[str] = []
    exclude_domains: List[str] = []
    max_sources: int = Field(default=15, ge=1, le=100)
    depth: ResearchDepth = ResearchDepth.MEDIUM
    fact_check: bool = True
    include_social_media: bool = False
    date_range_days: Optional[int] = None


class FactCheck(BaseModel):
    """Model for fact-checking results"""

    claim: str
    verification_sources: List[str]
    confidence_score: float = Field(ge=0.0, le=1.0)
    status: str  # "verified", "disputed", "unverified"
    details: str


class ResearchInsight(BaseModel):
    """Model for research insights"""

    topic: str
    key_points: List[str]
    evidence: List[str]
    confidence_level: float = Field(ge=0.0, le=1.0)
    related_sources: List[str]


class ResearchReport(BaseModel):
    """Model for the final research report"""

    query: str
    executive_summary: str
    key_insights: List[ResearchInsight]
    sources: List[ResearchSource]
    fact_checks: List[FactCheck]
    methodology: str
    limitations: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    generated_at: datetime
    total_sources_analyzed: int
    research_depth_achieved: ResearchDepth


class AgentTask(BaseModel):
    """Model for agent tasks"""

    task_id: str
    agent_name: str
    description: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any] = {}
    status: str = "pending"  # pending, in_progress, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class BrowserSession(BaseModel):
    """Model for browser session tracking"""

    session_id: str
    created_at: datetime
    pages_visited: List[str] = []
    current_fingerprint: Dict[str, Any] = {}
    proxy_used: Optional[str] = None
    user_agent: str = ""
    is_active: bool = True
