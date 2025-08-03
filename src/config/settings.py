"""
Configuration management system for Crew-Camufox
Centralized settings with Pydantic validation and environment variable support
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class LogLevel(str, Enum):
    """Available log levels"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ResearchDepth(str, Enum):
    """Research depth levels"""

    SURFACE = "surface"
    MEDIUM = "medium"
    DEEP = "deep"
    EXHAUSTIVE = "exhaustive"


class ReportType(str, Enum):
    """Report output types"""

    COMPREHENSIVE = "comprehensive"
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    SUMMARY = "summary"


class OllamaSettings(BaseModel):
    """Ollama/LLM configuration settings"""

    model_name: str = "magistral:latest"
    base_url: str = "http://localhost:11434"
    browser_model: str = "granite3.3:8b"
    timeout: int = 30
    max_retries: int = 3
    temperature: float = 0.1

    @classmethod
    def from_env(cls) -> "OllamaSettings":
        """Create settings from environment variables"""
        return cls(
            model_name=os.getenv("OLLAMA_MODEL", "magistral:latest"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            browser_model=os.getenv("BROWSER_MODEL", "granite3.3:8b"),
            timeout=int(os.getenv("OLLAMA_TIMEOUT", "30")),
            max_retries=int(os.getenv("OLLAMA_MAX_RETRIES", "3")),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
        )


class SearchSettings(BaseModel):
    """Search engine configuration settings"""

    # API Keys
    google_api_key: Optional[str] = None
    google_cse_id: Optional[str] = None
    bing_api_key: Optional[str] = None
    serp_api_key: Optional[str] = None

    # Search Configuration
    max_sources: int = 25  # Increased from 15 for more comprehensive reports
    request_delay: float = 1.0
    timeout: int = 30

    # Search Strategies
    enable_academic_search: bool = True
    enable_direct_browser: bool = True
    fallback_engines: List[str] = ["duckduckgo", "bing", "google"]

    @classmethod
    def from_env(cls) -> "SearchSettings":
        """Create settings from environment variables"""
        return cls(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            google_cse_id=os.getenv("GOOGLE_CSE_ID"),
            bing_api_key=os.getenv("BING_API_KEY"),
            serp_api_key=os.getenv("SERP_API_KEY"),
            max_sources=int(os.getenv("MAX_SOURCES", "25")),  # Increased default
            request_delay=float(os.getenv("REQUEST_DELAY", "1.0")),
            timeout=int(os.getenv("SEARCH_TIMEOUT", "30")),
            enable_academic_search=os.getenv("ENABLE_ACADEMIC_SEARCH", "true").lower() == "true",
            enable_direct_browser=os.getenv("ENABLE_DIRECT_BROWSER", "true").lower() == "true",
        )


class BrowserSettings(BaseModel):
    """Browser/Camoufox configuration settings"""

    headless: bool = True
    stealth_mode: bool = True
    user_agent_rotation: bool = True
    proxy_rotation: bool = False

    # Performance Settings
    page_timeout: int = 30
    navigation_delay: float = 2.0
    max_concurrent_pages: int = 3

    @classmethod
    def from_env(cls) -> "BrowserSettings":
        """Create settings from environment variables"""
        return cls(
            headless=os.getenv("BROWSER_HEADLESS", "true").lower() == "true",
            stealth_mode=os.getenv("BROWSER_STEALTH", "true").lower() == "true",
            user_agent_rotation=os.getenv("BROWSER_UA_ROTATION", "true").lower() == "true",
            proxy_rotation=os.getenv("BROWSER_PROXY_ROTATION", "false").lower() == "true",
            page_timeout=int(os.getenv("BROWSER_PAGE_TIMEOUT", "30")),
            navigation_delay=float(os.getenv("BROWSER_NAV_DELAY", "2.0")),
            max_concurrent_pages=int(os.getenv("BROWSER_MAX_PAGES", "3")),
        )


class CacheSettings(BaseModel):
    """Caching configuration settings"""

    enabled: bool = True
    redis_url: Optional[str] = None
    default_ttl: int = 3600  # 1 hour
    search_results_ttl: int = 7200  # 2 hours
    max_memory_cache_size: int = 1000

    @classmethod
    def from_env(cls) -> "CacheSettings":
        """Create settings from environment variables"""
        return cls(
            enabled=os.getenv("CACHE_ENABLED", "true").lower() == "true",
            redis_url=os.getenv("REDIS_URL"),
            default_ttl=int(os.getenv("CACHE_DEFAULT_TTL", "3600")),
            search_results_ttl=int(os.getenv("CACHE_SEARCH_TTL", "7200")),
            max_memory_cache_size=int(os.getenv("CACHE_MAX_MEMORY_SIZE", "1000")),
        )


class DatabaseSettings(BaseModel):
    """Database configuration settings"""

    url: str = "sqlite:///crew_camufox.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600

    @classmethod
    def from_env(cls) -> "DatabaseSettings":
        """Create settings from environment variables"""
        return cls(
            url=os.getenv("DATABASE_URL", "sqlite:///crew_camufox.db"),
            echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
            pool_size=int(os.getenv("DATABASE_POOL_SIZE", "5")),
            max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", "10")),
            pool_timeout=int(os.getenv("DATABASE_POOL_TIMEOUT", "30")),
            pool_recycle=int(os.getenv("DATABASE_POOL_RECYCLE", "3600")),
        )


class ResearchConfig:
    """Research depth configurations"""

    DEPTH_CONFIGS = {
        ResearchDepth.SURFACE: {
            "max_time_minutes": 20,  # Increased time
            "max_sources": 12,       # Increased from 8
            "sources_per_round": 4,  # Increased from 3
            "deep_dive_sources": 4,  # Increased from 2
            "search_rounds": 2,      # Increased from 1
            "fact_check_enabled": False,
            "description": "Quick overview research",
        },
        ResearchDepth.MEDIUM: {
            "max_time_minutes": 45,  # Increased time
            "max_sources": 18,       # Increased from 12
            "sources_per_round": 6,  # Increased from 4
            "deep_dive_sources": 8,  # Increased from 4
            "search_rounds": 3,      # Increased from 2
            "fact_check_enabled": True,
            "description": "Balanced analysis",
        },
        ResearchDepth.DEEP: {
            "max_time_minutes": 75,  # Increased time
            "max_sources": 25,       # Increased from 15
            "sources_per_round": 8,  # Increased from 5
            "deep_dive_sources": 12, # Increased from 6
            "search_rounds": 4,      # Increased from 3
            "fact_check_enabled": True,
            "description": "Thorough investigation",
        },
        ResearchDepth.EXHAUSTIVE: {
            "max_time_minutes": 150, # Increased time
            "max_sources": 35,       # Increased from 20
            "sources_per_round": 10, # Increased from 7
            "deep_dive_sources": 15, # Increased from 8
            "search_rounds": 5,      # Increased from 4
            "fact_check_enabled": True,
            "description": "Comprehensive study",
        },
    }

    @classmethod
    def get_config(cls, depth: ResearchDepth) -> Dict[str, Any]:
        """Get configuration for research depth level"""
        return cls.DEPTH_CONFIGS.get(depth, cls.DEPTH_CONFIGS[ResearchDepth.MEDIUM])


class AppSettings(BaseModel):
    """Main application settings."""

    # Core settings
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    log_to_file: bool = False
    logs_dir: str = "logs"
    log_rotation: bool = True

    # Component settings
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    browser: BrowserSettings = Field(default_factory=BrowserSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)

    @classmethod
    def from_env(cls) -> "AppSettings":
        """Create settings from environment variables"""
        return cls(
            environment=os.getenv("ENVIRONMENT", "development"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_to_file=os.getenv("LOG_TO_FILE", "false").lower() == "true",
            logs_dir=os.getenv("LOGS_DIR", "logs"),
            log_rotation=os.getenv("LOG_ROTATION", "true").lower() == "true",
            ollama=OllamaSettings.from_env(),
            search=SearchSettings.from_env(),
            browser=BrowserSettings.from_env(),
            cache=CacheSettings.from_env(),
            database=DatabaseSettings.from_env(),
        )


# Global settings instance
_settings: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    """Get application settings singleton"""
    global _settings
    if _settings is None:
        # Try to load from environment first, then use defaults
        try:
            _settings = AppSettings.from_env()
        except Exception:
            _settings = AppSettings()
    return _settings


def reload_settings() -> AppSettings:
    """Reload settings from environment/config files"""
    global _settings
    try:
        _settings = AppSettings.from_env()
    except Exception:
        _settings = AppSettings()
    return _settings


# Convenience functions
def get_ollama_settings() -> OllamaSettings:
    """Get Ollama configuration"""
    return get_settings().ollama


def get_search_settings() -> SearchSettings:
    """Get search configuration"""
    return get_settings().search


def get_browser_settings() -> BrowserSettings:
    """Get browser configuration"""
    return get_settings().browser


def get_cache_settings() -> CacheSettings:
    """Get cache configuration"""
    return get_settings().cache


def get_database_settings() -> DatabaseSettings:
    """Get database configuration"""
    return get_settings().database
