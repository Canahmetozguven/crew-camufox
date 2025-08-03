"""
Configuration package for Crew-Camufox
"""

from .settings import (
    AppSettings,
    OllamaSettings,
    SearchSettings,
    BrowserSettings,
    CacheSettings,
    DatabaseSettings,
    ResearchConfig,
    LogLevel,
    ResearchDepth,
    ReportType,
    get_settings,
    reload_settings,
    get_ollama_settings,
    get_search_settings,
    get_browser_settings,
    get_cache_settings,
    get_database_settings,
)

__all__ = [
    "AppSettings",
    "OllamaSettings",
    "SearchSettings",
    "BrowserSettings",
    "CacheSettings",
    "DatabaseSettings",
    "ResearchConfig",
    "LogLevel",
    "ResearchDepth",
    "ReportType",
    "get_settings",
    "reload_settings",
    "get_ollama_settings",
    "get_search_settings",
    "get_browser_settings",
    "get_cache_settings",
    "get_database_settings",
]
