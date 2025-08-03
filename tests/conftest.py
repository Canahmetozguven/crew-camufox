"""
Test configuration and fixtures for Crew-Camufox tests
"""

import pytest
import asyncio
import tempfile
import shutil
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from typing import Generator, Dict, Any

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.config import AppSettings, OllamaSettings, SearchSettings, BrowserSettings

    CONFIG_AVAILABLE = True
except ImportError:
    # Fallback for when src modules are not available
    CONFIG_AVAILABLE = False

    # Mock configuration classes
    class AppSettings:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class OllamaSettings:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class SearchSettings:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class BrowserSettings:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def test_settings(temp_dir: Path) -> AppSettings:
    """Create test configuration settings"""
    return AppSettings(
        debug=True,
        output_dir=str(temp_dir / "outputs"),
        logs_dir=str(temp_dir / "logs"),
        temp_dir=str(temp_dir / "temp"),
        log_level="DEBUG",
        max_concurrent_missions=1,
        mission_timeout=300,  # 5 minutes for tests
        ollama=OllamaSettings(
            model_name="test-model",
            base_url="http://test-ollama:11434",
            browser_model="test-browser-model",
            timeout=10,
        ),
        search=SearchSettings(
            max_sources=5,
            request_delay=0.1,
            timeout=10,
            enable_academic_search=False,  # Disable for testing
            enable_direct_browser=False,
        ),
        browser=BrowserSettings(
            headless=True, page_timeout=10, navigation_delay=0.1, max_concurrent_pages=1
        ),
    )


@pytest.fixture
def mock_ollama_client():
    """Mock Ollama client for testing"""
    mock = AsyncMock()
    mock.invoke.return_value.content = "Mock LLM response for testing"
    return mock


@pytest.fixture
def mock_browser():
    """Mock browser for testing"""
    mock = MagicMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=None)
    return mock


@pytest.fixture
def sample_search_results() -> list:
    """Sample search results for testing"""
    return [
        {
            "title": "Test Article 1",
            "url": "https://example.com/article1",
            "snippet": "This is a test article about testing methodologies.",
            "source": "test",
        },
        {
            "title": "Test Article 2",
            "url": "https://example.com/article2",
            "snippet": "Another test article with different content.",
            "source": "test",
        },
    ]


@pytest.fixture
def sample_research_plan() -> Dict[str, Any]:
    """Sample research plan for testing"""
    return {
        "id": "test-plan-001",
        "query": "Test research query",
        "config": {
            "max_time_minutes": 15,
            "max_sources": 5,
            "search_rounds": 1,
            "deep_dive_sources": 2,
        },
        "execution_phases": [
            {
                "phase": 1,
                "name": "Initial Discovery",
                "duration_minutes": 10,
                "activities": ["Execute primary search terms"],
            }
        ],
        "search_strategies": {
            "primary_terms": ["test", "research", "methodology"],
            "secondary_terms": ["testing", "analysis"],
        },
    }


@pytest.fixture
def sample_research_results() -> Dict[str, Any]:
    """Sample research results for testing"""
    return {
        "query": "Test research query",
        "sources": [
            {
                "url": "https://example.com/article1",
                "title": "Test Article 1",
                "content": "Test content for article 1",
                "credibility_score": 0.8,
                "relevance_score": 0.9,
            }
        ],
        "summary": "Test research summary",
        "key_findings": ["Finding 1", "Finding 2"],
        "confidence_score": 0.85,
    }


@pytest.fixture
def sample_final_report() -> Dict[str, Any]:
    """Sample final report for testing"""
    return {
        "report_id": "test-report-001",
        "query": "Test research query",
        "executive_summary": "Test executive summary",
        "sections": {
            "introduction": "Test introduction",
            "methodology": "Test methodology",
            "findings": "Test findings",
            "conclusion": "Test conclusion",
        },
        "sources": ["https://example.com/article1"],
        "formatted_outputs": {
            "markdown": "# Test Report\n\nTest content",
            "text": "Test Report\nTest content",
            "json": '{"test": "data"}',
        },
        "quality_assessment": {"overall_score": 0.85, "estimated_reading_time": 5},
    }


@pytest.fixture
def mock_health_checker():
    """Mock health checker for testing"""
    mock = AsyncMock()
    mock.check_ollama_health.return_value = True
    mock.check_redis_health.return_value = True
    mock.check_all_dependencies.return_value = {"ollama": True, "redis": True}
    mock.get_health_status.return_value = {
        "ollama": {"healthy": True, "last_check": 1234567890},
        "redis": {"healthy": True, "last_check": 1234567890},
    }
    return mock


@pytest.fixture
def caplog_handler():
    """Capture log output for testing"""
    import logging
    from io import StringIO

    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    logger = logging.getLogger("crew-camufox")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    yield log_capture

    logger.removeHandler(handler)


@pytest.fixture(autouse=True)
def patch_settings(test_settings, monkeypatch):
    """Automatically patch settings for all tests"""

    def mock_get_settings():
        return test_settings

    monkeypatch.setattr("src.config.get_settings", mock_get_settings)
    monkeypatch.setattr("src.config.settings.get_settings", mock_get_settings)


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "requires_ollama: marks tests that need Ollama running")
    config.addinivalue_line(
        "markers", "requires_internet: marks tests that need internet connection"
    )
    # Phase 3 Testing & Quality Assurance markers
    config.addinivalue_line("markers", "performance: marks tests as performance benchmarks")
    config.addinivalue_line("markers", "concurrency: marks tests as concurrency tests")
    config.addinivalue_line("markers", "scalability: marks tests as scalability tests")
    config.addinivalue_line("markers", "stress: marks tests as stress tests")
    config.addinivalue_line("markers", "reliability: marks tests as reliability tests")
