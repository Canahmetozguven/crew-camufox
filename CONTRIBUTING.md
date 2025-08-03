# Contributing to Crew-Camufox

Thank you for your interest in contributing to Crew-Camufox! This document provides guidelines and information for contributors.

## 🌟 Ways to Contribute

- 🐛 **Bug Reports**: Report issues you encounter
- 💡 **Feature Requests**: Suggest new functionality
- 📝 **Documentation**: Improve docs and examples
- 🧪 **Testing**: Add tests and improve test coverage
- 🔧 **Code**: Fix bugs and implement features
- 🎨 **UI/UX**: Improve user experience
- 🌐 **Translation**: Add support for more languages

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **Node.js** (for Camoufox)
- **Git**
- **Ollama** (for local LLM support)

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR-USERNAME/crew-camufox.git
   cd crew-camufox
   ```

2. **Set Up Environment**
   ```bash
   # Use the setup script
   chmod +x scripts/setup.sh
   ./scripts/setup.sh

   # Or manual setup
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -e ".[dev,test]"
   ```

3. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

5. **Verify Setup**
   ```bash
   ./scripts/dev.sh health
   python -m pytest tests/ -v
   ```

## 🔧 Development Workflow

### Branch Naming Convention

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `test/description` - Test improvements
- `refactor/description` - Code refactoring

### Making Changes

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Run Quality Checks**
   ```bash
   # Run all quality checks
   ./scripts/dev.sh quality

   # Individual checks
   ./scripts/dev.sh lint      # Linting
   ./scripts/dev.sh format    # Code formatting
   ./scripts/dev.sh typecheck # Type checking
   ```

4. **Run Tests**
   ```bash
   # Quick tests
   ./scripts/dev.sh test

   # Full test suite
   python -m pytest tests/ -v --cov=src

   # Specific test categories
   python -m pytest tests/unit/ -v
   python -m pytest tests/integration/ -v
   ```

5. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

### Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Test additions/changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `style:` - Code style changes
- `chore:` - Maintenance tasks

Examples:
```bash
feat: add multi-engine search composition
fix: resolve browser session timeout issues
docs: update installation instructions
test: add integration tests for agent coordination
```

## 📋 Code Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with these specifications:

- **Line Length**: 100 characters
- **Indentation**: 4 spaces
- **Imports**: Organized with `isort`
- **Formatting**: Automated with `black`
- **Type Hints**: Required for all public functions

### Code Quality Tools

- **Black**: Code formatting
- **isort**: Import sorting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pytest**: Testing framework

### Architecture Guidelines

1. **Async/Await**: Use async programming for I/O operations
2. **Type Hints**: Full type annotation for better IDE support
3. **Error Handling**: Comprehensive exception management
4. **Logging**: Structured logging with appropriate levels
5. **Documentation**: Detailed docstrings following Google style

### Example Code Structure

```python
"""Module docstring describing the purpose."""

import asyncio
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ExampleModel(BaseModel):
    """Example model with proper type hints and validation."""
    
    name: str = Field(..., description="The name field")
    count: int = Field(default=0, ge=0, description="Non-negative count")
    metadata: Optional[Dict[str, str]] = None


async def example_function(
    param1: str,
    param2: Optional[int] = None
) -> Dict[str, Union[str, int]]:
    """
    Example async function with proper type hints.
    
    Args:
        param1: Description of parameter 1
        param2: Optional parameter with default
        
    Returns:
        Dictionary containing processed results
        
    Raises:
        ValueError: When param1 is invalid
    """
    if not param1:
        raise ValueError("param1 cannot be empty")
    
    # Implementation here
    return {"result": param1, "count": param2 or 0}
```

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for component interaction
├── performance/    # Performance and benchmark tests
├── fixtures/       # Test data and fixtures
└── conftest.py     # Pytest configuration
```

### Writing Tests

1. **Test File Naming**: `test_<module_name>.py`
2. **Test Function Naming**: `test_<function_name>_<scenario>`
3. **Async Tests**: Use `@pytest.mark.asyncio` for async functions
4. **Fixtures**: Reusable test data in `conftest.py`
5. **Mocking**: Mock external dependencies appropriately

### Test Example

```python
import pytest
from unittest.mock import AsyncMock, patch

from src.agents.research_coordinator import ResearchCoordinator


class TestResearchCoordinator:
    """Test suite for ResearchCoordinator."""
    
    @pytest.fixture
    async def coordinator(self):
        """Create a test coordinator instance."""
        coordinator = ResearchCoordinator()
        await coordinator.setup()
        yield coordinator
        await coordinator.cleanup()
    
    @pytest.mark.asyncio
    async def test_create_research_plan_success(self, coordinator):
        """Test successful research plan creation."""
        query = "test query"
        depth = "medium"
        
        plan = await coordinator.create_research_plan(query, depth)
        
        assert plan is not None
        assert plan.query == query
        assert plan.depth == depth
        assert len(plan.tasks) > 0
    
    @pytest.mark.asyncio
    async def test_create_research_plan_invalid_depth(self, coordinator):
        """Test research plan creation with invalid depth."""
        with pytest.raises(ValueError, match="Invalid depth"):
            await coordinator.create_research_plan("query", "invalid")
```

### Test Categories

Mark tests with appropriate categories:

```python
@pytest.mark.unit
def test_unit_functionality():
    """Unit test example."""
    pass

@pytest.mark.integration
@pytest.mark.requires_ollama
async def test_integration_with_ollama():
    """Integration test requiring Ollama."""
    pass

@pytest.mark.performance
def test_performance_benchmark():
    """Performance test example."""
    pass
```

## 📖 Documentation

### Documentation Types

1. **API Documentation**: Auto-generated from docstrings
2. **User Guides**: Step-by-step tutorials
3. **Developer Guides**: Architecture and design docs
4. **Examples**: Practical usage examples

### Writing Documentation

- **Clear and Concise**: Use simple, direct language
- **Examples**: Include practical code examples
- **Up-to-date**: Keep docs synchronized with code
- **Accessible**: Consider different skill levels

### Docstring Format

We use Google-style docstrings:

```python
def example_function(param1: str, param2: int = 0) -> bool:
    """
    Brief description of the function.
    
    Longer description if needed, explaining the purpose and behavior
    in more detail.
    
    Args:
        param1: Description of the first parameter.
        param2: Description of the second parameter with default value.
        
    Returns:
        Description of the return value.
        
    Raises:
        ValueError: When param1 is invalid.
        RuntimeError: When operation fails.
        
    Examples:
        >>> result = example_function("test", 5)
        >>> print(result)
        True
    """
```

## 🔍 Code Review Process

### Review Checklist

- [ ] **Functionality**: Does the code work as intended?
- [ ] **Tests**: Are there adequate tests with good coverage?
- [ ] **Documentation**: Is the code well-documented?
- [ ] **Style**: Does it follow our coding standards?
- [ ] **Performance**: Are there any performance concerns?
- [ ] **Security**: Are there any security issues?
- [ ] **Backwards Compatibility**: Does it break existing APIs?

### Review Guidelines

1. **Be Constructive**: Provide helpful, specific feedback
2. **Be Respectful**: Maintain a positive, collaborative tone
3. **Explain Reasoning**: Help others understand your suggestions
4. **Ask Questions**: Clarify unclear parts of the code
5. **Suggest Alternatives**: Offer better approaches when possible

## 🐛 Bug Reports

### Before Reporting

1. **Search Existing Issues**: Check if the bug is already reported
2. **Reproduce the Bug**: Ensure you can consistently reproduce it
3. **Gather Information**: Collect relevant system information

### Bug Report Template

```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.11.5]
- Crew-Camufox: [e.g., 2.0.0]
- Ollama: [e.g., 0.3.0]

**Additional Context**
- Error logs
- Screenshots
- Configuration details
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Feature Description**
A clear description of the feature you'd like to see.

**Problem Statement**
What problem does this feature solve?

**Proposed Solution**
How do you envision this feature working?

**Alternatives Considered**
What other approaches have you considered?

**Additional Context**
- Use cases
- Examples
- Related features
```

## 🚀 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

### Release Checklist

- [ ] Update version in `pyproject.toml`
- [ ] Update `CHANGELOG.md`
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Create release notes
- [ ] Tag release in Git
- [ ] Publish to PyPI

## 🤝 Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be:

- **Respectful**: Treat others with respect and courtesy
- **Inclusive**: Welcome people of all backgrounds and identities
- **Collaborative**: Work together constructively
- **Professional**: Maintain professional communication

### Getting Help

- **Documentation**: Check the official documentation first
- **Issues**: Search existing issues for similar problems
- **Discussions**: Use GitHub Discussions for questions
- **Community**: Join our community channels

### Recognition

Contributors are recognized in several ways:

- **Contributors List**: Listed in README.md
- **Release Notes**: Mentioned in release announcements  
- **Hall of Fame**: Featured contributors in documentation

## 📞 Contact

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and community discussion
- **Email**: For private matters or security issues

---

Thank you for contributing to Crew-Camufox! Your contributions help make this project better for everyone. 🚀
