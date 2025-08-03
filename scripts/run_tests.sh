#!/bin/bash

# Test runner script for crew-camufox
# Runs unit tests for Phase 1 infrastructure components

echo "🧪 Running Unit Tests for Phase 1 Infrastructure..."
echo "================================================="

# Navigate to project directory
cd /home/canahmet/Desktop/projects/crew-camufox

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv package manager not found"
    echo "Please install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Install dependencies if not already installed
echo "📦 Installing dependencies with uv..."
if ! uv sync --quiet; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Run specific test suites
echo ""
echo "🔧 Testing Configuration Management..."
if uv run python -m pytest tests/unit/test_config.py -v; then
    echo "✅ Configuration tests passed"
else
    echo "❌ Configuration tests failed"
fi

echo ""
echo "📝 Testing Logging System..."
if uv run python -m pytest tests/unit/test_logging.py -v; then
    echo "✅ Logging tests passed"
else
    echo "❌ Logging tests failed"
fi

echo ""
echo "🛡️ Testing Resilience Patterns..."
if uv run python -m pytest tests/unit/test_resilience.py -v; then
    echo "✅ Resilience tests passed"
else
    echo "❌ Resilience tests failed"
fi

echo ""
echo "🤖 Testing Orchestrator Integration..."
if uv run python -m pytest tests/unit/test_orchestrator.py -v; then
    echo "✅ Orchestrator tests passed"
else
    echo "❌ Orchestrator tests failed"
fi

echo ""
echo "📊 Running All Tests Together..."
if uv run python -m pytest tests/unit/ -v --tb=short; then
    echo ""
    echo "🎉 All Phase 1 Infrastructure Tests Completed Successfully!"
    echo "✅ Configuration Management: Working"
    echo "✅ Enhanced Logging: Working"  
    echo "✅ Resilience Patterns: Working"
    echo "✅ Orchestrator Integration: Working"
    echo ""
    echo "🚀 Phase 1 Infrastructure & Foundation is complete!"
    echo "Ready to proceed to Phase 2: Performance & Scalability"
else
    echo ""
    echo "❌ Some tests failed. Please review the output above."
    echo "💡 Tip: Run individual test files to debug specific issues"
fi

echo ""
echo "📁 Test artifacts location: tests/unit/"
echo "🔍 For detailed test output, run: uv run python -m pytest tests/unit/ -v --tb=long"
