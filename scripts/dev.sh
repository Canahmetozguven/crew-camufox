#!/bin/bash
# Development workflow management script for Crew-Camufox

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

print_header() {
    echo -e "${PURPLE}================================================${NC}"
    echo -e "${PURPLE} $1${NC}"
    echo -e "${PURPLE}================================================${NC}"
}

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to show usage
show_usage() {
    echo "Crew-Camufox Development Workflow Manager"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Available commands:"
    echo "  setup          - Set up development environment"
    echo "  test           - Run test suite"
    echo "  test-unit      - Run unit tests only"
    echo "  test-integration - Run integration tests only"
    echo "  lint           - Run code linting (flake8)"
    echo "  format         - Format code (black + isort)"
    echo "  typecheck      - Run type checking (mypy)"
    echo "  quality        - Run all quality checks (lint + typecheck + format)"
    echo "  docs           - Generate documentation"
    echo "  clean          - Clean temporary files and caches"
    echo "  health         - Check system health"
    echo "  demo           - Run demo research mission"
    echo "  install        - Install/update dependencies"
    echo "  build          - Build project for distribution"
    echo "  help           - Show this help message"
    echo ""
}

# Ensure we're in project root
check_project_root() {
    if [[ ! -f "pyproject.toml" ]]; then
        print_error "Not in project root directory"
        exit 1
    fi
}

# Ensure virtual environment is active
ensure_venv() {
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_warning "Virtual environment not active"
        print_info "Activating virtual environment..."
        source .venv/bin/activate
    fi
}

# Setup command
cmd_setup() {
    print_header "Development Environment Setup"
    ./scripts/setup.sh
}

# Test commands
cmd_test() {
    print_header "Running Full Test Suite"
    ensure_venv
    
    print_info "Running pytest with coverage..."
    pytest --cov=src --cov-report=term-missing --cov-report=html --cov-report=xml
    
    print_status "Test suite completed"
    print_info "Coverage report available in htmlcov/index.html"
}

cmd_test_unit() {
    print_header "Running Unit Tests"
    ensure_venv
    
    pytest tests/unit/ -v --tb=short
}

cmd_test_integration() {
    print_header "Running Integration Tests"
    ensure_venv
    
    pytest tests/integration/ -v --tb=short
}

# Code quality commands
cmd_lint() {
    print_header "Running Code Linting"
    ensure_venv
    
    print_info "Running flake8..."
    flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W503
    print_status "Linting completed"
}

cmd_format() {
    print_header "Formatting Code"
    ensure_venv
    
    print_info "Running black..."
    black src/ tests/ --line-length=100
    
    print_info "Running isort..."
    isort src/ tests/ --profile=black --line-length=100
    
    print_status "Code formatting completed"
}

cmd_typecheck() {
    print_header "Running Type Checking"
    ensure_venv
    
    print_info "Running mypy..."
    mypy src/ --ignore-missing-imports --strict-optional --warn-unused-ignores
    print_status "Type checking completed"
}

cmd_quality() {
    print_header "Running All Quality Checks"
    
    print_info "Step 1/3: Code formatting..."
    cmd_format
    
    print_info "Step 2/3: Code linting..."
    cmd_lint
    
    print_info "Step 3/3: Type checking..."
    cmd_typecheck
    
    print_status "All quality checks completed"
}

# Documentation command
cmd_docs() {
    print_header "Generating Documentation"
    ensure_venv
    
    print_info "Generating API documentation with sphinx..."
    if [[ -f "docs/conf.py" ]]; then
        sphinx-build -b html docs/ docs/_build/html/
        print_status "Documentation generated in docs/_build/html/"
    else
        print_warning "Sphinx not configured, generating simple docs..."
        # Generate basic API docs
        mkdir -p docs/api
        python -c "
import pkgutil
import src
for importer, modname, ispkg in pkgutil.walk_packages(src.__path__, src.__name__ + '.'):
    print(f'Found module: {modname}')
"
        print_status "Basic documentation generated"
    fi
}

# Clean command
cmd_clean() {
    print_header "Cleaning Project"
    
    print_info "Removing Python cache files..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    
    print_info "Removing test artifacts..."
    rm -rf .pytest_cache/ htmlcov/ .coverage coverage.xml 2>/dev/null || true
    
    print_info "Removing temporary files..."
    rm -rf temp/ logs/*.log 2>/dev/null || true
    
    print_info "Removing build artifacts..."
    rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true
    
    print_status "Project cleaned"
}

# Health check command
cmd_health() {
    print_header "System Health Check"
    ensure_venv
    
    print_info "Checking Python environment..."
    python --version
    pip --version
    
    print_info "Checking key dependencies..."
    python -c "
import sys
import subprocess

packages = ['rich', 'pydantic', 'aiohttp', 'tenacity']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}: OK')
    except ImportError as e:
        print(f'❌ {pkg}: MISSING')
"
    
    print_info "Checking Ollama service..."
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_status "Ollama service: RUNNING"
        
        print_info "Available models:"
        curl -s http://localhost:11434/api/tags | python -c "
import json, sys
data = json.load(sys.stdin)
for model in data.get('models', []):
    print(f'  • {model[\"name\"]}')
" 2>/dev/null || echo "  (Unable to list models)"
    else
        print_warning "Ollama service: NOT RUNNING"
        print_info "Start Ollama with: ollama serve"
    fi
    
    print_info "Checking configuration..."
    python -c "
from src.config import get_settings
settings = get_settings()
print(f'✅ Config loaded: {settings.app_name} v{settings.version}')
print(f'  • Output directory: {settings.output_dir}')
print(f'  • Log level: {settings.log_level}')
print(f'  • Debug mode: {settings.debug}')
"
}

# Demo command
cmd_demo() {
    print_header "Running Demo Research Mission"
    ensure_venv
    
    print_info "Starting demo with sample query..."
    python -c "
import asyncio
from src.agents.multi_agent_orchestrator import main
asyncio.run(main())
"
}

# Install command
cmd_install() {
    print_header "Installing/Updating Dependencies"
    ensure_venv
    
    print_info "Updating dependencies with uv..."
    uv pip install -e ".[dev]" --upgrade
    print_status "Dependencies updated"
}

# Build command  
cmd_build() {
    print_header "Building Project"
    ensure_venv
    
    print_info "Running quality checks first..."
    cmd_quality
    
    print_info "Building distribution packages..."
    python -m build
    print_status "Build completed - packages in dist/"
}

# Main command dispatcher
main() {
    check_project_root
    
    case "${1:-help}" in
        setup)          cmd_setup ;;
        test)           cmd_test ;;
        test-unit)      cmd_test_unit ;;
        test-integration) cmd_test_integration ;;
        lint)           cmd_lint ;;
        format)         cmd_format ;;
        typecheck)      cmd_typecheck ;;
        quality)        cmd_quality ;;
        docs)           cmd_docs ;;
        clean)          cmd_clean ;;
        health)         cmd_health ;;
        demo)           cmd_demo ;;
        install)        cmd_install ;;
        build)          cmd_build ;;
        help|--help|-h) show_usage ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
