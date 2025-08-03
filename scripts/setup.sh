#!/bin/bash
# Development environment setup script for Crew-Camufox

set -e  # Exit on any error

echo "🚀 Setting up Crew-Camufox development environment..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Check for Python 3.11+
print_info "Checking Python version..."
if ! python3 --version | grep -E "Python 3\.(11|12|13)" > /dev/null; then
    print_error "Python 3.11+ is required"
    echo "Please install Python 3.11 or newer"
    exit 1
fi
print_status "Python version check passed"

# Install uv if not present
print_info "Checking for uv package manager..."
if ! command -v uv &> /dev/null; then
    print_warning "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    
    if ! command -v uv &> /dev/null; then
        print_error "Failed to install uv. Please install manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
fi
print_status "uv package manager ready"

# Create virtual environment and install dependencies
print_info "Creating virtual environment..."
if [[ ! -d ".venv" ]]; then
    uv venv
    print_status "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
print_info "Installing project dependencies..."
uv pip install -e ".[dev]"
print_status "Dependencies installed"

# Install pre-commit hooks
print_info "Setting up pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    print_status "Pre-commit hooks installed"
else
    print_warning "Pre-commit not available in PATH, skipping hook installation"
fi

# Create necessary directories
print_info "Creating project directories..."
directories=(
    "logs"
    "research_outputs"  
    "temp"
    "tests/fixtures"
    "docs/api"
    "docs/user"
    "docs/development"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
done
print_status "Project directories created"

# Copy environment template if .env doesn't exist
print_info "Setting up environment configuration..."
if [[ ! -f ".env" ]]; then
    cp .env.example .env
    print_status "Environment file created from template"
    print_warning "Please edit .env with your configuration (API keys, etc.)"
else
    print_warning ".env file already exists"
fi

# Initialize database (if applicable)
print_info "Checking database setup..."
if [[ -f "alembic.ini" ]]; then
    print_info "Running database migrations..."
    alembic upgrade head
    print_status "Database migrations completed"
else
    print_warning "No Alembic configuration found, skipping database setup"
fi

# Run basic health checks
print_info "Running health checks..."

# Check if Ollama is running
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    print_status "Ollama service is running"
else
    print_warning "Ollama service not detected at localhost:11434"
    print_info "Please start Ollama: https://ollama.ai/download"
fi

# Check if required models are available
print_info "Checking for required LLM models..."
if curl -s http://localhost:11434/api/tags | grep -q "magistral"; then
    print_status "Magistral model found"
else
    print_warning "Magistral model not found"
    print_info "Install with: ollama pull magistral:latest"
fi

# Create a simple test script
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""
Quick setup test for Crew-Camufox
"""

def test_imports():
    """Test that all main modules can be imported"""
    try:
        from src.config import get_settings
        from src.utils import get_logger
        from src.agents.multi_agent_orchestrator import MultiAgentResearchOrchestrator
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    try:
        from src.config import get_settings
        settings = get_settings()
        print(f"✅ Configuration loaded - Output dir: {settings.output_dir}")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_logging():
    """Test logging system"""
    try:
        from src.utils import get_logger
        logger = get_logger("test")
        logger.info("Test log message")
        print("✅ Logging system working")
        return True
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Running setup tests...")
    
    tests = [test_imports, test_config, test_logging]
    results = [test() for test in tests]
    
    if all(results):
        print("\n🎉 All tests passed! Setup completed successfully.")
        print("\nNext steps:")
        print("1. Edit .env with your API keys")
        print("2. Start Ollama and install models")
        print("3. Run: python -m src.main --help")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
EOF

# Run the test
print_info "Running setup validation..."
python3 test_setup.py

# Clean up test file
rm test_setup.py

print_status "Development environment setup complete!"

echo ""
echo "🎯 Quick Start:"
echo "1. Activate environment: source .venv/bin/activate"
echo "2. Edit configuration: nano .env"
echo "3. Start development: python enhanced_simple_runner.py --help"
echo ""
echo "📚 Development commands:"
echo "• Run tests: ./scripts/dev.sh test"
echo "• Format code: ./scripts/dev.sh format"
echo "• Type check: ./scripts/dev.sh typecheck"
echo "• Lint code: ./scripts/dev.sh lint"
echo "• Quality checks: ./scripts/dev.sh quality"
echo ""
echo "🤝 Contributing:"
echo "• Read CONTRIBUTING.md for guidelines"
echo "• Create feature branch: git checkout -b feature/your-feature"
echo "• Make changes and test: ./scripts/dev.sh quality"
echo "• Submit pull request"
echo ""
echo "📖 Documentation:"
echo "• README.md - Project overview"
echo "• CONTRIBUTING.md - How to contribute"
echo "• docs/ - Detailed documentation"
echo ""
echo "Happy coding! 🚀"
