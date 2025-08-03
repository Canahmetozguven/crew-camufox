# Installation Guide

This guide provides detailed instructions for installing Crew-Camufox in different environments.

## System Requirements

### Minimum Requirements
- **Python**: 3.11 or higher
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **OS**: Linux, macOS, or Windows

### Recommended Requirements
- **Python**: 3.12
- **RAM**: 8GB or more
- **Storage**: 5GB free space
- **CPU**: 4+ cores for optimal performance

## Installation Methods

### Method 1: Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/crew-camufox.git
cd crew-camufox

# Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh
```

The setup script will:
- Create a virtual environment
- Install all dependencies
- Set up pre-commit hooks
- Create necessary directories
- Copy environment configuration

### Method 2: Manual Installation

#### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/crew-camufox.git
cd crew-camufox
```

#### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Or using conda
conda create -n crew-camufox python=3.11
conda activate crew-camufox
```

#### Step 3: Install Dependencies
```bash
# Install with pip
pip install -e ".[dev]"

# Or install with uv (faster)
pip install uv
uv pip install -e ".[dev]"
```

#### Step 4: Setup Environment
```bash
# Copy environment template
cp .env.example .env

# Edit with your configuration
nano .env
```

### Method 3: Development Installation

For contributors and developers:

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/crew-camufox.git
cd crew-camufox

# Add upstream remote
git remote add upstream https://github.com/original-username/crew-camufox.git

# Install with development dependencies
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install
```

## Environment Configuration

### Required Configuration

Edit your `.env` file with the following required settings:

```bash
# Ollama Configuration (Required)
OLLAMA_MODEL=magistral:latest
OLLAMA_BASE_URL=http://localhost:11434

# Application Settings
OUTPUT_DIR=research_outputs
LOG_LEVEL=INFO
```

### Optional API Keys

For enhanced functionality, add these optional API keys:

```bash
# Search APIs (Optional but recommended)
GOOGLE_API_KEY=your_google_api_key
SERPER_API_KEY=your_serper_api_key
BRAVE_SEARCH_API_KEY=your_brave_api_key

# LLM APIs (Optional - for cloud models)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
```

## Installing Ollama

Crew-Camufox uses Ollama for local LLM support. Follow these steps:

### Linux/macOS
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve

# In another terminal, install required models
ollama pull magistral:latest
ollama pull llama3.1:8b
```

### Windows
1. Download Ollama from [https://ollama.ai/download](https://ollama.ai/download)
2. Install and run the application
3. Open Command Prompt and run:
```cmd
ollama pull magistral:latest
ollama pull llama3.1:8b
```

### Verify Ollama Installation
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Should return a JSON list of installed models
```

## Platform-Specific Instructions

### Ubuntu/Debian
```bash
# Install system dependencies
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip firefox-esr

# Install Node.js for Camoufox
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Continue with standard installation
```

### macOS
```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and Node.js
brew install python@3.11 node

# Continue with standard installation
```

### Windows
1. Install Python 3.11+ from [python.org](https://python.org)
2. Install Node.js from [nodejs.org](https://nodejs.org)
3. Install Git from [git-scm.com](https://git-scm.com)
4. Use PowerShell or Command Prompt for installation commands

## Docker Installation (Alternative)

### Using Docker Compose
```bash
# Clone repository
git clone https://github.com/your-username/crew-camufox.git
cd crew-camufox

# Build and run with Docker
docker-compose up --build
```

### Manual Docker Build
```bash
# Build image
docker build -t crew-camufox .

# Run container
docker run -it --rm \
  -v $(pwd)/research_outputs:/app/research_outputs \
  -v $(pwd)/.env:/app/.env \
  crew-camufox
```

## Verification

### Test Installation
```bash
# Basic functionality test
python enhanced_simple_runner.py --status

# Run health check
python -c "from src.config import get_settings; print('✅ Installation successful')"
```

### Run Example
```bash
# Simple research query
python enhanced_simple_runner.py "artificial intelligence trends"
```

## Troubleshooting

### Common Issues

#### Python Version Issues
```bash
# Check Python version
python --version

# Use specific Python version
python3.11 -m venv .venv
```

#### Permission Issues on Linux/macOS
```bash
# Fix script permissions
chmod +x scripts/*.sh

# Install with user permissions
pip install --user -e .
```

#### Ollama Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama service
ollama serve

# Check firewall settings if needed
```

#### Browser Issues
```bash
# Install Firefox ESR for better compatibility
sudo apt install firefox-esr  # Ubuntu/Debian
brew install firefox          # macOS
```

### Getting Help

If you encounter issues:

1. **Check the logs**: Look in `logs/` directory for error messages
2. **Run diagnostics**: Use `python enhanced_simple_runner.py --status`
3. **Check documentation**: Review relevant docs in `docs/`
4. **Search issues**: Look for similar problems in GitHub Issues
5. **Ask for help**: Create a new issue with detailed information

## Next Steps

After successful installation:

1. **Configuration**: Review and update your `.env` file
2. **First Run**: Try the basic examples in `examples/`
3. **Documentation**: Read the User Guide in `docs/user/`
4. **Contributing**: See `CONTRIBUTING.md` if you want to contribute

## Performance Optimization

### For Better Performance
```bash
# Install uvloop for faster async operations
pip install uvloop

# Use SSD storage for better I/O
# Ensure adequate RAM (8GB+ recommended)

# Configure browser settings for headless operation
BROWSER_HEADLESS=true
BROWSER_STEALTH=true
```

### Resource Management
```bash
# Limit concurrent operations
MAX_CONCURRENT_MISSIONS=2
BROWSER_MAX_PAGES=3

# Enable caching
CACHE_ENABLED=true
CACHE_DEFAULT_TTL=3600
```

---

For more detailed configuration options, see the [Configuration Guide](configuration.md).
