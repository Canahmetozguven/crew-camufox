# Quick Start Guide

Get up and running with Crew-Camufox in 5 minutes!

## 🚀 Rapid Setup

### Prerequisites Check
```bash
# Check Python version (3.11+ required)
python --version

# Check if you have git
git --version
```

### 1-Command Installation
```bash
# Clone and setup everything
git clone https://github.com/your-username/crew-camufox.git && \
cd crew-camufox && \
chmod +x scripts/setup.sh && \
./scripts/setup.sh
```

## ⚡ First Research Query

### Activate Environment
```bash
# Activate the virtual environment
source .venv/bin/activate
```

### Run Your First Query
```bash
# Simple research query
python enhanced_simple_runner.py "latest artificial intelligence developments"
```

Expected output:
```
🚀 Crew-Camufox: Advanced Multi-Agent Research Platform
🎯 Research Query: latest artificial intelligence developments
🔍 Analyzing query and planning research strategy...
📊 Executing research mission...
✅ Research completed successfully!
📋 Results saved to: research_outputs/enhanced_mission_[timestamp]_complete.json
```

## 🎯 Essential Commands

### Basic Research
```bash
# Simple query
python enhanced_simple_runner.py "your research topic"

# With depth control
python enhanced_simple_runner.py "climate change solutions" --depth deep

# With focus areas
python enhanced_simple_runner.py "machine learning" --focus "healthcare applications"

# Show browser (for debugging)
python enhanced_simple_runner.py "quantum computing" --show-browser
```

### System Status
```bash
# Check system health
python enhanced_simple_runner.py --status

# List available models
python enhanced_simple_runner.py --list-models
```

## 📁 Understanding Output

Research results are saved in `research_outputs/` with three files:

1. **`*_complete.json`** - Full structured data
2. **`*_report.md`** - Human-readable report
3. **`*_report.txt`** - Plain text summary

### Quick View Results
```bash
# View latest report
ls -lt research_outputs/*.md | head -1 | awk '{print $9}' | xargs cat
```

## 🛠️ Common Configurations

### Edit Configuration
```bash
# Open environment file
nano .env
```

### Essential Settings
```bash
# For better performance
BROWSER_HEADLESS=true
CACHE_ENABLED=true
MAX_SOURCES=20

# For development/debugging
BROWSER_HEADLESS=false
LOG_LEVEL=DEBUG
DEBUG=true
```

## 🔧 Ollama Setup

Crew-Camufox requires Ollama for local AI processing:

### Install Ollama
```bash
# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# Start service
ollama serve
```

### Install Required Models
```bash
# Required model
ollama pull magistral:latest

# Optional but recommended
ollama pull llama3.1:8b
ollama pull granite3.3:8b
```

### Verify Ollama
```bash
# Check if running
curl http://localhost:11434/api/tags

# Should return JSON with model list
```

## 🎮 Interactive Mode

For guided research sessions:

```bash
# Start interactive mode
python enhanced_simple_runner.py

# Follow the prompts:
# 1. Enter your research query
# 2. Choose research depth (surface/medium/deep)
# 3. Set maximum sources
# 4. Add focus areas (optional)
# 5. Add exclusions (optional)
```

## 📊 Research Examples

### Academic Research
```bash
python enhanced_simple_runner.py \
  "CRISPR gene editing clinical trials 2024" \
  --depth deep \
  --focus "peer-reviewed studies" "FDA approval" \
  --max-sources 25
```

### Market Analysis
```bash
python enhanced_simple_runner.py \
  "electric vehicle market trends" \
  --depth medium \
  --focus "sales data" "market share" \
  --exclude "opinion pieces"
```

### Technical Documentation
```bash
python enhanced_simple_runner.py \
  "kubernetes security best practices" \
  --depth deep \
  --focus "official documentation" "security guides" \
  --exclude "forums" "social media"
```

### News Analysis
```bash
python enhanced_simple_runner.py \
  "renewable energy policy 2024" \
  --depth medium \
  --focus "government sources" "policy documents"
```

## 🐛 Quick Troubleshooting

### Common Issues & Fixes

#### "Ollama not found"
```bash
# Start Ollama service
ollama serve

# Check if running
curl http://localhost:11434/api/tags
```

#### "Permission denied on scripts"
```bash
# Fix script permissions
chmod +x scripts/*.sh
```

#### "Module not found"
```bash
# Activate virtual environment
source .venv/bin/activate

# Reinstall if needed
pip install -e ".[dev]"
```

#### "Browser fails to start"
```bash
# Try headless mode
export BROWSER_HEADLESS=true

# Or install Firefox ESR
sudo apt install firefox-esr  # Ubuntu/Debian
brew install firefox          # macOS
```

### Check System Health
```bash
# Comprehensive health check
python enhanced_simple_runner.py --status

# Should show all systems as "✅ Active"
```

## 🎓 Next Steps

### Learn More
1. **Read the full documentation**: `docs/user/`
2. **Try examples**: `examples/` directory
3. **Configure APIs**: Add search API keys for better results
4. **Explore templates**: Use research templates for specific domains

### Advanced Usage
```bash
# Use research templates
python enhanced_simple_runner.py \
  "AI ethics guidelines" \
  --template academic \
  --depth comprehensive

# Multiple focus areas
python enhanced_simple_runner.py \
  "sustainable energy" \
  --focus "solar power" "wind energy" "storage solutions" \
  --exclude "fossil fuels"
```

### Configuration Optimization
```bash
# Performance tuning
MAX_CONCURRENT_MISSIONS=2
BROWSER_MAX_PAGES=3
CACHE_DEFAULT_TTL=7200

# Add search APIs for better results
GOOGLE_API_KEY=your_key
SERPER_API_KEY=your_key
```

## 🤝 Getting Help

### Resources
- **Documentation**: `docs/` directory
- **Examples**: `examples/` directory  
- **Issues**: GitHub Issues for bugs
- **Discussions**: GitHub Discussions for questions

### Community
- Report bugs with detailed information
- Request features with use cases
- Share your research templates
- Contribute improvements

## 🎉 Success Indicators

You'll know everything is working when:

✅ `python enhanced_simple_runner.py --status` shows all systems active  
✅ Simple research query completes successfully  
✅ Results are saved in `research_outputs/`  
✅ Generated reports contain relevant information  
✅ No error messages in the logs  

**Happy researching! 🚀**

---

*For detailed configuration and advanced features, see the complete documentation in `docs/user/`.*
