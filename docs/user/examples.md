# Examples and Use Cases

This document showcases various ways to use Crew-Camufox for different research needs.

## 🎯 Basic Examples

### Simple Research Query
```bash
# Basic research on a topic
python enhanced_simple_runner.py "artificial intelligence applications in healthcare"
```

### Research with Depth Control
```bash
# Surface level (quick overview)
python enhanced_simple_runner.py "blockchain technology" --depth surface

# Medium depth (balanced research)
python enhanced_simple_runner.py "renewable energy" --depth medium

# Deep research (comprehensive analysis)
python enhanced_simple_runner.py "quantum computing" --depth deep
```

### Focused Research
```bash
# Research with specific focus areas
python enhanced_simple_runner.py "machine learning" \
  --focus "healthcare applications" "diagnostic imaging" \
  --max-sources 20
```

## 🎓 Academic Research

### Scientific Literature Review
```bash
# Academic research with scholarly focus
python enhanced_simple_runner.py \
  "CRISPR gene editing therapeutic applications" \
  --depth deep \
  --focus "peer-reviewed papers" "clinical trials" "FDA approval" \
  --exclude "news articles" "opinion pieces" \
  --max-sources 30
```

### Research Paper Investigation
```bash
# Investigating specific research areas
python enhanced_simple_runner.py \
  "neural network architectures for natural language processing" \
  --depth comprehensive \
  --focus "transformer models" "attention mechanisms" "research papers" \
  --exclude "tutorials" "blogs"
```

### Grant Research
```bash
# Finding funding opportunities
python enhanced_simple_runner.py \
  "renewable energy research grants 2024" \
  --focus "government funding" "NSF grants" "DOE programs" \
  --depth medium
```

## 💼 Business Intelligence

### Market Research
```bash
# Market analysis and trends
python enhanced_simple_runner.py \
  "electric vehicle market trends 2024" \
  --depth medium \
  --focus "market share" "sales data" "consumer adoption" "competitive analysis" \
  --max-sources 25
```

### Competitive Analysis
```bash
# Competitor research
python enhanced_simple_runner.py \
  "cloud computing platforms comparison" \
  --focus "AWS features" "Azure capabilities" "Google Cloud services" \
  --exclude "promotional content" \
  --depth deep
```

### Industry Analysis
```bash
# Industry trends and forecasts
python enhanced_simple_runner.py \
  "fintech industry outlook 2024" \
  --focus "digital payments" "cryptocurrency adoption" "regulatory changes" \
  --depth medium
```

## 🔬 Technical Research

### Technology Documentation
```bash
# Technical documentation research
python enhanced_simple_runner.py \
  "kubernetes security best practices" \
  --depth deep \
  --focus "official documentation" "security guides" "RBAC configuration" \
  --exclude "forums" "social media" "outdated content"
```

### Software Architecture
```bash
# Architecture patterns research
python enhanced_simple_runner.py \
  "microservices architecture patterns" \
  --focus "design patterns" "best practices" "scalability" \
  --depth comprehensive
```

### Technology Comparison
```bash
# Comparing technologies
python enhanced_simple_runner.py \
  "Python vs JavaScript for web development" \
  --focus "performance comparison" "ecosystem" "learning curve" \
  --depth medium
```

## 📰 News and Current Events

### Current Events Analysis
```bash
# Recent news analysis
python enhanced_simple_runner.py \
  "climate change policy developments 2024" \
  --focus "government policies" "international agreements" "environmental impact" \
  --depth medium
```

### Trend Analysis
```bash
# Analyzing emerging trends
python enhanced_simple_runner.py \
  "remote work trends post-pandemic" \
  --focus "productivity studies" "company policies" "technology adoption" \
  --exclude "opinion blogs"
```

## 🏥 Healthcare Research

### Medical Research
```bash
# Medical condition research
python enhanced_simple_runner.py \
  "immunotherapy cancer treatment advances" \
  --depth deep \
  --focus "clinical trials" "FDA approvals" "research studies" \
  --exclude "alternative medicine" "unverified claims" \
  --max-sources 25
```

### Public Health
```bash
# Public health analysis
python enhanced_simple_runner.py \
  "mental health impact of social media" \
  --focus "research studies" "statistical data" "expert opinions" \
  --depth medium
```

## 🌍 Environmental Research

### Climate Research
```bash
# Climate change research
python enhanced_simple_runner.py \
  "carbon capture technologies effectiveness" \
  --depth deep \
  --focus "scientific studies" "pilot projects" "cost analysis" \
  --exclude "political opinion"
```

### Sustainability
```bash
# Sustainability practices
python enhanced_simple_runner.py \
  "sustainable agriculture practices" \
  --focus "organic farming" "permaculture" "crop rotation" \
  --depth medium
```

## 💡 Innovation Research

### Emerging Technologies
```bash
# Cutting-edge technology research
python enhanced_simple_runner.py \
  "augmented reality applications in education" \
  --focus "case studies" "implementation examples" "learning outcomes" \
  --depth deep
```

### Patent Research
```bash
# Patent and IP research
python enhanced_simple_runner.py \
  "artificial intelligence patents 2023-2024" \
  --focus "patent filings" "innovation trends" "major companies" \
  --depth medium
```

## 📊 Data Analysis Research

### Data Science Trends
```bash
# Data science methodology research
python enhanced_simple_runner.py \
  "machine learning model interpretability techniques" \
  --focus "SHAP values" "LIME methodology" "research papers" \
  --depth comprehensive
```

### Big Data Applications
```bash
# Big data use cases
python enhanced_simple_runner.py \
  "big data applications in retail analytics" \
  --focus "customer behavior" "inventory management" "case studies" \
  --depth medium
```

## 🎮 Creative and Media Research

### Content Strategy
```bash
# Content marketing research
python enhanced_simple_runner.py \
  "video content marketing trends 2024" \
  --focus "engagement metrics" "platform strategies" "success stories" \
  --depth medium
```

### Gaming Industry
```bash
# Gaming market analysis
python enhanced_simple_runner.py \
  "virtual reality gaming market growth" \
  --focus "market size" "hardware adoption" "game development trends" \
  --depth deep
```

## 🔒 Security Research

### Cybersecurity
```bash
# Security threat analysis
python enhanced_simple_runner.py \
  "cybersecurity threats in cloud computing" \
  --depth deep \
  --focus "threat vectors" "security measures" "compliance requirements" \
  --exclude "vendor marketing"
```

### Privacy Research
```bash
# Privacy and data protection
python enhanced_simple_runner.py \
  "GDPR compliance in AI systems" \
  --focus "regulatory requirements" "implementation guidelines" "case studies" \
  --depth medium
```

## 🎨 Using Python API

### Basic Python Usage
```python
import asyncio
from enhanced_simple_runner import EnhancedSimpleRunner

async def research_example():
    runner = EnhancedSimpleRunner()
    await runner.setup()
    
    result = await runner.research(
        query="sustainable energy solutions",
        depth="medium",
        max_sources=15,
        focus_areas=["solar power", "wind energy"],
        exclude_domains=["wikipedia.org"]
    )
    
    runner.display_results(result)
    runner.save_results(result)

# Run the research
asyncio.run(research_example())
```

### Advanced Python Usage
```python
import asyncio
from enhanced_simple_runner import EnhancedSimpleRunner

async def advanced_research():
    runner = EnhancedSimpleRunner()
    await runner.setup()
    
    # Multiple research queries
    queries = [
        "artificial intelligence in healthcare",
        "machine learning in finance", 
        "AI ethics and bias"
    ]
    
    results = []
    for query in queries:
        result = await runner.research(
            query=query,
            depth="medium",
            max_sources=10,
            fact_check=True
        )
        results.append(result)
    
    # Process combined results
    for i, result in enumerate(results):
        print(f"\n=== Research {i+1}: {queries[i]} ===")
        runner.display_results(result)

asyncio.run(advanced_research())
```

## 🎛️ Advanced Configuration Examples

### High-Performance Research
```bash
# Configure for maximum performance
export MAX_CONCURRENT_MISSIONS=5
export BROWSER_MAX_PAGES=5
export CACHE_ENABLED=true
export CACHE_DEFAULT_TTL=7200

python enhanced_simple_runner.py \
  "artificial intelligence market analysis" \
  --depth deep \
  --max-sources 50
```

### Privacy-Focused Research
```bash
# Maximum privacy settings
export BROWSER_STEALTH=true
export BROWSER_UA_ROTATION=true
export BROWSER_CLEAR_DATA=true

python enhanced_simple_runner.py \
  "privacy protection technologies" \
  --depth medium
```

### Development and Testing
```bash
# Debug mode with visible browser
export DEBUG=true
export LOG_LEVEL=DEBUG
export BROWSER_HEADLESS=false

python enhanced_simple_runner.py \
  "test query" \
  --show-browser
```

## 📈 Performance Optimization Examples

### Cached Research
```bash
# Enable aggressive caching
export CACHE_ENABLED=true
export CACHE_SEARCH_TTL=14400  # 4 hours
export CACHE_DEFAULT_TTL=7200  # 2 hours

# First run (builds cache)
python enhanced_simple_runner.py "machine learning frameworks"

# Subsequent runs (uses cache)
python enhanced_simple_runner.py "machine learning frameworks comparison"
```

### Parallel Processing
```bash
# Configure for parallel execution
export MAX_CONCURRENT_AGENTS=3
export BROWSER_MAX_PAGES=3

python enhanced_simple_runner.py \
  "comprehensive technology review" \
  --depth deep \
  --max-sources 30
```

## 🔄 Workflow Examples

### Multi-Stage Research
```bash
# Stage 1: Broad overview
python enhanced_simple_runner.py \
  "renewable energy" \
  --depth surface \
  --max-sources 10

# Stage 2: Focused deep dive
python enhanced_simple_runner.py \
  "solar panel efficiency improvements" \
  --depth deep \
  --focus "perovskite cells" "silicon alternatives" \
  --max-sources 25

# Stage 3: Implementation analysis
python enhanced_simple_runner.py \
  "commercial solar panel deployment challenges" \
  --depth medium \
  --focus "cost analysis" "installation" "maintenance"
```

### Research Pipeline
```bash
#!/bin/bash
# Automated research pipeline

queries=(
    "AI in healthcare:healthcare applications,medical diagnosis"
    "AI in finance:algorithmic trading,risk assessment" 
    "AI ethics:bias detection,fairness,regulation"
)

for query_info in "${queries[@]}"; do
    IFS=':' read -r query focus <<< "$query_info"
    
    python enhanced_simple_runner.py "$query" \
        --depth medium \
        --focus "$focus" \
        --max-sources 15
done
```

## 📋 Result Processing Examples

### Custom Result Analysis
```python
import json
import glob

def analyze_research_results():
    # Find latest research files
    result_files = glob.glob("research_outputs/*_complete.json")
    latest_file = max(result_files, key=os.path.getctime)
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    # Extract key metrics
    total_sources = len(data.get('sources', []))
    avg_credibility = sum(s.get('credibility_score', 0) 
                         for s in data.get('sources', [])) / total_sources
    
    print(f"Sources analyzed: {total_sources}")
    print(f"Average credibility: {avg_credibility:.2f}")
    
    # Show top sources
    sources = data.get('sources', [])
    top_sources = sorted(sources, 
                        key=lambda x: x.get('credibility_score', 0), 
                        reverse=True)[:5]
    
    print("\nTop 5 most credible sources:")
    for i, source in enumerate(top_sources, 1):
        print(f"{i}. {source.get('title', 'N/A')} "
              f"(Score: {source.get('credibility_score', 0):.2f})")

analyze_research_results()
```

## 🎯 Best Practices

### Research Query Optimization
```bash
# Good: Specific and focused
python enhanced_simple_runner.py \
  "machine learning model deployment best practices" \
  --focus "MLOps" "containerization" "monitoring"

# Better: Include context and constraints
python enhanced_simple_runner.py \
  "machine learning model deployment in production environments" \
  --focus "MLOps pipelines" "Docker containers" "model monitoring" \
  --exclude "tutorials" "basic concepts" \
  --depth deep
```

### Source Quality Control
```bash
# Exclude low-quality sources
python enhanced_simple_runner.py \
  "climate change scientific evidence" \
  --focus "peer-reviewed studies" "IPCC reports" \
  --exclude "blogs" "opinion pieces" "social media" \
  --depth deep
```

### Efficient Resource Usage
```bash
# Balance depth and speed
python enhanced_simple_runner.py \
  "quick market overview" \
  --depth surface \
  --max-sources 8

python enhanced_simple_runner.py \
  "comprehensive analysis" \
  --depth deep \
  --max-sources 25
```

---

For more advanced usage patterns and customization options, see the [Developer Guide](../developer/) and [API Documentation](../api/).
