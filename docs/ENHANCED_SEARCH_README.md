# Enhanced Multi-Agent Research System

## 🚀 Premium Search Engine Integration

This enhanced version integrates **Google Search API**, **Google Scholar**, and **Bing Academic** for superior research quality and academic source discovery.

### ✨ Key Enhancements

- **🔍 Google Search API**: High-quality, structured web search results
- **🎓 Google Scholar**: Academic papers, citations, and scholarly content  
- **📚 Bing Academic**: Additional academic and research sources
- **🦆 DuckDuckGo Fallback**: Ensures comprehensive coverage even without API keys
- **📊 Enhanced Quality Metrics**: Search engine diversity, academic ratio, domain diversity
- **🔗 Smart Deduplication**: Removes duplicate sources across search engines
- **⚡ Async Processing**: Parallel search execution for faster results

### 🔧 Setup Instructions

#### 1. API Keys Configuration

**Google Search API:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project and enable "Custom Search JSON API"
3. Create an API key
4. Set up a Custom Search Engine at [cse.google.com](https://cse.google.com/)
5. Get your Search Engine ID

**Bing Search API:**
1. Go to [Azure Portal](https://portal.azure.com/)
2. Create a "Bing Search v7" resource
3. Get your API key

#### 2. Environment Setup

```bash
# Copy configuration template
cp config/enhanced_research.env .env

# Edit with your API keys
nano .env
```

Set these environment variables:
```bash
export GOOGLE_API_KEY="your_google_api_key"
export GOOGLE_CSE_ID="your_search_engine_id" 
export BING_API_KEY="your_bing_api_key"
```

#### 3. Install Dependencies

```bash
# Using uv (recommended)
uv add requests beautifulsoup4 camoufox crewai langchain-ollama

# Or using pip
pip install requests beautifulsoup4 camoufox crewai langchain-ollama
```

### 🎯 Usage Examples

#### Basic Enhanced Research
```python
from src.agents.deep_researcher_enhanced import EnhancedDeepResearcherAgent
from src.agents.final_writer import FinalWriterAgent

# Initialize with API keys
researcher = EnhancedDeepResearcherAgent(
    google_api_key="your_api_key",
    google_cse_id="your_cse_id", 
    bing_api_key="your_bing_key"
)

# Research plan
plan = {
    "query": "artificial intelligence in healthcare 2024",
    "research_depth": "comprehensive",
    "max_sources_per_phase": 15
}

# Execute enhanced research
results = await researcher.execute_enhanced_research_plan(plan)
```

#### Run the Demo
```bash
python examples/enhanced_research_demo.py
```

### 📊 Search Engine Comparison

| Engine | Strengths | Use Case |
|--------|-----------|----------|
| **Google Search API** | Comprehensive, high relevance | General web research |
| **Google Scholar** | Academic papers, citations | Scholarly research |
| **Bing Academic** | Research papers, datasets | Additional academic sources |
| **DuckDuckGo** | Privacy, no API limits | Fallback/supplementary |

### 🔍 Enhanced Features

#### Multi-Engine Search Strategy
```python
# The system automatically:
1. Searches Google API for high-quality web results
2. Queries Google Scholar for academic sources
3. Checks Bing Academic for additional research papers
4. Falls back to DuckDuckGo for comprehensive coverage
5. Deduplicates and ranks all results
```

#### Quality Metrics
- **Overall Score**: Weighted combination of all metrics
- **Search Diversity**: Number of search engines successfully used  
- **Academic Ratio**: Percentage of sources from academic engines
- **Domain Diversity**: Variety of domains in source list
- **Credibility Score**: Source reliability assessment

#### Smart Source Processing
- Automatic content extraction and analysis
- Source type detection (academic, news, blog, etc.)
- Relevance scoring using LLM assessment
- Quality filtering and ranking

### 🎯 Benefits Over Standard Search

#### Higher Quality Results
- **API-based search**: More reliable than web scraping
- **Academic focus**: Direct access to scholarly content
- **Structured data**: Better metadata and source information
- **Rate limit friendly**: Sustainable for production use

#### Better Coverage
- **Multi-engine approach**: Combines strengths of different engines
- **Academic specialization**: Dedicated scholarly search
- **Fallback redundancy**: Works even without API keys
- **Source diversity**: Broader range of content types

### 🔧 Configuration Options

#### Research Configuration
```python
research_plan = {
    "query": "your research topic",
    "research_depth": "comprehensive",  # basic, standard, comprehensive
    "max_sources_per_phase": 15,       # sources per search engine
    "quality_threshold": 0.7,          # minimum quality score
    "execution_phases": [...]          # custom research phases
}
```

#### Search Engine Selection
```python
# Control which engines to use
enhanced_search._run(
    query="research topic",
    include_google=True,      # Google Search API
    include_scholar=True,     # Google Scholar  
    include_bing_academic=True, # Bing Academic
    max_results=20
)
```

### 📈 Performance Improvements

- **Parallel Processing**: All search engines queried simultaneously
- **Async Architecture**: Non-blocking operations throughout
- **Smart Caching**: Avoids redundant API calls
- **Error Handling**: Graceful fallbacks when APIs fail
- **Rate Limiting**: Respects API quotas and limits

### 🆚 Comparison: Standard vs Enhanced

| Feature | Standard System | Enhanced System |
|---------|----------------|-----------------|
| Search Engines | DuckDuckGo, Startpage | Google API, Scholar, Bing Academic + fallbacks |
| Academic Sources | Limited | Comprehensive via Scholar & Bing Academic |
| Result Quality | Good | Excellent with API-structured data |
| Search Reliability | Variable (scraping) | High (official APIs) |
| Rate Limits | Prone to blocking | API quota managed |
| Source Diversity | Medium | High with multi-engine approach |
| Setup Complexity | Simple | Moderate (requires API keys) |
| Cost | Free | API costs (Google/Bing usage) |

### 💡 Tips for Best Results

1. **Use specific queries**: More targeted searches yield better results
2. **Set appropriate thresholds**: Balance quality vs quantity  
3. **Enable all engines**: Maximum coverage and source diversity
4. **Monitor API quotas**: Track usage to avoid limits
5. **Regular key rotation**: Maintain API key security

### 🔍 Troubleshooting

**No Google results?**
- Check API key validity
- Verify Custom Search Engine setup
- Check quota limits in Google Cloud Console

**Bing Academic not working?**
- Confirm Bing Search v7 resource is active
- Verify API key has proper permissions
- Check Azure subscription status

**General search failures?**
- System falls back to DuckDuckGo automatically
- Check network connectivity and firewall settings
- Review logs for specific error messages

### 📚 Additional Resources

- [Google Custom Search Documentation](https://developers.google.com/custom-search/v1/introduction)
- [Bing Search API Documentation](https://docs.microsoft.com/en-us/bing/search-apis/bing-web-search/)
- [Research Quality Assessment Guide](docs/research_quality.md)
- [API Cost Optimization Tips](docs/cost_optimization.md)
