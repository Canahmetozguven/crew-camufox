# Configuration Guide

This guide covers all configuration options available in Crew-Camufox.

## Configuration Overview

Crew-Camufox uses environment variables for configuration, loaded from:
1. `.env` file (primary)
2. System environment variables
3. Default values

## Environment File Setup

### Creating Configuration
```bash
# Copy the example file
cp .env.example .env

# Edit with your preferences
nano .env
```

### Configuration Categories

## Core Application Settings

### Basic Configuration
```bash
# Application Mode
DEBUG=false                    # Enable debug mode (development only)
LOG_LEVEL=INFO                # Logging level: DEBUG, INFO, WARNING, ERROR
LOG_TO_FILE=true              # Write logs to file
LOG_ROTATION=true             # Enable log rotation

# Output Directories
OUTPUT_DIR=research_outputs   # Research results directory
LOGS_DIR=logs                # Log files directory
TEMP_DIR=temp                # Temporary files directory
```

### Performance Settings
```bash
# Concurrency Limits
MAX_CONCURRENT_MISSIONS=3     # Max simultaneous research missions
MISSION_TIMEOUT=7200         # Mission timeout in seconds (2 hours)

# API Server (if using web interface)
API_HOST=localhost           # API server host
API_PORT=8000               # API server port
API_WORKERS=1               # Number of worker processes
```

## LLM Configuration

### Ollama Settings (Primary)
```bash
# Model Configuration
OLLAMA_MODEL=magistral:latest # Primary model for research
BROWSER_MODEL=granite3.3:8b  # Model for browser interactions
OLLAMA_BASE_URL=http://localhost:11434  # Ollama server URL

# Connection Settings
OLLAMA_TIMEOUT=30            # Request timeout in seconds
OLLAMA_MAX_RETRIES=3         # Maximum retry attempts
OLLAMA_TEMPERATURE=0.1       # Model temperature (creativity)
```

### Cloud LLM APIs (Optional)
```bash
# OpenAI
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=4000

# Anthropic
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# Fallback Configuration
LLM_FALLBACK_ENABLED=true    # Use fallback providers
LLM_FALLBACK_ORDER=ollama,openai,anthropic
```

## Search Engine Configuration

### Search APIs
```bash
# Google Custom Search (Recommended)
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id

# Bing Search API
BING_API_KEY=your_bing_api_key

# SerpAPI (Alternative)
SERP_API_KEY=your_serp_api_key

# Brave Search API
BRAVE_SEARCH_API_KEY=your_brave_api_key
```

### Search Behavior
```bash
# Search Limits
MAX_SOURCES=15               # Maximum sources per query
SEARCH_TIMEOUT=30           # Search timeout per engine
REQUEST_DELAY=1.0           # Delay between requests

# Search Features
ENABLE_ACADEMIC_SEARCH=true  # Include academic sources
ENABLE_DIRECT_BROWSER=true   # Use direct browser search
SEARCH_ENGINES=google,bing,duckduckgo  # Enabled engines
```

## Browser Configuration

### Camoufox Settings
```bash
# Browser Behavior
BROWSER_HEADLESS=true        # Run browser in headless mode
BROWSER_STEALTH=true         # Enable stealth browsing
BROWSER_UA_ROTATION=true     # Rotate user agents
BROWSER_PROXY_ROTATION=false # Enable proxy rotation

# Performance
BROWSER_PAGE_TIMEOUT=30      # Page load timeout
BROWSER_NAV_DELAY=2.0       # Navigation delay
BROWSER_MAX_PAGES=3         # Max pages per session

# Privacy
BROWSER_CLEAR_DATA=true     # Clear browser data after use
BROWSER_DISABLE_IMAGES=false # Disable image loading
BROWSER_DISABLE_JS=false    # Disable JavaScript
```

### Browser Profiles
```bash
# Profile Management
BROWSER_PROFILES_DIR=browser_profiles
BROWSER_USE_PROFILES=true
BROWSER_PROFILE_ROTATION=true
```

## Caching Configuration

### Cache Settings
```bash
# Basic Caching
CACHE_ENABLED=true          # Enable caching system
CACHE_DEFAULT_TTL=3600      # Default TTL in seconds (1 hour)
CACHE_SEARCH_TTL=7200      # Search cache TTL (2 hours)
CACHE_MAX_MEMORY_SIZE=1000  # Max cache entries in memory

# Cache Storage
CACHE_BACKEND=memory        # Backend: memory, redis, file
CACHE_DIR=cache            # File cache directory
```

### Redis Cache (Optional)
```bash
# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_DB=0
REDIS_PASSWORD=your_redis_password
REDIS_MAX_CONNECTIONS=10
```

## Database Configuration

### SQLite (Default)
```bash
# SQLite Settings
DATABASE_URL=sqlite:///crew_camufox.db
DATABASE_ECHO=false         # Log SQL queries
DATABASE_POOL_SIZE=5        # Connection pool size
DATABASE_MAX_OVERFLOW=10    # Max overflow connections
```

### PostgreSQL (Production)
```bash
# PostgreSQL Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/crew_camufox
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=3600
```

### MySQL (Alternative)
```bash
# MySQL Configuration
DATABASE_URL=mysql://user:password@localhost:3306/crew_camufox
DATABASE_CHARSET=utf8mb4
```

## Proxy Configuration

### HTTP Proxies
```bash
# Basic Proxy
HTTP_PROXY=http://proxy.example.com:8080
HTTPS_PROXY=http://proxy.example.com:8080

# Authenticated Proxy
PROXY_USERNAME=your_username
PROXY_PASSWORD=your_password
```

### Proxy Rotation
```bash
# Multiple Proxies
PROXY_LIST=proxy1.com:8080,proxy2.com:8080,proxy3.com:8080
PROXY_ROTATION_ENABLED=true
PROXY_ROTATION_INTERVAL=300  # Rotate every 5 minutes
```

## Monitoring Configuration

### Logging
```bash
# Log Configuration
LOG_FORMAT=json             # Format: json, text
LOG_MAX_SIZE=100MB         # Max log file size
LOG_BACKUP_COUNT=5         # Number of backup files
LOG_STRUCTURED=true        # Enable structured logging
```

### Metrics
```bash
# Performance Monitoring
METRICS_ENABLED=true       # Enable metrics collection
METRICS_INTERVAL=60        # Collection interval in seconds
METRICS_RETENTION=7d       # Metrics retention period
```

### Health Checks
```bash
# Health Monitoring
HEALTH_CHECK_ENABLED=true
HEALTH_CHECK_INTERVAL=30   # Check interval in seconds
HEALTH_CHECK_TIMEOUT=10    # Check timeout
```

## Security Configuration

### API Security
```bash
# API Security
API_KEY=your_secret_api_key
JWT_SECRET=your_jwt_secret
JWT_EXPIRATION=3600        # Token expiration in seconds

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100    # Requests per minute
RATE_LIMIT_WINDOW=60       # Window in seconds
```

### Data Protection
```bash
# Data Security
ENCRYPT_SENSITIVE_DATA=true
ENCRYPTION_KEY=your_encryption_key
DATA_RETENTION_DAYS=30     # Auto-delete old data

# Privacy
ANONYMIZE_LOGS=true        # Remove sensitive info from logs
GDPR_COMPLIANCE=true       # Enable GDPR features
```

## Feature Flags

### Experimental Features
```bash
# Feature Toggles
ENABLE_ML_FEATURES=true    # Machine learning enhancements
ENABLE_ADVANCED_SEARCH=true # Advanced search composition
ENABLE_COLLABORATION=false  # Multi-user features
ENABLE_REAL_TIME=true      # Real-time updates

# Research Features
ENABLE_FACT_CHECKING=true  # Automated fact checking
ENABLE_BIAS_DETECTION=true # Bias detection in sources
ENABLE_SENTIMENT_ANALYSIS=false # Sentiment analysis
```

## Development Configuration

### Development Mode
```bash
# Development Settings
DEVELOPMENT_MODE=true      # Enable development features
AUTO_RELOAD=true          # Auto-reload on code changes
PROFILING_ENABLED=false   # Enable performance profiling

# Testing
TEST_DATABASE_URL=sqlite:///test.db
TEST_DISABLE_CACHE=true   # Disable cache in tests
TEST_MOCK_APIS=true       # Mock external APIs
```

### Debug Features
```bash
# Debugging
DEBUG_TOOLBAR=true        # Enable debug toolbar
DEBUG_SQL=false          # Log SQL queries
DEBUG_REQUESTS=false     # Log HTTP requests
VERBOSE_ERRORS=true      # Detailed error messages
```

## Environment-Specific Configurations

### Production
```env
DEBUG=false
LOG_LEVEL=INFO
BROWSER_HEADLESS=true
CACHE_ENABLED=true
METRICS_ENABLED=true
DATABASE_POOL_SIZE=20
```

### Development
```env
DEBUG=true
LOG_LEVEL=DEBUG
BROWSER_HEADLESS=false
CACHE_ENABLED=false
AUTO_RELOAD=true
DATABASE_ECHO=true
```

### Testing
```env
DEBUG=false
LOG_LEVEL=WARNING
CACHE_ENABLED=false
TEST_MODE=true
DATABASE_URL=sqlite:///test.db
```

## Configuration Validation

### Required Settings
The following settings are required for basic operation:
- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `OUTPUT_DIR`

### Validation Script
```bash
# Validate configuration
python -c "
from src.config import get_settings
settings = get_settings()
print('Configuration valid!')
"
```

## Best Practices

### Security
- Never commit `.env` files to version control
- Use strong, unique API keys
- Regularly rotate credentials
- Use environment-specific configurations

### Performance
- Enable caching in production
- Use appropriate database for scale
- Configure reasonable timeouts
- Monitor resource usage

### Reliability
- Set up proper logging
- Configure health checks
- Use database connection pooling
- Implement proper error handling

## Troubleshooting

### Common Issues

#### Configuration Not Loading
```bash
# Check file exists
ls -la .env

# Validate syntax
python -c "import dotenv; print(dotenv.load_dotenv('.env'))"
```

#### Database Connection Issues
```bash
# Test database connection
python -c "
from src.database import get_engine
engine = get_engine()
print('Database connection successful!')
"
```

#### API Key Issues
```bash
# Test API keys (remove sensitive output)
python -c "
from src.config import get_settings
settings = get_settings()
print(f'Ollama URL: {settings.ollama_base_url}')
"
```

### Getting Help

For configuration issues:
1. Check the logs in `logs/` directory
2. Validate your `.env` file syntax
3. Review the example configuration
4. Check GitHub Issues for similar problems

---

For specific feature configuration, see the relevant documentation in `docs/`.
