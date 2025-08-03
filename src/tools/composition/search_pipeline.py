"""
Enhanced Search Pipeline
Specialized pipeline for multi-engine search with composition, parallel processing, and intelligent filtering
"""

from .pipeline import ToolPipeline, PipelineStage, ToolResult
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import logging
import re

# Try to import browser search, fallback to simple search
try:
    from ..browser_search import BrowserSearchTool
    BROWSER_SEARCH_AVAILABLE = True
except ImportError:
    BROWSER_SEARCH_AVAILABLE = False

try:
    from ..simple_search import RealSearchTool, SimpleSearchTool
    SIMPLE_SEARCH_AVAILABLE = True
except ImportError:
    SIMPLE_SEARCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnhancedSearchPipeline(ToolPipeline):
    """
    Specialized pipeline for multi-engine search with composition
    
    Features:
    - Query optimization and variant generation
    - Parallel multi-engine search execution
    - Intelligent result deduplication and ranking
    - Quality validation and filtering
    - Relevance scoring and result enhancement
    """
    
    def __init__(self, headless: bool = True):
        super().__init__("enhanced_search")
        self.headless = headless
        self._setup_pipeline()
        self.supported_engines = ["google", "bing", "duckduckgo", "scholar"]
    
    def _setup_pipeline(self):
        """Configure the search pipeline stages with timeouts and parallel groups"""
        
        # Stage 1: Query preprocessing (30 seconds timeout)
        self.add_stage(PipelineStage.PREPROCESSING, [
            self._optimize_query,
            self._generate_variants,
            self._select_engines
        ], timeout=30.0)
        
        # Stage 2: Parallel search execution (120 seconds timeout for multiple engines)
        self.add_stage(PipelineStage.EXECUTION, [
            self._execute_parallel_searches
        ], timeout=120.0)
        
        # Stage 3: Result processing (60 seconds timeout)
        self.add_stage(PipelineStage.POSTPROCESSING, [
            self._deduplicate_results,
            self._score_relevance,
            self._rank_results,
            self._enhance_metadata
        ], timeout=60.0)
        
        # Stage 4: Quality validation (30 seconds timeout)
        self.add_stage(PipelineStage.VALIDATION, [
            self._validate_quality,
            self._apply_filters,
            self._finalize_results
        ], timeout=30.0)
        
        # Define parallel groups for search engines
        self.add_parallel_group(["google_search", "bing_search", "duckduckgo_search", "scholar_search"])
        
        # Add fallback strategies
        self.add_fallback_strategy("google_search", self._fallback_search)
        self.add_fallback_strategy("bing_search", self._fallback_search)
        self.add_fallback_strategy("duckduckgo_search", self._fallback_search)
        
        # Add error handlers
        self.add_error_handler("execution", self._handle_search_errors)
        
        logger.info("Enhanced search pipeline configured with 4 stages and parallel execution")
    
    async def _optimize_query(self, query_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize search query for better results"""
        query = query_data.get("query", "")
        
        if not query:
            raise ValueError("Query is required for search pipeline")
        
        # Clean and optimize the original query
        optimized_query = self._clean_query(query)
        
        # Generate optimized variants
        optimized_variants = [
            optimized_query,  # Original cleaned
            f'"{optimized_query}"',  # Exact phrase (with quotes)
            f"{optimized_query} research study",  # Academic focus
            f"{optimized_query} analysis report",  # Report focus
            f"{optimized_query} latest news",  # Recent content
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for variant in optimized_variants:
            if variant not in seen:
                seen.add(variant)
                unique_variants.append(variant)
        
        query_data["optimized_queries"] = unique_variants
        query_data["primary_query"] = optimized_query
        
        logger.debug(f"Query optimization: '{query}' -> {len(unique_variants)} variants")
        return query_data
    
    async def _generate_variants(self, query_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate additional query variants for comprehensive search"""
        primary_query = query_data.get("primary_query", "")
        existing_variants = query_data.get("optimized_queries", [])
        
        # Generate semantic variants
        additional_variants = []
        
        # Add synonym-based variants
        if "AI" in primary_query or "artificial intelligence" in primary_query.lower():
            additional_variants.extend([
                primary_query.replace("AI", "artificial intelligence"),
                primary_query.replace("artificial intelligence", "AI"),
                primary_query + " machine learning",
                primary_query + " deep learning"
            ])
        
        # Add domain-specific variants
        if any(term in primary_query.lower() for term in ["research", "study", "analysis"]):
            additional_variants.extend([
                primary_query + " methodology",
                primary_query + " findings",
                primary_query + " conclusions"
            ])
        
        # Add temporal variants for current topics
        if any(term in primary_query.lower() for term in ["2024", "recent", "latest", "new"]):
            additional_variants.extend([
                primary_query + " 2024",
                primary_query + " recent developments",
                primary_query + " latest trends"
            ])
        
        # Combine with existing variants, remove duplicates
        all_variants = existing_variants + additional_variants
        seen = set()
        unique_variants = []
        for variant in all_variants:
            if variant and variant not in seen:
                seen.add(variant)
                unique_variants.append(variant)
        
        query_data["all_query_variants"] = unique_variants[:8]  # Limit to prevent overwhelming
        logger.debug(f"Generated {len(unique_variants)} total query variants")
        
        return query_data
    
    async def _select_engines(self, query_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Select optimal search engines based on query type and context"""
        query = query_data.get("primary_query", "")
        max_results = query_data.get("max_results", 15)  # Use 15 as default instead of 10
        
        # Default engine selection
        selected_engines = ["google", "bing", "duckduckgo"]
        
        # Add specialized engines based on query content
        if any(term in query.lower() for term in ["academic", "research", "study", "paper", "journal"]):
            selected_engines.append("scholar")
            logger.debug("Added Google Scholar for academic query")
        
        # Adjust based on context if provided
        if context:
            preferred_engines = context.get("preferred_engines", [])
            if preferred_engines:
                # Merge with preferred engines, maintaining order
                for engine in preferred_engines:
                    if engine in self.supported_engines and engine not in selected_engines:
                        selected_engines.append(engine)
        
        # Calculate results per engine - allow each engine to contribute more results
        # Instead of dividing, set a reasonable target per engine to get comprehensive coverage
        results_per_engine = max(10, min(15, max_results // 2))  # Each engine gets 10-15 results
        
        query_data["selected_engines"] = selected_engines
        query_data["results_per_engine"] = results_per_engine
        
        logger.info(f"Selected {len(selected_engines)} engines: {selected_engines}")
        return query_data
    
    async def _execute_parallel_searches(self, query_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute searches across multiple engines in parallel"""
        engines = query_data.get("selected_engines", ["google", "bing", "duckduckgo"])
        queries = query_data.get("all_query_variants", [query_data.get("primary_query", "")])
        results_per_engine = query_data.get("results_per_engine", 15)  # Increased from 5 to 15
        
        # Create search tasks for parallel execution
        search_tasks = []
        
        for engine in engines:
            # Use primary query for each engine to avoid overwhelming
            primary_query = queries[0] if queries else ""
            if primary_query:
                task_name = f"{engine}_search"
                task = asyncio.create_task(
                    self._search_single_engine(engine, primary_query, results_per_engine),
                    name=task_name
                )
                search_tasks.append((engine, task))
        
        # Execute searches in parallel with timeout
        logger.info(f"Executing {len(search_tasks)} parallel searches")
        
        all_results = []
        engine_stats = {}
        
        # Wait for all tasks to complete
        for engine, task in search_tasks:
            try:
                results = await asyncio.wait_for(task, timeout=45.0)  # 45 second timeout per engine
                all_results.extend(results)
                engine_stats[engine] = {
                    "status": "success",
                    "results_count": len(results),
                    "error": None
                }
                logger.debug(f"{engine.capitalize()} search completed: {len(results)} results")
                
            except asyncio.TimeoutError:
                logger.warning(f"{engine.capitalize()} search timed out after 45 seconds")
                engine_stats[engine] = {
                    "status": "timeout",
                    "results_count": 0,
                    "error": "Search timed out"
                }
                
            except Exception as e:
                logger.error(f"{engine.capitalize()} search failed: {e}")
                engine_stats[engine] = {
                    "status": "failed",
                    "results_count": 0,
                    "error": str(e)
                }
        
        query_data["raw_results"] = all_results
        query_data["engine_stats"] = engine_stats
        query_data["total_raw_results"] = len(all_results)
        
        logger.info(f"Parallel search completed: {len(all_results)} total results from {len(engines)} engines")
        return query_data
    
    async def _search_single_engine(self, engine: str, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search a single engine with enhanced error handling and fallback support"""
        try:
            # Try browser search first (if available)
            if BROWSER_SEARCH_AVAILABLE:
                try:
                    async with BrowserSearchTool(headless=self.headless, page_timeout=30) as browser_search:
                        if engine == "google":
                            results = await browser_search.search_google(query, max_results)
                        elif engine == "bing":
                            results = await browser_search.search_bing(query, max_results)
                        elif engine == "duckduckgo":
                            results = await browser_search.search_duckduckgo(query, max_results)
                        elif engine == "scholar":
                            # For now, use Google search as scholar implementation
                            results = await browser_search.search_google(f"site:scholar.google.com {query}", max_results)
                        else:
                            logger.warning(f"Unsupported engine: {engine}")
                            results = []
                        
                        # Add engine metadata to results
                        for result in results:
                            result["search_engine"] = engine
                            result["search_timestamp"] = datetime.now().isoformat()
                            result["search_method"] = "browser"
                        
                        return results
                        
                except Exception as browser_error:
                    logger.warning(f"Browser search failed for {engine}: {browser_error}, falling back to simple search")
            
            # Fallback to simple search
            if SIMPLE_SEARCH_AVAILABLE:
                real_search = RealSearchTool()
                
                if engine in ["google", "bing", "duckduckgo"]:
                    # Use the main search method
                    results = await real_search.search(query, max_results)
                elif engine == "scholar":
                    # For scholar, modify query and use simple search
                    scholar_query = f"{query} research study academic"
                    results = await real_search.search(scholar_query, max_results)
                else:
                    results = []
                
                # Add engine metadata to results
                for result in results:
                    result["search_engine"] = engine
                    result["search_timestamp"] = datetime.now().isoformat()
                    result["search_method"] = "api"
                
                return results
            
            # If no search methods are available, return empty results
            logger.error(f"No search methods available for {engine}")
            return []
                
        except Exception as e:
            logger.error(f"Search failed for {engine}: {e}")
            return []
    
    async def _deduplicate_results(self, query_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Remove duplicate search results based on URL and content similarity"""
        raw_results = query_data.get("raw_results", [])
        
        if not raw_results:
            query_data["deduplicated_results"] = []
            return query_data
        
        seen_urls = set()
        seen_titles = set()
        unique_results = []
        
        for result in raw_results:
            url = result.get("url", "")
            title = result.get("title", "").lower().strip()
            
            # Normalize URL for comparison
            normalized_url = self._normalize_url(url)
            
            # Check for exact URL duplicates
            if normalized_url in seen_urls:
                continue
            
            # Check for very similar titles (potential duplicates)
            if title and len(title) > 10:
                title_similarity = any(
                    self._calculate_text_similarity(title, seen_title) > 0.85
                    for seen_title in seen_titles
                )
                if title_similarity:
                    continue
            
            # Add to unique results
            seen_urls.add(normalized_url)
            if title:
                seen_titles.add(title)
            unique_results.append(result)
        
        query_data["deduplicated_results"] = unique_results
        query_data["deduplication_stats"] = {
            "original_count": len(raw_results),
            "after_dedup": len(unique_results),
            "duplicates_removed": len(raw_results) - len(unique_results)
        }
        
        logger.info(f"Deduplication: {len(raw_results)} -> {len(unique_results)} results")
        return query_data
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL for duplicate detection"""
        if not url:
            return ""
        
        # Remove common tracking parameters
        url = re.sub(r'[?&](utm_|ref|source|medium|campaign|gclid|fbclid)=[^&]*', '', url)
        
        # Remove trailing slashes and fragments
        url = url.rstrip('/').split('#')[0]
        
        # Convert to lowercase
        return url.lower()
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity between two strings"""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def _score_relevance(self, query_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Score results for relevance using multiple factors"""
        results = query_data.get("deduplicated_results", [])
        primary_query = query_data.get("primary_query", "")
        
        for result in results:
            relevance_score = self._calculate_relevance_score(result, primary_query)
            result["relevance_score"] = relevance_score
            
            # Add additional scoring factors
            result["quality_indicators"] = self._extract_quality_indicators(result)
            result["content_type"] = self._classify_content_type(result)
        
        query_data["scored_results"] = results
        logger.debug(f"Scored {len(results)} results for relevance")
        
        return query_data
    
    def _calculate_relevance_score(self, result: Dict[str, Any], query: str) -> float:
        """Calculate comprehensive relevance score for a result"""
        title = result.get("title", "").lower()
        snippet = result.get("snippet", "").lower()
        url = result.get("url", "").lower()
        query_lower = query.lower()
        
        score = 0.0
        
        # Title relevance (highest weight)
        if query_lower in title:
            score += 0.4
        else:
            # Partial word matches in title
            query_words = query_lower.split()
            title_matches = sum(1 for word in query_words if word in title)
            score += (title_matches / len(query_words)) * 0.3
        
        # Snippet relevance
        if query_lower in snippet:
            score += 0.25
        else:
            # Partial word matches in snippet
            query_words = query_lower.split()
            snippet_matches = sum(1 for word in query_words if word in snippet)
            score += (snippet_matches / len(query_words)) * 0.15
        
        # URL relevance (domain authority and path relevance)
        domain_score = self._calculate_domain_authority(url)
        score += domain_score * 0.2
        
        # Content type bonus
        content_type = self._classify_content_type(result)
        if content_type in ["academic", "government", "news"]:
            score += 0.1
        
        # Existing credibility score integration
        existing_credibility = result.get("credibility_score", 0.5)
        score += existing_credibility * 0.15
        
        return min(score, 1.0)
    
    def _calculate_domain_authority(self, url: str) -> float:
        """Calculate domain authority score"""
        if not url:
            return 0.0
        
        high_authority_domains = [
            ".edu", ".gov", ".org",
            "wikipedia.org", "arxiv.org", "pubmed.ncbi.nlm.nih.gov",
            "ieee.org", "acm.org", "nature.com", "science.org",
            "reuters.com", "bbc.com", "npr.org"
        ]
        
        medium_authority_domains = [
            ".com", ".net",
            "techcrunch.com", "wired.com", "arstechnica.com"
        ]
        
        url_lower = url.lower()
        
        for domain in high_authority_domains:
            if domain in url_lower:
                return 0.9
        
        for domain in medium_authority_domains:
            if domain in url_lower:
                return 0.6
        
        return 0.4  # Default score
    
    def _classify_content_type(self, result: Dict[str, Any]) -> str:
        """Classify the content type of a result"""
        url = result.get("url", "").lower()
        title = result.get("title", "").lower()
        
        if any(domain in url for domain in [".edu", "arxiv.org", "pubmed", "scholar.google"]):
            return "academic"
        elif any(domain in url for domain in [".gov", "who.int", "nih.gov"]):
            return "government"
        elif "wikipedia.org" in url:
            return "encyclopedia"
        elif any(domain in url for domain in ["reuters.com", "bbc.com", "npr.org", "news"]):
            return "news"
        elif any(word in title for word in ["blog", "opinion", "review"]):
            return "blog"
        elif any(word in title for word in ["buy", "shop", "price", "sale"]):
            return "commercial"
        else:
            return "general"
    
    def _extract_quality_indicators(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract quality indicators from result metadata"""
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        url = result.get("url", "")
        
        indicators = {
            "has_date": bool(re.search(r'\d{4}', snippet)),  # Contains year
            "has_author": any(word in snippet.lower() for word in ["by ", "author", "written"]),
            "has_statistics": bool(re.search(r'\d+%|\d+(?:,\d+)*', snippet)),  # Contains numbers/percentages
            "secure_url": url.startswith("https://"),
            "academic_keywords": sum(1 for word in ["research", "study", "analysis", "methodology"] if word in title.lower()),
            "content_length": len(snippet),
            "title_length": len(title)
        }
        
        return indicators
    
    async def _rank_results(self, query_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Rank results by relevance, quality, and other factors"""
        results = query_data.get("scored_results", [])
        
        if not results:
            query_data["ranked_results"] = []
            return query_data
        
        # Sort by multiple criteria
        ranked_results = sorted(
            results,
            key=lambda x: (
                x.get("relevance_score", 0),
                x.get("credibility_score", 0),
                x.get("quality_indicators", {}).get("academic_keywords", 0),
                -len(x.get("url", "")),  # Prefer shorter URLs (negative for descending)
            ),
            reverse=True
        )
        
        # Add ranking metadata
        for i, result in enumerate(ranked_results):
            result["rank"] = i + 1
            result["ranking_score"] = (
                result.get("relevance_score", 0) * 0.6 +
                result.get("credibility_score", 0) * 0.4
            )
        
        query_data["ranked_results"] = ranked_results
        logger.info(f"Ranked {len(ranked_results)} results by relevance and quality")
        
        return query_data
    
    async def _enhance_metadata(self, query_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enhance results with additional metadata"""
        results = query_data.get("ranked_results", [])
        
        for result in results:
            # Add search metadata
            result["search_metadata"] = {
                "pipeline_version": "1.0.0",
                "processed_at": datetime.now().isoformat(),
                "search_engines_used": query_data.get("selected_engines", []),
                "query_variants_used": len(query_data.get("all_query_variants", [])),
                "deduplication_applied": True,
                "relevance_scored": True
            }
            
            # Add confidence score
            relevance = result.get("relevance_score", 0)
            credibility = result.get("credibility_score", 0)
            quality_score = self._calculate_quality_score(result.get("quality_indicators", {}))
            
            confidence_score = (relevance * 0.5 + credibility * 0.3 + quality_score * 0.2)
            result["confidence_score"] = round(confidence_score, 3)
        
        query_data["enhanced_results"] = results
        return query_data
    
    def _calculate_quality_score(self, quality_indicators: Dict[str, Any]) -> float:
        """Calculate overall quality score from indicators"""
        if not quality_indicators:
            return 0.5
        
        score = 0.0
        max_score = 0.0
        
        # Weight different quality indicators
        weights = {
            "has_date": 0.1,
            "has_author": 0.15,
            "has_statistics": 0.1,
            "secure_url": 0.05,
            "academic_keywords": 0.2,
        }
        
        for indicator, weight in weights.items():
            max_score += weight
            value = quality_indicators.get(indicator, 0)
            if isinstance(value, bool):
                score += weight if value else 0
            elif isinstance(value, (int, float)):
                # Normalize numeric values
                normalized_value = min(value / 5.0, 1.0) if indicator == "academic_keywords" else (1.0 if value > 0 else 0.0)
                score += weight * normalized_value
        
        # Content length bonus (normalized)
        content_length = quality_indicators.get("content_length", 0)
        if content_length > 100:
            score += 0.1
        max_score += 0.1
        
        return min(score / max_score, 1.0) if max_score > 0 else 0.5
    
    async def _validate_quality(self, query_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Validate result quality and filter out low-quality results"""
        results = query_data.get("enhanced_results", [])
        
        # Quality thresholds - relaxed for comprehensive research
        min_relevance = context.get("min_relevance", 0.1) if context else 0.1  # Reduced from 0.2
        min_credibility = context.get("min_credibility", 0.2) if context else 0.2  # Reduced from 0.3
        min_confidence = context.get("min_confidence", 0.15) if context else 0.15  # Reduced from 0.25
        
        quality_results = []
        filtered_out = []
        
        for result in results:
            relevance = result.get("relevance_score", 0)
            credibility = result.get("credibility_score", 0)
            confidence = result.get("confidence_score", 0)
            
            # Apply quality filters
            if (relevance >= min_relevance and 
                credibility >= min_credibility and 
                confidence >= min_confidence):
                quality_results.append(result)
            else:
                filtered_out.append({
                    "url": result.get("url", ""),
                    "title": result.get("title", "")[:50],
                    "reason": f"Low scores: R={relevance:.2f}, C={credibility:.2f}, Conf={confidence:.2f}"
                })
        
        query_data["quality_results"] = quality_results
        query_data["quality_filter_stats"] = {
            "input_count": len(results),
            "passed_quality": len(quality_results),
            "filtered_out": len(filtered_out),
            "filter_reasons": filtered_out
        }
        
        logger.info(f"Quality validation: {len(results)} -> {len(quality_results)} results")
        return query_data
    
    async def _apply_filters(self, query_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply final filters and constraints"""
        results = query_data.get("quality_results", [])
        max_results = query_data.get("max_results", 15)  # Use 15 as default instead of 10
        
        # Apply content type filters if specified in context
        if context and "content_types" in context:
            allowed_types = context["content_types"]
            results = [r for r in results if r.get("content_type") in allowed_types]
        
        # Apply domain filters if specified
        if context and "excluded_domains" in context:
            excluded_domains = context["excluded_domains"]
            results = [r for r in results if not any(domain in r.get("url", "") for domain in excluded_domains)]
        
        # Apply final result limit - increased for comprehensive research reports
        # Don't apply strict limit for research purposes - let all quality results through
        # Only limit if specifically requested in context
        if context and "enforce_max_results" in context and context["enforce_max_results"]:
            filtered_results = results[:max_results]
        else:
            # For research reports, use all quality results up to a reasonable limit
            filtered_results = results[:min(len(results), 35)]  # Allow up to 35 sources for comprehensive research
        
        query_data["filtered_results"] = filtered_results
        return query_data
    
    async def _finalize_results(self, query_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Finalize results and prepare final output"""
        results = query_data.get("filtered_results", [])
        
        # Prepare final result structure
        final_output = {
            "query": query_data.get("primary_query"),
            "results": results,
            "metadata": {
                "search_summary": {
                    "engines_used": query_data.get("selected_engines", []),
                    "total_raw_results": query_data.get("total_raw_results", 0),
                    "after_deduplication": len(query_data.get("deduplicated_results", [])),
                    "after_quality_filter": len(query_data.get("quality_results", [])),
                    "final_count": len(results)
                },
                "engine_performance": query_data.get("engine_stats", {}),
                "deduplication_stats": query_data.get("deduplication_stats", {}),
                "quality_filter_stats": query_data.get("quality_filter_stats", {}),
                "processing_completed_at": datetime.now().isoformat()
            }
        }
        
        return final_output
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize search query"""
        if not query:
            return ""
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Remove problematic characters for URLs
        query = re.sub(r'[<>{}|\\^`\[\]]', '', query)
        
        return query
    
    async def _fallback_search(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Fallback search strategy when primary search fails"""
        try:
            # Extract query from data
            if isinstance(data, dict) and "primary_query" in data:
                query = data["primary_query"]
            else:
                query = str(data)
            
            # Simple fallback: try DuckDuckGo as it's usually most reliable
            fallback_results = await self._search_single_engine("duckduckgo", query, 5)
            
            return ToolResult(
                data=fallback_results,
                metadata={"fallback_strategy": "duckduckgo_only"},
                success=True,
                tool_name="fallback_search"
            )
            
        except Exception as e:
            return ToolResult(
                data=[],
                metadata={"fallback_error": str(e)},
                success=False,
                error=f"Fallback search failed: {e}",
                tool_name="fallback_search"
            )
    
    async def _handle_search_errors(self, failed_result: ToolResult, data: Any, context: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Handle search execution errors with recovery strategies"""
        logger.warning(f"Search execution failed: {failed_result.error}")
        
        # Try to extract any partial results
        raw_results = []
        if isinstance(data, dict) and "raw_results" in data:
            raw_results = data["raw_results"]
        
        if raw_results:
            # We have some results, continue with reduced dataset
            recovery_data = data.copy()
            recovery_data["raw_results"] = raw_results
            return ToolResult(
                data=recovery_data,
                metadata={"recovery_strategy": "partial_results", "original_error": failed_result.error},
                success=True,
                tool_name="error_recovery"
            )
        else:
            # No results available, try fallback
            return await self._fallback_search(data, context)