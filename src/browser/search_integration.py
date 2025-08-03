#!/usr/bin/env python3
"""
Enhanced Search Integration with Advanced CAPTCHA and Anti-Bot Bypass
Integrates the advanced anti-detection browser with the existing search pipeline
"""

import asyncio
import json
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

try:
    from rich.console import Console
    console = Console()
except ImportError:
    class MockConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = MockConsole()

from .anti_detection import AntiDetectionBrowserManager
from .captcha_solver import AdvancedCaptchaSolver
from ..tools.composition.search_pipeline import EnhancedSearchPipeline

class AdvancedSearchManager:
    """
    Advanced search manager that combines stealth browsing with intelligent search
    """
    
    def __init__(self,
                 headless: bool = True,
                 llm_endpoint: str = "http://localhost:11434/api/generate",
                 llm_model: str = "granite3.3:8b",
                 proxy_config: Optional[Dict[str, str]] = None):
        
        self.headless = headless
        self.proxy_config = proxy_config
        
        # Initialize components
        self.browser_manager = AntiDetectionBrowserManager(
            llm_endpoint=llm_endpoint,
            llm_model=llm_model
        )
        
        self.search_pipeline = EnhancedSearchPipeline(headless=headless)
        
        # Active search sessions
        self.active_sessions = {}
        self.search_results_cache = {}
        
        console.print("[green]🚀 Advanced Search Manager initialized[/green]")
        console.print(f"[cyan]   • Headless mode: {headless}[/cyan]")
        console.print(f"[cyan]   • LLM Model: {llm_model}[/cyan]")
        console.print(f"[cyan]   • Proxy: {'enabled' if proxy_config else 'disabled'}[/cyan]")
    
    async def create_search_session(self, 
                                  session_name: Optional[str] = None,
                                  enable_visual_solving: bool = True) -> str:
        """Create a new advanced search session with anti-detection capabilities"""
        
        if not session_name:
            session_name = f"search_session_{int(time.time())}_{random.randint(1000, 9999)}"
        
        try:
            # Create stealth browser session
            browser_session_id = await self.browser_manager.create_stealth_browser(
                headless=self.headless,
                proxy_config=self.proxy_config
            )
            
            # Create stealth page for searching
            page_id = await self.browser_manager.create_stealth_page(browser_session_id)
            
            # Store session information
            self.active_sessions[session_name] = {
                'browser_session_id': browser_session_id,
                'page_id': page_id,
                'created_at': datetime.now(),
                'searches_performed': 0,
                'captchas_solved': 0,
                'success_rate': 1.0,
                'enable_visual_solving': enable_visual_solving,
                'last_activity': datetime.now()
            }
            
            console.print(f"[green]✅ Created advanced search session: {session_name}[/green]")
            return session_name
            
        except Exception as e:
            console.print(f"[red]❌ Failed to create search session: {e}[/red]")
            raise
    
    async def perform_advanced_search(self,
                                    session_name: str,
                                    query: str,
                                    max_results: int = 15,
                                    engines: Optional[List[str]] = None,
                                    handle_captchas: bool = True,
                                    deep_search: bool = False) -> List[Dict[str, Any]]:
        """Perform advanced search with anti-detection and CAPTCHA handling"""
        
        if session_name not in self.active_sessions:
            raise ValueError(f"Search session {session_name} not found")
        
        session_data = self.active_sessions[session_name]
        page_id = session_data['page_id']
        
        console.print(f"[blue]🔍 Starting advanced search for: {query}[/blue]")
        console.print(f"[cyan]   • Session: {session_name}[/cyan]")
        console.print(f"[cyan]   • Max results: {max_results}[/cyan]")
        console.print(f"[cyan]   • Engines: {engines or 'auto-select'}[/cyan]")
        console.print(f"[cyan]   • CAPTCHA handling: {handle_captchas}[/cyan]")
        
        try:
            # Update session activity
            session_data['last_activity'] = datetime.now()
            session_data['searches_performed'] += 1
            
            # Prepare search engines
            search_engines = engines or ['google', 'bing', 'duckduckgo']
            all_results = []
            
            # Perform searches on each engine with anti-detection
            for engine in search_engines:
                try:
                    console.print(f"[cyan]🌐 Searching {engine.capitalize()}...[/cyan]")
                    
                    engine_results = await self._search_engine_with_protection(
                        page_id=page_id,
                        engine=engine,
                        query=query,
                        max_results=max_results // len(search_engines) + 2,
                        handle_captchas=handle_captchas
                    )
                    
                    if engine_results:
                        console.print(f"[green]✅ {engine.capitalize()}: {len(engine_results)} results[/green]")
                        all_results.extend(engine_results)
                    else:
                        console.print(f"[yellow]⚠️ {engine.capitalize()}: No results[/yellow]")
                        
                    # Brief delay between engines to avoid rate limiting
                    await asyncio.sleep(random.uniform(2, 5))
                    
                except Exception as e:
                    console.print(f"[red]❌ {engine.capitalize()} search failed: {e}[/red]")
                    continue
            
            # Process and enhance results
            if all_results:
                processed_results = await self._process_search_results(
                    all_results, query, max_results, deep_search
                )
                
                # Cache results
                cache_key = f"{session_name}_{hash(query)}_{max_results}"
                self.search_results_cache[cache_key] = {
                    'results': processed_results,
                    'timestamp': datetime.now(),
                    'query': query,
                    'session': session_name
                }
                
                # Update session success rate
                if processed_results:
                    session_data['success_rate'] = min(
                        1.0, 
                        session_data['success_rate'] * 0.9 + 0.1
                    )
                
                console.print(f"[green]🎯 Advanced search completed: {len(processed_results)} results[/green]")
                return processed_results
            else:
                console.print("[yellow]⚠️ No results found from any engine[/yellow]")
                
                # Update session success rate
                session_data['success_rate'] *= 0.8
                
                # Fallback to pipeline search if browser search fails
                console.print("[cyan]🔄 Falling back to pipeline search...[/cyan]")
                return await self._fallback_pipeline_search(query, max_results)
                
        except Exception as e:
            console.print(f"[red]❌ Advanced search failed: {e}[/red]")
            session_data['success_rate'] *= 0.7
            
            # Fallback to pipeline search
            return await self._fallback_pipeline_search(query, max_results)
    
    async def _search_engine_with_protection(self,
                                           page_id: str,
                                           engine: str,
                                           query: str,
                                           max_results: int,
                                           handle_captchas: bool) -> List[Dict[str, Any]]:
        """Search a specific engine with full protection and CAPTCHA handling"""
        
        try:
            # Define search URLs for different engines
            search_urls = {
                'google': f'https://www.google.com/search?q={query.replace(" ", "+")}',
                'bing': f'https://www.bing.com/search?q={query.replace(" ", "+")}',
                'duckduckgo': f'https://duckduckgo.com/?q={query.replace(" ", "+")}&t=h_&ia=web',
                'yandex': f'https://yandex.com/search/?text={query.replace(" ", "+")}',
                'baidu': f'https://www.baidu.com/s?wd={query.replace(" ", "+")}'
            }
            
            if engine not in search_urls:
                console.print(f"[yellow]⚠️ Unsupported engine: {engine}[/yellow]")
                return []
            
            search_url = search_urls[engine]
            
            # Navigate to search engine with protection
            success = await self.browser_manager.navigate_with_protection(
                page_id=page_id,
                url=search_url,
                handle_captcha=handle_captchas,
                max_retries=3
            )
            
            if not success:
                console.print(f"[red]❌ Failed to navigate to {engine}[/red]")
                return []
            
            # Wait for search results to load
            await asyncio.sleep(random.uniform(2, 4))
            
            # Extract search results using engine-specific methods
            page_data = self.browser_manager.active_pages[page_id]
            page = page_data['page']
            
            results = await self._extract_search_results(page, engine, max_results)
            
            # Add metadata to results
            for result in results:
                result['search_engine'] = engine
                result['search_timestamp'] = datetime.now().isoformat()
                result['search_method'] = 'advanced_browser'
                result['anti_detection'] = True
            
            return results
            
        except Exception as e:
            console.print(f"[red]❌ Engine search failed for {engine}: {e}[/red]")
            return []
    
    async def _extract_search_results(self, page, engine: str, max_results: int) -> List[Dict[str, Any]]:
        """Extract search results from a search engine page"""
        
        try:
            results = []
            
            if engine == 'google':
                # Google search result extraction
                result_elements = await page.query_selector_all('div.g, div[data-ved]')
                
                for i, element in enumerate(result_elements[:max_results]):
                    try:
                        # Extract title
                        title_elem = await element.query_selector('h3')
                        title = await title_elem.inner_text() if title_elem else f"Result {i+1}"
                        
                        # Extract URL
                        link_elem = await element.query_selector('a[href]')
                        url = await link_elem.get_attribute('href') if link_elem else ""
                        
                        # Clean Google redirect URLs
                        if url and '/url?q=' in url:
                            import urllib.parse
                            url = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get('q', [url])[0]
                        
                        # Extract snippet
                        snippet_elem = await element.query_selector('[data-sncf="1"], .VwiC3b, .s3v9rd')
                        snippet = await snippet_elem.inner_text() if snippet_elem else "No description available"
                        
                        if title and url and len(title) > 5:
                            results.append({
                                'title': title[:150],
                                'url': url,
                                'snippet': snippet[:300],
                                'credibility_score': self._calculate_credibility_score(url, title),
                                'source_type': self._determine_source_type(url),
                                'date': datetime.now().strftime('%Y-%m-%d')
                            })
                            
                    except Exception as e:
                        console.print(f"[yellow]⚠️ Error extracting Google result {i+1}: {e}[/yellow]")
                        continue
            
            elif engine == 'bing':
                # Bing search result extraction
                result_elements = await page.query_selector_all('li.b_algo')
                
                for i, element in enumerate(result_elements[:max_results]):
                    try:
                        # Extract title
                        title_elem = await element.query_selector('h2 a')
                        title = await title_elem.inner_text() if title_elem else f"Result {i+1}"
                        
                        # Extract URL
                        url = await title_elem.get_attribute('href') if title_elem else ""
                        
                        # Extract snippet
                        snippet_elem = await element.query_selector('.b_caption p')
                        snippet = await snippet_elem.inner_text() if snippet_elem else "No description available"
                        
                        if title and url and len(title) > 5:
                            results.append({
                                'title': title[:150],
                                'url': url,
                                'snippet': snippet[:300],
                                'credibility_score': self._calculate_credibility_score(url, title),
                                'source_type': self._determine_source_type(url),
                                'date': datetime.now().strftime('%Y-%m-%d')
                            })
                            
                    except Exception as e:
                        console.print(f"[yellow]⚠️ Error extracting Bing result {i+1}: {e}[/yellow]")
                        continue
            
            elif engine == 'duckduckgo':
                # DuckDuckGo search result extraction
                result_elements = await page.query_selector_all('article[data-testid="result"]')
                
                for i, element in enumerate(result_elements[:max_results]):
                    try:
                        # Extract title
                        title_elem = await element.query_selector('h2 a')
                        title = await title_elem.inner_text() if title_elem else f"Result {i+1}"
                        
                        # Extract URL
                        url = await title_elem.get_attribute('href') if title_elem else ""
                        
                        # Extract snippet
                        snippet_elem = await element.query_selector('[data-result="snippet"]')
                        snippet = await snippet_elem.inner_text() if snippet_elem else "No description available"
                        
                        if title and url and len(title) > 5:
                            results.append({
                                'title': title[:150],
                                'url': url,
                                'snippet': snippet[:300],
                                'credibility_score': self._calculate_credibility_score(url, title),
                                'source_type': self._determine_source_type(url),
                                'date': datetime.now().strftime('%Y-%m-%d')
                            })
                            
                    except Exception as e:
                        console.print(f"[yellow]⚠️ Error extracting DuckDuckGo result {i+1}: {e}[/yellow]")
                        continue
            
            return results
            
        except Exception as e:
            console.print(f"[red]❌ Failed to extract results from {engine}: {e}[/red]")
            return []
    
    def _calculate_credibility_score(self, url: str, title: str) -> float:
        """Calculate credibility score for a search result"""
        
        score = 0.5  # Base score
        
        # High credibility domains
        high_cred_domains = [
            '.edu', '.gov', 'wikipedia.org', 'arxiv.org', 'nature.com',
            'science.org', 'ieee.org', 'acm.org', 'nih.gov', 'who.int',
            'pubmed.ncbi.nlm.nih.gov'
        ]
        
        # Medium credibility domains
        med_cred_domains = [
            '.org', 'reuters.com', 'bbc.com', 'npr.org', 'pbs.org',
            'techcrunch.com', 'wired.com', 'arstechnica.com'
        ]
        
        url_lower = url.lower()
        title_lower = title.lower()
        
        # Domain-based scoring
        for domain in high_cred_domains:
            if domain in url_lower:
                score += 0.3
                break
        else:
            for domain in med_cred_domains:
                if domain in url_lower:
                    score += 0.15
                    break
        
        # Content quality indicators
        quality_indicators = [
            'research', 'study', 'analysis', 'report', 'journal',
            'academic', 'scientific', 'peer-reviewed', 'methodology'
        ]
        
        indicator_count = sum(1 for indicator in quality_indicators if indicator in title_lower)
        score += min(0.2, indicator_count * 0.05)
        
        # Avoid promotional content
        promotional_indicators = [
            'buy', 'shop', 'sale', 'discount', 'advertisement',
            'sponsored', 'promotion', 'deal'
        ]
        
        if any(indicator in title_lower for indicator in promotional_indicators):
            score -= 0.1
        
        return min(max(score, 0.0), 1.0)
    
    def _determine_source_type(self, url: str) -> str:
        """Determine the type of source based on URL"""
        
        url_lower = url.lower()
        
        if any(domain in url_lower for domain in ['.edu', 'arxiv.org', 'nature.com', 'science.org']):
            return 'academic'
        elif any(domain in url_lower for domain in ['.gov', 'who.int', 'nih.gov']):
            return 'government'
        elif 'wikipedia.org' in url_lower:
            return 'encyclopedia'
        elif any(domain in url_lower for domain in ['reuters.com', 'bbc.com', 'npr.org']):
            return 'news'
        elif any(domain in url_lower for domain in ['blog', 'medium.com', 'substack.com']):
            return 'blog'
        else:
            return 'web'
    
    async def _process_search_results(self, 
                                    raw_results: List[Dict[str, Any]], 
                                    query: str, 
                                    max_results: int,
                                    deep_search: bool) -> List[Dict[str, Any]]:
        """Process and enhance search results"""
        
        try:
            # Remove duplicates based on URL
            seen_urls = set()
            unique_results = []
            
            for result in raw_results:
                url = result.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_results.append(result)
            
            # Sort by credibility score and relevance
            sorted_results = sorted(
                unique_results,
                key=lambda x: (
                    x.get('credibility_score', 0.5),
                    len(x.get('title', '')),
                    -len(x.get('url', ''))  # Prefer shorter URLs
                ),
                reverse=True
            )
            
            # Add ranking information
            for i, result in enumerate(sorted_results[:max_results]):
                result['rank'] = i + 1
                result['relevance_score'] = self._calculate_relevance_score(result, query)
                result['confidence_score'] = (
                    result.get('credibility_score', 0.5) * 0.6 +
                    result.get('relevance_score', 0.5) * 0.4
                )
            
            # If deep search is enabled, perform additional analysis
            if deep_search and len(sorted_results) > 0:
                console.print("[cyan]🔬 Performing deep analysis...[/cyan]")
                enhanced_results = await self._perform_deep_analysis(sorted_results[:max_results])
                return enhanced_results
            
            return sorted_results[:max_results]
            
        except Exception as e:
            console.print(f"[yellow]⚠️ Error processing results: {e}[/yellow]")
            return raw_results[:max_results]
    
    def _calculate_relevance_score(self, result: Dict[str, Any], query: str) -> float:
        """Calculate relevance score based on query matching"""
        
        title = result.get('title', '').lower()
        snippet = result.get('snippet', '').lower()
        query_lower = query.lower()
        query_words = query_lower.split()
        
        score = 0.0
        
        # Exact query match in title
        if query_lower in title:
            score += 0.4
        
        # Individual word matches in title
        title_matches = sum(1 for word in query_words if word in title)
        score += (title_matches / len(query_words)) * 0.3
        
        # Exact query match in snippet
        if query_lower in snippet:
            score += 0.2
        
        # Individual word matches in snippet
        snippet_matches = sum(1 for word in query_words if word in snippet)
        score += (snippet_matches / len(query_words)) * 0.1
        
        return min(score, 1.0)
    
    async def _perform_deep_analysis(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform deep analysis on search results using AI"""
        
        # This is a placeholder for advanced AI analysis
        # In a full implementation, this could include:
        # - Content extraction and summarization
        # - Fact-checking
        # - Bias detection
        # - Source verification
        
        for result in results:
            # Add deep analysis metadata
            result['deep_analysis'] = {
                'analyzed_at': datetime.now().isoformat(),
                'content_extracted': False,  # Would be True if content was extracted
                'summary_available': False,   # Would be True if summary was generated
                'fact_checked': False,        # Would be True if fact-checked
                'bias_detected': None         # Would contain bias analysis
            }
        
        return results
    
    async def _fallback_pipeline_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Fallback to the standard search pipeline"""
        
        try:
            console.print("[cyan]🔄 Using fallback search pipeline...[/cyan]")
            
            # Use the enhanced search pipeline as fallback
            query_data = {
                'query': query,
                'max_results': max_results,
                'primary_query': query
            }
            
            result = await self.search_pipeline.execute(query_data)
            
            if result.success and result.data:
                pipeline_results = result.data.get('results', [])
                console.print(f"[green]✅ Pipeline search: {len(pipeline_results)} results[/green]")
                return pipeline_results
            else:
                console.print("[yellow]⚠️ Pipeline search also failed[/yellow]")
                return []
                
        except Exception as e:
            console.print(f"[red]❌ Fallback search failed: {e}[/red]")
            return []
    
    async def close_search_session(self, session_name: str):
        """Close a search session and cleanup resources"""
        
        if session_name in self.active_sessions:
            try:
                session_data = self.active_sessions[session_name]
                browser_session_id = session_data['browser_session_id']
                
                # Close browser session
                await self.browser_manager.close_browser(browser_session_id)
                
                # Remove from active sessions
                del self.active_sessions[session_name]
                
                console.print(f"[green]🗑️ Closed search session: {session_name}[/green]")
                
            except Exception as e:
                console.print(f"[yellow]⚠️ Error closing session {session_name}: {e}[/yellow]")
    
    async def cleanup_all_sessions(self):
        """Cleanup all active search sessions"""
        
        console.print("[blue]🧹 Cleaning up all search sessions...[/blue]")
        
        session_names = list(self.active_sessions.keys())
        for session_name in session_names:
            await self.close_search_session(session_name)
        
        # Cleanup browser manager
        await self.browser_manager.cleanup_all()
        
        # Clear cache
        self.search_results_cache.clear()
        
        console.print("[green]✅ All sessions cleaned up[/green]")
    
    async def get_session_status(self) -> Dict[str, Any]:
        """Get status of all active search sessions"""
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'active_sessions': len(self.active_sessions),
            'cached_results': len(self.search_results_cache),
            'browser_status': await self.browser_manager.get_status_report(),
            'sessions': {}
        }
        
        for session_name, session_data in self.active_sessions.items():
            status['sessions'][session_name] = {
                'created_at': session_data['created_at'].isoformat(),
                'searches_performed': session_data['searches_performed'],
                'captchas_solved': session_data['captchas_solved'],
                'success_rate': session_data['success_rate'],
                'last_activity': session_data['last_activity'].isoformat()
            }
        
        return status

# Usage example
async def demo_advanced_search():
    """Demonstration of the advanced search capabilities"""
    
    console.print("[blue]🚀 Starting Advanced Search Demo[/blue]")
    
    # Initialize the advanced search manager
    search_manager = AdvancedSearchManager(
        headless=False,  # Set to True for headless operation
        proxy_config=None  # Add proxy config if needed
    )
    
    try:
        # Create a search session
        session_name = await search_manager.create_search_session("demo_session")
        
        # Perform advanced searches
        test_queries = [
            "artificial intelligence safety research 2024",
            "climate change mitigation strategies",
            "quantum computing breakthroughs"
        ]
        
        for query in test_queries:
            console.print(f"\n[cyan]🔍 Searching: {query}[/cyan]")
            
            results = await search_manager.perform_advanced_search(
                session_name=session_name,
                query=query,
                max_results=10,
                engines=['google', 'bing', 'duckduckgo'],
                handle_captchas=True,
                deep_search=True
            )
            
            console.print(f"[green]📊 Found {len(results)} results[/green]")
            
            # Display top results
            for i, result in enumerate(results[:3]):
                console.print(f"   {i+1}. {result['title'][:60]}...")
                console.print(f"      Score: {result.get('confidence_score', 0):.2f}")
        
        # Get session status
        status = await search_manager.get_session_status()
        console.print(f"\n[blue]📈 Session Status:[/blue]")
        console.print(f"   • Active sessions: {status['active_sessions']}")
        console.print(f"   • Cached results: {status['cached_results']}")
        
    finally:
        # Cleanup
        await search_manager.cleanup_all_sessions()
        console.print("[green]🎯 Demo completed successfully[/green]")

if __name__ == "__main__":
    asyncio.run(demo_advanced_search())
