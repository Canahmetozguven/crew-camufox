#!/usr/bin/env python3
"""
Deep Researcher Agent
Specialized in comprehensive web research and content extraction
"""

import asyncio
import json
import random
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
import re

from bs4 import BeautifulSoup
from camoufox import AsyncCamoufox
from crewai import Agent
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.models.research_models import ResearchSource, BrowserSession
from src.utils.helpers import extract_text_content, detect_source_type, calculate_credibility_score

console = Console()


class DeepResearcherAgent:
    """Agent responsible for comprehensive web research and content extraction"""

    def __init__(
        self,
        model_name: str = "magistral:latest",
        browser_model_name: str = "granite3.3:8b",
        ollama_base_url: str = "http://localhost:11434",
        headless: bool = True,
        proxy: Optional[Dict[str, str]] = None,
    ):
        self.model_name = model_name
        self.browser_model_name = browser_model_name
        self.ollama_base_url = ollama_base_url
        self.headless = headless
        self.proxy = proxy

        # Main LLM for general research tasks (magistral)
        self.llm = ChatOllama(
            model=model_name,
            base_url=ollama_base_url,
            temperature=0.1,
            num_ctx=40000,  # Extended context for comprehensive research
        )

        # Browser-specific LLM for content analysis and extraction (granite3.3:8b)
        self.browser_llm = ChatOllama(
            model=browser_model_name,
            base_url=ollama_base_url,
            temperature=0.1,
            num_ctx=40000,  # Extended context for better content processing
        )

        console.print(f"[cyan]🤖 Main Model: {model_name}[/cyan]")
        console.print(f"[cyan]🌐 Browser Model: {browser_model_name}[/cyan]")

        # Initialize the researcher agent with main LLM
        self.agent = Agent(
            role="Multi-Engine Web Researcher",
            goal="Execute comprehensive web research using multiple search engines with stealth browsing",
            backstory=f"""You are an elite cyber-researcher with expertise in multi-engine web scraping,
            stealth browsing, and information extraction. You operate with a dual-model architecture:
            
            - Main Intelligence ({model_name}): Research planning, analysis, and synthesis
            - Browser Intelligence ({browser_model_name}): Fast content processing and extraction
            
            With access to Google, Google Scholar, Bing, and DuckDuckGo through advanced browser automation, you excel at:
            
            - Multi-engine web scraping (Google, Scholar, Bing, DuckDuckGo)
            - Anti-detection browsing with Camoufox
            - Academic source discovery via Google Scholar  
            - Fast content processing with optimized models
            - Content quality assessment and filtering
            - Cross-referencing and fact verification
            - Deep link analysis and citation mapping
            - Multi-lingual content processing
            
            You approach each research task methodically, leveraging multiple search engines and
            optimized AI models while maintaining the highest standards of accuracy and source verification.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,  # Use main LLM for agent operations
        )

    async def execute_research_plan(self, research_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a comprehensive research plan"""

        session_id = f"research_{int(datetime.now().timestamp())}"
        console.print(
            f"[bold blue]🔍 Executing Research Plan: {research_plan['query']}[/bold blue]"
        )
        console.print(f"[yellow]Session ID: {session_id}[/yellow]")
        console.print(f"[yellow]Depth: {research_plan['research_depth']}[/yellow]")

        research_results = {
            "session_id": session_id,
            "plan_id": research_plan.get("id"),
            "query": research_plan["query"],
            "started_at": datetime.now().isoformat(),
            "sources": [],
            "failed_urls": [],
            "phase_results": [],
            "quality_metrics": {},
            "completion_status": "in_progress",
        }

        try:
            # Execute research in phases with timeouts and better error handling
            successful_phases = 0
            total_sources_found = 0

            for phase in research_plan["execution_phases"]:
                console.print(f"\n[cyan]📋 Phase {phase['phase']}: {phase['name']}[/cyan]")

                try:
                    # Extended timeout per phase to 5 minutes for better source collection
                    phase_result = await asyncio.wait_for(
                        self._execute_research_phase(phase, research_plan, research_results),
                        timeout=300.0,  # 5 minute timeout per phase
                    )
                    research_results["phase_results"].append(phase_result)

                    # Update sources list
                    new_sources = phase_result.get("sources", [])
                    research_results["sources"].extend(new_sources)
                    research_results["failed_urls"].extend(phase_result.get("failed_urls", []))

                    successful_phases += 1
                    total_sources_found += len(new_sources)
                    console.print(
                        f"[green]✅ Phase {phase['phase']} completed: {len(new_sources)} sources found[/green]"
                    )

                except asyncio.TimeoutError:
                    console.print(f"[red]❌ Phase {phase['phase']} timed out after 5 minutes[/red]")
                    console.print(
                        f"[yellow]⚠️ Skipping Phase {phase['phase']} and continuing with available data...[/yellow]"
                    )

                    phase_result = {
                        "phase": phase["phase"],
                        "name": phase["name"],
                        "status": "timeout",
                        "sources": [],
                        "failed_urls": [],
                        "error": "Phase execution timed out",
                        "timeout_duration": "5 minutes",
                    }
                    research_results["phase_results"].append(phase_result)

                    # Don't break - continue with next phase
                    continue

                except Exception as e:
                    console.print(f"[red]❌ Phase {phase['phase']} failed: {e}[/red]")
                    console.print(
                        f"[yellow]⚠️ Skipping Phase {phase['phase']} and continuing with available data...[/yellow]"
                    )

                    phase_result = {
                        "phase": phase["phase"],
                        "name": phase["name"],
                        "status": "failed",
                        "sources": [],
                        "failed_urls": [],
                        "error": str(e),
                    }
                    research_results["phase_results"].append(phase_result)

                    # Don't break - continue with next phase
                    continue

                # If we have enough sources from successful phases, we can proceed
                if total_sources_found >= 5:  # Minimum viable sources
                    console.print(
                        f"[green]✅ Sufficient sources found ({total_sources_found}), proceeding to analysis...[/green]"
                    )
                    break

            # Evaluate if we have enough data to continue
            final_source_count = len(research_results["sources"])

            if final_source_count == 0:
                console.print(
                    f"[red]❌ No sources found in any phase - research failed completely[/red]"
                )
                research_results["completion_status"] = "failed"
                research_results["error"] = "No sources could be extracted from any research phase"
                research_results["failed_at"] = datetime.now().isoformat()
                return research_results

            elif final_source_count < 3:
                console.print(
                    f"[yellow]⚠️ Limited sources found ({final_source_count}) - results may be incomplete[/yellow]"
                )
                research_results["completion_status"] = "partial"
                research_results["warning"] = (
                    f"Only {final_source_count} sources found, results may be limited"
                )

            else:
                console.print(
                    f"[green]✅ Adequate sources found ({final_source_count}) - proceeding with analysis[/green]"
                )
                research_results["completion_status"] = "completed"

            # Post-processing
            await self._post_process_results(research_results, research_plan)

            research_results["completion_status"] = "completed"
            research_results["completed_at"] = datetime.now().isoformat()

            console.print(f"[green]✅ Research completed successfully![/green]")
            console.print(f"[green]📊 Total sources: {len(research_results['sources'])}[/green]")

            return research_results

        except Exception as e:
            console.print(f"[red]❌ Research failed: {e}[/red]")
            research_results["completion_status"] = "failed"
            research_results["error"] = str(e)
            research_results["failed_at"] = datetime.now().isoformat()
            return research_results

    async def _execute_research_phase(
        self, phase: Dict[str, Any], research_plan: Dict[str, Any], research_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single research phase"""

        phase_result = {
            "phase": phase["phase"],
            "name": phase["name"],
            "started_at": datetime.now().isoformat(),
            "sources": [],
            "failed_urls": [],
            "metrics": {},
        }

        config = research_plan["config"]
        search_strategies = research_plan["search_strategies"]

        # Use context manager for proper browser lifecycle
        async with AsyncCamoufox(
            headless=self.headless,
            proxy=self.proxy,
            geoip=True,
            humanize=True,
            block_images=True,
            i_know_what_im_doing=True,  # Suppress WAF detection warnings for image blocking
            locale=["en-US", "en"],
            args=[
                "--ignore-certificate-errors",
                "--ignore-ssl-errors",
                "--ignore-certificate-errors-spki-list",
            ],
        ) as browser:

            page = await browser.new_page()

            try:
                # Execute searches based on phase
                if phase["phase"] == 1:
                    # Initial Discovery Phase
                    sources = await self._execute_initial_searches(
                        page, search_strategies["primary_terms"], config["sources_per_round"]
                    )
                elif phase["phase"] == 2:
                    # Deep Analysis Phase
                    sources = await self._execute_secondary_searches(
                        page, search_strategies["secondary_terms"], config["sources_per_round"]
                    )
                else:
                    # Validation & Synthesis Phase
                    sources = await self._execute_validation_searches(
                        page, research_results["sources"], research_plan["fact_check_strategy"]
                    )

                # Process and analyze sources with timeout
                try:
                    processed_sources = await asyncio.wait_for(
                        self._process_sources(page, sources, research_plan["quality_criteria"]),
                        timeout=300.0,  # 5 minute timeout for processing sources
                    )
                except asyncio.TimeoutError:
                    console.print(f"[red]❌ Source processing timed out after 5 minutes[/red]")
                    processed_sources = []

                phase_result["sources"] = processed_sources
                phase_result["metrics"] = {
                    "sources_found": len(sources),
                    "sources_processed": len(processed_sources),
                    "success_rate": len(processed_sources) / max(len(sources), 1),
                }

            except Exception as e:
                console.print(f"[red]❌ Phase {phase['phase']} failed: {e}[/red]")
                phase_result["error"] = str(e)
            finally:
                await page.close()

        phase_result["completed_at"] = datetime.now().isoformat()
        return phase_result

    async def _execute_initial_searches(
        self, page, search_terms: List[str], max_sources: int
    ) -> List[str]:
        """Execute initial search queries to discover sources"""

        all_urls = []

        for term in search_terms[:3]:  # Limit to top 3 terms for initial phase
            console.print(f"[yellow]🔍 Searching for: {term}[/yellow]")

            try:
                urls = await self._search_multiple_engines(
                    page, term, max_sources // len(search_terms[:3])
                )
                all_urls.extend(urls)

                # Add delay between searches
                await asyncio.sleep(random.uniform(2, 4))

            except Exception as e:
                console.print(f"[red]❌ Search failed for '{term}': {e}[/red]")
                continue

        # Remove duplicates and return
        unique_urls = list(dict.fromkeys(all_urls))  # Preserves order
        console.print(f"[green]✅ Found {len(unique_urls)} unique URLs[/green]")

        return unique_urls[:max_sources]

    async def _execute_secondary_searches(
        self, page, search_terms: List[str], max_sources: int
    ) -> List[str]:
        """Execute secondary searches for deeper analysis"""

        all_urls = []

        # Use more specific search terms for deeper research
        for term in search_terms[:5]:  # More terms for secondary phase
            try:
                # Try different search engines/approaches
                urls = await self._search_multiple_engines(page, term, max_sources // 3)
                all_urls.extend(urls)

                # Add phrase search if applicable (without quotes to avoid search engine issues)
                if '"' not in term and " " in term and len(term.split()) > 1:
                    # Use the original term without quotes for better search engine compatibility
                    phrase_urls = await self._search_multiple_engines(page, term, max_sources // 6)
                    all_urls.extend(phrase_urls)

                await asyncio.sleep(random.uniform(2, 4))

            except Exception as e:
                console.print(f"[red]❌ Secondary search failed for '{term}': {e}[/red]")
                continue

        unique_urls = list(dict.fromkeys(all_urls))
        return unique_urls[:max_sources]

    async def _execute_validation_searches(
        self, page, existing_sources: List[Dict[str, Any]], fact_check_strategy: Dict[str, Any]
    ) -> List[str]:
        """Execute validation searches for fact-checking"""

        if not fact_check_strategy.get("enabled", False):
            return []

        validation_urls = []

        # Extract key claims from existing sources for fact-checking
        key_claims = self._extract_key_claims(existing_sources)

        for claim in key_claims[:5]:  # Limit fact-checking queries
            try:
                # Search for verification of specific claims (without quotes for better search compatibility)
                fact_check_query = f"{claim} verification facts"
                urls = await self._search_multiple_engines(page, fact_check_query, 3)
                validation_urls.extend(urls)

                await asyncio.sleep(random.uniform(2, 4))

            except Exception as e:
                console.print(f"[red]❌ Fact-check search failed for claim: {e}[/red]")
                continue

        return validation_urls

    def _clean_query(self, query: str) -> str:
        """Clean search query by removing unnecessary quotes and formatting for search engines"""
        if not query:
            return ""

        # Remove surrounding quotes that might interfere with search engines
        query = query.strip()

        # Remove quotes from the beginning and end if they wrap the entire query
        if query.startswith('"') and query.endswith('"') and query.count('"') == 2:
            query = query[1:-1].strip()
        elif query.startswith("'") and query.endswith("'") and query.count("'") == 2:
            query = query[1:-1].strip()

        # Remove any remaining problematic characters that might interfere with URL encoding
        query = query.replace('"', "").replace("'", "")

        return query.strip()

    async def _search_duckduckgo(self, page, query: str, max_results: int) -> List[str]:
        """Search DuckDuckGo and extract result URLs"""

        # Clean the query to remove any problematic quotes
        clean_query = self._clean_query(query)
        search_url = f"https://duckduckgo.com/?q={clean_query.replace(' ', '+')}"

        try:
            await page.goto(search_url, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(3)  # Wait for dynamic content

            content = await page.content()
            soup = BeautifulSoup(content, "html.parser")

            urls = []

            # Extract search result URLs with better parsing
            for link in soup.find_all("a", href=True):
                try:
                    attrs = getattr(link, "attrs", {})
                    href = attrs.get("href") if attrs else None

                    if href and isinstance(href, str) and href.startswith("http"):
                        # Filter out unwanted domains
                        if not any(
                            exclude in href.lower()
                            for exclude in [
                                "duckduckgo.com",
                                "google.com",
                                "bing.com",
                                "facebook.com",
                                "twitter.com",
                                "instagram.com",
                                "youtube.com",
                                "tiktok.com",
                                "pinterest.com",
                                "amazon.com",
                                "ebay.com",
                                "ads.",
                            ]
                        ):
                            urls.append(href)
                            if len(urls) >= max_results:
                                break

                except (AttributeError, TypeError):
                    continue

            return urls

        except Exception as e:
            console.print(f"[red]❌ DuckDuckGo search failed: {e}[/red]")
            return []

    async def _search_google(self, page, query: str, max_results: int) -> List[str]:
        """Search Google.com and extract result URLs"""

        # Clean the query to remove any problematic quotes
        clean_query = self._clean_query(query)
        search_url = f"https://www.google.com/search?q={clean_query.replace(' ', '+')}"

        try:
            await page.goto(search_url, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(3)  # Wait for dynamic content

            # Handle potential consent/cookie popup
            try:
                # Try to accept cookies/consent if present
                accept_button = await page.query_selector(
                    'button[id*="accept"], button[id*="agree"], div[role="button"] >> text="Accept"'
                )
                if accept_button:
                    await accept_button.click()
                    await asyncio.sleep(2)
            except:
                pass  # Ignore if no consent popup

            content = await page.content()
            soup = BeautifulSoup(content, "html.parser")

            urls = []

            # Extract search result URLs from Google results
            # Google uses different selectors, try multiple patterns
            result_selectors = [
                'div[data-sokoban-container] a[href^="http"]',  # Main results
                'a[href^="/url?q="]',  # URL redirects
                'h3 a[href^="http"]',  # Title links
                'div[data-ved] a[href^="http"]',  # General result links
                'a[ping^="/url?"]',  # Alternative format
            ]

            for selector in result_selectors:
                try:
                    for link in soup.select(selector):
                        href = link.get("href", "")

                        # Ensure href is a string
                        if not isinstance(href, str):
                            continue

                        # Handle Google's URL redirects
                        if href.startswith("/url?q="):
                            # Extract actual URL from Google's redirect
                            import urllib.parse

                            try:
                                parsed = urllib.parse.parse_qs(href[7:])  # Remove '/url?q='
                                if "q" in parsed and parsed["q"]:
                                    href = parsed["q"][0]
                            except:
                                continue

                        # Clean and validate URL
                        if href and href.startswith("http"):
                            # Filter out unwanted domains
                            if not any(
                                exclude in href.lower()
                                for exclude in [
                                    "google.com",
                                    "googleusercontent.com",
                                    "youtube.com",
                                    "facebook.com",
                                    "twitter.com",
                                    "instagram.com",
                                    "tiktok.com",
                                    "pinterest.com",
                                    "amazon.com",
                                    "ebay.com",
                                    "ads.",
                                    "doubleclick.net",
                                    "googlesyndication.com",
                                ]
                            ):
                                if href not in urls:  # Avoid duplicates
                                    urls.append(href)
                                    if len(urls) >= max_results:
                                        break

                    if len(urls) >= max_results:
                        break

                except Exception as e:
                    console.print(f"[yellow]Warning: Selector {selector} failed: {e}[/yellow]")
                    continue

            console.print(f"[green]✅ Google search found {len(urls)} URLs[/green]")
            return urls

        except Exception as e:
            console.print(f"[red]❌ Google search failed: {e}[/red]")
            return []

    async def _search_google_scholar(self, page, query: str, max_results: int) -> List[str]:
        """Search Google Scholar and extract academic result URLs"""

        # Clean the query to remove any problematic quotes
        clean_query = self._clean_query(query)
        search_url = f"https://scholar.google.com/scholar?q={clean_query.replace(' ', '+')}"

        try:
            await page.goto(search_url, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(3)  # Wait for dynamic content

            content = await page.content()
            soup = BeautifulSoup(content, "html.parser")

            urls = []

            # Extract academic paper URLs from Google Scholar
            # Scholar has different structure focused on academic papers
            result_selectors = [
                'h3.gs_rt a[href^="http"]',  # Paper title links
                'div.gs_r div.gs_ggs a[href^="http"]',  # PDF and other links
                'div.gs_or_ggsm a[href^="http"]',  # Additional paper links
                'span.gs_ctg2 a[href^="http"]',  # Related links
            ]

            for selector in result_selectors:
                try:
                    for link in soup.select(selector):
                        href = link.get("href", "")

                        # Ensure href is a string
                        if not isinstance(href, str):
                            continue

                        # Clean and validate URL
                        if href and href.startswith("http"):
                            # Prioritize academic domains for Scholar
                            academic_domains = [
                                ".edu",
                                ".org",
                                ".gov",
                                "arxiv.org",
                                "pubmed.ncbi.nlm.nih.gov",
                                "ieee.org",
                                "acm.org",
                                "springer.com",
                                "wiley.com",
                                "elsevier.com",
                                "nature.com",
                                "science.org",
                                "jstor.org",
                            ]

                            # Filter out non-academic and unwanted domains
                            if not any(
                                exclude in href.lower()
                                for exclude in [
                                    "google.com",
                                    "googleusercontent.com",
                                    "youtube.com",
                                    "facebook.com",
                                    "twitter.com",
                                    "instagram.com",
                                    "tiktok.com",
                                    "pinterest.com",
                                    "amazon.com",
                                    "ebay.com",
                                ]
                            ):
                                if href not in urls:  # Avoid duplicates
                                    urls.append(href)
                                    if len(urls) >= max_results:
                                        break

                    if len(urls) >= max_results:
                        break

                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Scholar selector {selector} failed: {e}[/yellow]"
                    )
                    continue

            console.print(f"[green]✅ Google Scholar found {len(urls)} URLs[/green]")
            return urls

        except Exception as e:
            console.print(f"[red]❌ Google Scholar search failed: {e}[/red]")
            return []

    async def _search_bing(self, page, query: str, max_results: int) -> List[str]:
        """Search Bing.com and extract result URLs - Updated to use working method"""

        # Clean the query to remove any problematic quotes
        clean_query = self._clean_query(query)
        search_url = f"https://www.bing.com/search?q={clean_query.replace(' ', '+')}"

        try:
            await page.goto(search_url, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(5)  # Bing needs more time to load completely

            # Handle potential consent popup
            try:
                accept_button = await page.query_selector(
                    'button[id*="accept"], button[id*="bnp_btn_accept"]'
                )
                if accept_button:
                    await accept_button.click()
                    await asyncio.sleep(2)
            except:
                pass  # Ignore if no consent popup

            urls = []

            # Use Playwright selectors directly (more reliable than BeautifulSoup for dynamic content)
            try:
                # Wait a bit more for results to fully load
                await asyncio.sleep(2)
                
                # Get result elements using selectors we know work
                result_elements = await page.query_selector_all('li.b_algo')
                if not result_elements:
                    result_elements = await page.query_selector_all('.b_algo')

                console.print(f"[blue]🔍 Found {len(result_elements)} Bing result elements[/blue]")

                for i, element in enumerate(result_elements[:max_results * 2]):  # Get extra to filter
                    try:
                        # Extract title link
                        title_elem = await element.query_selector("h2 a")
                        if not title_elem:
                            continue

                        href = await title_elem.get_attribute("href")
                        if not href or len(href) < 10:
                            continue

                        # Clean Bing redirect URLs (basic cleanup)
                        if "bing.com/ck/a" in href:
                            # For now, accept Bing redirect URLs - they resolve correctly when visited
                            # More sophisticated URL extraction can be added later
                            pass
                        
                        # Validate URL
                        if href.startswith("http"):
                            # Filter out unwanted domains, but ALLOW bing.com redirects
                            if not any(
                                exclude in href.lower()
                                for exclude in [
                                    "microsoft.com",
                                    "msn.com", 
                                    "facebook.com",
                                    "twitter.com",
                                    "instagram.com",
                                    "youtube.com",
                                    "tiktok.com",
                                    "pinterest.com",
                                    "amazon.com",
                                    "ebay.com",
                                    "ads.",
                                ]
                            ):
                                if href not in urls:  # Avoid duplicates
                                    urls.append(href)
                                    if len(urls) >= max_results:
                                        break

                    except Exception as e:
                        console.print(f"[yellow]Warning: Bing element {i+1} extraction failed: {e}[/yellow]")
                        continue

            except Exception as e:
                console.print(f"[red]❌ Bing result extraction failed: {e}[/red]")

            console.print(f"[green]✅ Bing search found {len(urls)} URLs[/green]")
            return urls

        except Exception as e:
            console.print(f"[red]❌ Bing search failed: {e}[/red]")
            return []

    # Parallel search methods - each creates its own browser session
    async def _search_google_parallel(self, query: str, max_results: int) -> List[str]:
        """Search Google.com with dedicated browser session for parallel execution"""

        async with AsyncCamoufox(
            headless=self.headless,
            proxy=self.proxy,
            geoip=True,
            humanize=True,
            block_images=True,
            i_know_what_im_doing=True,  # Suppress WAF detection warnings for image blocking
            locale=["en-US", "en"],
        ) as browser:
            page = await browser.new_page()

            try:
                return await self._search_google(page, query, max_results)
            finally:
                await page.close()

    async def _search_google_scholar_parallel(self, query: str, max_results: int) -> List[str]:
        """Search Google Scholar with dedicated browser session for parallel execution"""

        async with AsyncCamoufox(
            headless=self.headless,
            proxy=self.proxy,
            geoip=True,
            humanize=True,
            block_images=True,
            i_know_what_im_doing=True,  # Suppress WAF detection warnings for image blocking
            locale=["en-US", "en"],
        ) as browser:
            page = await browser.new_page()

            try:
                return await self._search_google_scholar(page, query, max_results)
            finally:
                await page.close()

    async def _search_bing_parallel(self, query: str, max_results: int) -> List[str]:
        """Search Bing.com with dedicated browser session for parallel execution"""

        async with AsyncCamoufox(
            headless=self.headless,
            proxy=self.proxy,
            geoip=True,
            humanize=True,
            block_images=True,
            i_know_what_im_doing=True,  # Suppress WAF detection warnings for image blocking
            locale=["en-US", "en"],
        ) as browser:
            page = await browser.new_page()

            try:
                return await self._search_bing(page, query, max_results)
            finally:
                await page.close()

    async def _search_duckduckgo_parallel(self, query: str, max_results: int) -> List[str]:
        """Search DuckDuckGo with dedicated browser session for parallel execution"""

        async with AsyncCamoufox(
            headless=self.headless,
            proxy=self.proxy,
            geoip=True,
            humanize=True,
            block_images=True,
            i_know_what_im_doing=True,  # Suppress WAF detection warnings for image blocking
            locale=["en-US", "en"],
        ) as browser:
            page = await browser.new_page()

            try:
                return await self._search_duckduckgo(page, query, max_results)
            finally:
                await page.close()

    async def _search_multiple_engines(self, page, query: str, max_results: int) -> List[str]:
        """Search using multiple engines sequentially to avoid browser overload"""

        console.print(f"[cyan]🔍 Multi-engine search for: '{query}'[/cyan]")

        # Define engine search tasks - use sequential execution to avoid browser overload
        search_engines = [
            ("Google", self._search_google),
            ("Google Scholar", self._search_google_scholar),
            ("Bing", self._search_bing),
        ]

        # Calculate results per engine
        results_per_engine = max(max_results // len(search_engines), 2)
        console.print(f"[yellow]Target: {results_per_engine} results per engine[/yellow]")

        all_urls = []

        # Execute searches sequentially to prevent browser session overload
        for engine_name, search_func in search_engines:
            try:
                console.print(f"[blue]🔍 Starting {engine_name} search...[/blue]")

                # Add timeout to prevent hanging
                urls = await asyncio.wait_for(
                    search_func(page, query, results_per_engine),
                    timeout=45.0,  # 45 second timeout per engine
                )

                console.print(f"[green]✅ {engine_name}: {len(urls)} results[/green]")
                all_urls.extend(urls)

                # Add delay between engines to prevent rate limiting
                await asyncio.sleep(random.uniform(1, 3))

            except asyncio.TimeoutError:
                console.print(f"[yellow]⏰ {engine_name} search timed out after 45s[/yellow]")
                continue
            except Exception as e:
                console.print(f"[red]❌ {engine_name} search failed: {e}[/red]")
                continue

        # Remove duplicates and return combined results
        unique_urls = []
        seen_urls = set()

        for url in all_urls:
            if url not in seen_urls:
                unique_urls.append(url)
                seen_urls.add(url)

        console.print(f"[green]🎯 Search completed: {len(unique_urls)} unique URLs found[/green]")

        # Apply smart filtering using granite LLM
        if len(unique_urls) > max_results:
            console.print(f"[cyan]🧠 Applying smart URL filtering with granite3.3:8b...[/cyan]")
            filtered_urls = await self._smart_url_filter(unique_urls, query)
            return filtered_urls[:max_results]

        return unique_urls[:max_results]

    async def _process_sources(
        self, page, urls: List[str], quality_criteria: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Process and analyze discovered sources with controlled batching to prevent memory issues"""

        console.print(f"[cyan]📄 Processing {len(urls)} sources in controlled batches...[/cyan]")

        if not urls:
            return []

        processed_sources = []

        # Process in larger batches for better performance
        batch_size = 5  # Increased from 2 to process faster with fewer batch overhead

        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(urls) + batch_size - 1) // batch_size

            console.print(
                f"[yellow]📄 Processing batch {batch_num}/{total_batches} ({len(batch_urls)} sources)...[/yellow]"
            )

            # Process each URL in the batch sequentially to avoid memory issues
            for url in batch_urls:
                try:
                    # Use existing page instead of creating new browser sessions
                    source_data = await asyncio.wait_for(
                        self._extract_source_content(page, url),
                        timeout=25.0,  # Reduced timeout to 25 seconds per source
                    )

                    if source_data and self._meets_quality_criteria(source_data, quality_criteria):
                        enhanced_source = await self._enhance_source_analysis(source_data)
                        processed_sources.append(enhanced_source)
                        console.print(
                            f"[green]✅ Processed: {enhanced_source.get('title', url)[:60]}...[/green]"
                        )
                    else:
                        console.print(f"[yellow]⚠️ Quality check failed for: {url}[/yellow]")

                except asyncio.TimeoutError:
                    console.print(f"[red]❌ Timeout after 25s: {url}[/red]")
                    console.print(f"[yellow]⚠️ Skipping this source and continuing...[/yellow]")
                    continue

                except Exception as e:
                    console.print(f"[red]❌ Source processing failed: {url} - {e}[/red]")
                    continue

                # Small delay between sources to prevent overwhelming
                await asyncio.sleep(0.5)  # Reduced from 1 second to speed up processing

            # Delay between batches to allow memory cleanup
            if i + batch_size < len(urls):
                console.print(
                    f"[cyan]⏳ Batch {batch_num} complete, preparing next batch...[/cyan]"
                )
                await asyncio.sleep(1)  # Reduced from 3 seconds to speed up processing

        console.print(f"[green]✅ Successfully processed {len(processed_sources)} sources[/green]")
        return processed_sources

    async def _extract_source_content(self, page, url: str) -> Optional[Dict[str, Any]]:
        """Extract comprehensive content from a source"""

        try:
            # Try to navigate with extended timeout and better SSL handling
            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
            except Exception as nav_error:
                # Try with different wait conditions if networkidle fails
                if "SEC_ERROR_UNKNOWN_ISSUER" in str(nav_error) or "SSL" in str(nav_error):
                    console.print(f"[yellow]⚠️ SSL issue for {url}, trying without SSL verification[/yellow]")
                    # Skip this source for SSL issues rather than failing
                    return None
                elif "timeout" in str(nav_error).lower():
                    console.print(f"[yellow]⚠️ Timeout loading {url}, trying domcontentloaded[/yellow]")
                    await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                else:
                    raise nav_error

            # Extract basic information
            title = await page.title()
            content_html = await page.content()

            # Parse with BeautifulSoup
            soup = BeautifulSoup(content_html, "html.parser")

            # Remove unwanted elements
            for element in soup(
                [
                    "script",
                    "style",
                    "nav",
                    "header",
                    "footer",
                    "aside",
                    "advertisement",
                    ".ad",
                    "#ad",
                ]
            ):
                element.decompose()

            # Extract structured content
            content_data = {
                "url": url,
                "title": title or "No Title",
                "domain": urlparse(url).netloc,
                "extracted_at": datetime.now().isoformat(),
            }

            # Extract main text content
            text_content = soup.get_text(separator="\n", strip=True)
            lines = [line.strip() for line in text_content.split("\n") if line.strip()]
            clean_content = "\n".join(lines)

            content_data["content"] = clean_content[:8000]  # Increased from default limit
            content_data["word_count"] = len(clean_content.split())
            content_data["char_count"] = len(clean_content)
            content_data["full_content_available"] = len(clean_content) > 8000

            # Extract metadata
            content_data["metadata"] = self._extract_metadata(soup)

            # Extract links and citations
            content_data["links"] = self._extract_links(soup, url)
            content_data["citations"] = self._extract_citations(soup)

            # Calculate quality scores
            content_data["relevance_score"] = self._calculate_relevance_score(content_data)
            content_data["credibility_score"] = self._calculate_credibility_score(content_data)
            content_data["source_type"] = detect_source_type(url)

            # Quick content assessment using granite3.3:8b
            try:
                quick_assessment = await self._quick_content_assessment(
                    title or "No Title",
                    clean_content[:1000],  # First 1000 chars for speed
                    content_data["domain"],
                )
                content_data["granite_assessment"] = quick_assessment
                console.print(
                    f"[cyan]🧠 Granite assessment: {quick_assessment.get('quality', 'unknown')} quality, {quick_assessment.get('content_type', 'unknown')} type[/cyan]"
                )
            except Exception as e:
                console.print(f"[yellow]Warning: Quick assessment failed: {e}[/yellow]")

            return content_data

        except Exception as e:
            console.print(f"[red]❌ Content extraction failed for {url}: {e}[/red]")
            return None

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from HTML"""

        metadata = {}

        # Extract meta tags
        for meta in soup.find_all("meta"):
            attrs = getattr(meta, "attrs", {})
            name = attrs.get("name") or attrs.get("property", "").replace("og:", "")
            content = attrs.get("content")

            if name and content:
                metadata[name] = content

        # Extract publication date
        date_selectors = ["[datetime]", ".date", ".published", ".post-date", "time", ".timestamp"]

        for selector in date_selectors:
            for date_elem in soup.select(selector):
                date_text = date_elem.get("datetime") or date_elem.get_text()
                if date_text and isinstance(date_text, str):
                    metadata["publication_date"] = date_text.strip()
                    break  # Extract author information
        author_selectors = [
            ".author",
            ".byline",
            '[rel="author"]',
            ".writer",
            ".post-author",
            ".article-author",
        ]

        for selector in author_selectors:
            author_elem = soup.select_one(selector)
            if author_elem:
                metadata["author"] = author_elem.get_text().strip()
                break

        return metadata

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract and categorize links from content"""

        links = []

        for link in soup.find_all("a", href=True):
            try:
                attrs = getattr(link, "attrs", {})
                href = attrs.get("href", "")
                text = link.get_text().strip()

                if href and text:
                    # Convert relative URLs to absolute
                    full_url = urljoin(base_url, href)

                    link_data = {
                        "url": full_url,
                        "text": text,
                        "type": self._classify_link(full_url, text),
                    }

                    links.append(link_data)

            except (AttributeError, TypeError):
                continue

        return links[:20]  # Limit to prevent memory issues

    def _extract_citations(self, soup: BeautifulSoup) -> List[str]:
        """Extract citations and references from content"""

        citations = []

        # Look for common citation patterns
        citation_selectors = [
            ".citation",
            ".reference",
            ".footnote",
            "[cite]",
            ".bibliography",
            ".refs",
        ]

        for selector in citation_selectors:
            for elem in soup.select(selector):
                citation_text = elem.get_text().strip()
                if citation_text and len(citation_text) > 10:
                    citations.append(citation_text)

        # Extract numbered references
        text_content = soup.get_text()
        reference_pattern = r"\[(\d+)\]|\((\d+)\)"
        references = re.findall(reference_pattern, text_content)

        if references:
            citations.extend([f"Reference {r[0] or r[1]}" for r in references[:10]])

        return citations[:15]  # Limit citations

    def _calculate_relevance_score(self, content_data: Dict[str, Any]) -> float:
        """Calculate relevance score based on content analysis"""

        score = 0.5  # Base score

        # Content length bonus
        word_count = content_data.get("word_count", 0)
        if word_count > 300:
            score += 0.1
        if word_count > 1000:
            score += 0.1
        if word_count > 2000:
            score += 0.1

        # Title quality bonus
        title = content_data.get("title", "").lower()
        if any(keyword in title for keyword in ["research", "analysis", "study", "report"]):
            score += 0.1

        # Citations bonus
        citations = content_data.get("citations", [])
        if len(citations) > 0:
            score += 0.1
        if len(citations) > 5:
            score += 0.1

        return min(score, 1.0)

    def _calculate_credibility_score(self, content_data: Dict[str, Any]) -> float:
        """Calculate credibility score based on various factors"""

        score = 0.5  # Base score

        domain = content_data.get("domain", "").lower()

        # Domain credibility
        if any(domain.endswith(tld) for tld in [".edu", ".gov"]):
            score += 0.3
        elif any(domain.endswith(tld) for tld in [".org"]):
            score += 0.2
        elif any(trusted in domain for trusted in ["reuters", "bbc", "ap.org", "nature.com"]):
            score += 0.25

        # Author presence
        metadata = content_data.get("metadata", {})
        if metadata.get("author"):
            score += 0.1

        # Publication date (recent content)
        if metadata.get("publication_date"):
            score += 0.1

        # Citations and references
        citations = content_data.get("citations", [])
        if len(citations) > 0:
            score += 0.1

        # Content quality indicators
        content = content_data.get("content", "").lower()
        quality_keywords = ["research", "study", "analysis", "data", "methodology"]
        if any(keyword in content for keyword in quality_keywords):
            score += 0.1

        return min(score, 1.0)

    def _classify_link(self, url: str, text: str) -> str:
        """Classify link type for better organization"""

        url_lower = url.lower()
        text_lower = text.lower()

        if any(ref in text_lower for ref in ["reference", "citation", "source"]):
            return "reference"
        elif any(doc in url_lower for doc in [".pdf", "doc", "documentation"]):
            return "document"
        elif any(news in url_lower for news in ["news", "article", "blog"]):
            return "article"
        elif any(academic in url_lower for academic in [".edu", "academic", "research"]):
            return "academic"
        else:
            return "general"

    def _meets_quality_criteria(
        self, source_data: Dict[str, Any], criteria: Dict[str, Any]
    ) -> bool:
        """Check if source meets quality criteria"""

        content_criteria = criteria.get("content_quality", {})

        # Check minimum word count (reduced for better source acceptance)
        min_words = content_criteria.get("min_word_count", 100)  # Reduced from 200 to 100
        if source_data.get("word_count", 0) < min_words:
            return False

        # Check credibility threshold (reduced for better source acceptance)
        min_credibility = 0.3  # Reduced from 0.4 to 0.3
        if source_data.get("credibility_score", 0) < min_credibility:
            return False

        # Check exclusion criteria
        exclusions = criteria.get("exclusion_criteria", [])
        url = source_data.get("url", "").lower()

        for exclusion in exclusions:
            if (
                exclusion == "paywall_protected"
                and "subscribe" in source_data.get("content", "").lower()
            ):
                return False
            elif exclusion == "broken_links" and not source_data.get("content"):
                return False

        return True

    async def _enhance_source_analysis(self, source_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance source analysis with LLM insights"""

        try:
            # Prepare content summary for LLM analysis
            content = source_data.get("content", "")[:2000]  # Limit for LLM

            analysis_prompt = f"""
            Analyze this research source and provide insights:
            
            Title: {source_data.get('title', 'Unknown')}
            Domain: {source_data.get('domain', 'Unknown')}
            Content Preview: {content[:500]}...
            
            Please provide:
            1. Key topics covered (3-5 topics)
            2. Content quality assessment (1-5 scale)
            3. Potential bias indicators
            4. Relevance to research queries
            5. Credibility indicators
            
            Respond in JSON format with these fields:
            - topics: []
            - quality_score: number
            - bias_indicators: []
            - relevance_notes: string
            - credibility_notes: string
            """

            # Use browser LLM (granite3.3:8b) for fast content analysis
            response = self.browser_llm.invoke(analysis_prompt)
            llm_analysis_text = response.content if hasattr(response, "content") else str(response)

            # Ensure we have a string for JSON parsing
            if isinstance(llm_analysis_text, list):
                llm_analysis_text = str(llm_analysis_text[0]) if llm_analysis_text else ""
            elif not isinstance(llm_analysis_text, str):
                llm_analysis_text = str(llm_analysis_text)

            # Try to parse JSON response
            try:
                llm_analysis = json.loads(llm_analysis_text)
                source_data["llm_analysis"] = llm_analysis
            except json.JSONDecodeError:
                source_data["llm_analysis"] = {"raw_response": llm_analysis_text}

        except Exception as e:
            console.print(f"[yellow]⚠️ LLM analysis failed: {e}[/yellow]")
            source_data["llm_analysis"] = {"error": str(e)}

        return source_data

    async def _quick_content_assessment(
        self, title: str, content: str, domain: str
    ) -> Dict[str, Any]:
        """Quick content assessment using granite3.3:8b for fast browser processing"""

        try:
            # Limit content for fast processing
            content_preview = content[:800] if content else ""

            assessment_prompt = f"""
            Quickly assess this web content:
            
            Domain: {domain}
            Title: {title}
            Content: {content_preview}
            
            Provide a brief assessment in JSON format:
            {{
                "relevance_score": 0.0-1.0,
                "quality": "high/medium/low",
                "content_type": "academic/news/blog/commercial/other",
                "key_topics": ["topic1", "topic2"],
                "trustworthy": true/false
            }}
            """

            response = await self.browser_llm.ainvoke(assessment_prompt)
            assessment_text = response.content if hasattr(response, "content") else str(response)

            # Ensure we have a string
            if isinstance(assessment_text, list):
                assessment_text = str(assessment_text[0]) if assessment_text else ""
            elif not isinstance(assessment_text, str):
                assessment_text = str(assessment_text)

            try:
                assessment = json.loads(assessment_text)
                return assessment
            except json.JSONDecodeError:
                # Fallback assessment
                return {
                    "relevance_score": 0.5,
                    "quality": "medium",
                    "content_type": "other",
                    "key_topics": [],
                    "trustworthy": True,
                    "raw_response": assessment_text,
                }

        except Exception as e:
            console.print(f"[yellow]Warning: Quick assessment failed: {e}[/yellow]")
            return {
                "relevance_score": 0.5,
                "quality": "medium",
                "content_type": "other",
                "key_topics": [],
                "trustworthy": True,
            }

    async def _smart_url_filter(self, urls: List[str], query: str) -> List[str]:
        """Use granite3.3:8b to intelligently filter and rank URLs based on query relevance"""

        if len(urls) <= 15:
            return urls  # No need to filter if we have few URLs

        try:
            # Group URLs for batch processing
            url_info = []
            for url in urls[:30]:  # Limit for processing speed
                domain = url.split("/")[2] if len(url.split("/")) > 2 else url
                path = "/".join(url.split("/")[3:])[:50] if len(url.split("/")) > 3 else ""
                url_info.append(f"Domain: {domain}, Path: {path}")

            filter_prompt = f"""
            Research Query: {query}
            
            URLs to evaluate:
            {chr(10).join([f"{i+1}. {info}" for i, info in enumerate(url_info)])}
            
            Return the numbers (1-{len(url_info)}) of the most relevant URLs, separated by commas.
            Focus on: academic sources, reputable domains, relevant content paths.
            Example: 1,3,7,12,15
            """

            response = await self.browser_llm.ainvoke(filter_prompt)
            filter_text = response.content if hasattr(response, "content") else str(response)

            # Ensure we have a string
            if isinstance(filter_text, list):
                filter_text = str(filter_text[0]) if filter_text else ""
            elif not isinstance(filter_text, str):
                filter_text = str(filter_text)

            # Parse the response to get selected URL indices
            selected_indices = []
            for num in filter_text.replace(" ", "").split(","):
                try:
                    idx = int(num) - 1  # Convert to 0-based index
                    if 0 <= idx < len(urls):
                        selected_indices.append(idx)
                except ValueError:
                    continue

            # Return filtered URLs, or original if filtering failed
            if selected_indices:
                filtered_urls = [urls[i] for i in selected_indices[:15]]  # Limit to top 15
                console.print(
                    f"[cyan]🎯 Smart filter: {len(urls)} → {len(filtered_urls)} URLs[/cyan]"
                )
                return filtered_urls
            else:
                return urls[:15]  # Fallback to first 15

        except Exception as e:
            console.print(f"[yellow]Warning: Smart URL filtering failed: {e}[/yellow]")
            return urls[:15]  # Fallback

    def _extract_key_claims(self, sources: List[Dict[str, Any]]) -> List[str]:
        """Extract key claims for fact-checking"""

        claims = []

        for source in sources[:5]:  # Limit to first 5 sources
            content = source.get("content", "")

            # Look for statements with statistical claims
            stat_pattern = (
                r"(\d+(?:\.\d+)?%|\d+(?:,\d+)*(?:\.\d+)?\s*(?:million|billion|thousand|percent))"
            )
            stats = re.findall(stat_pattern, content)

            for stat in stats[:3]:  # Limit claims per source
                # Find sentence containing the statistic
                sentences = content.split(".")
                for sentence in sentences:
                    if stat in sentence and len(sentence.strip()) > 20:
                        claims.append(sentence.strip())
                        break

        return claims[:10]  # Limit total claims

    async def _post_process_results(
        self, research_results: Dict[str, Any], research_plan: Dict[str, Any]
    ) -> None:
        """Post-process research results for quality and completeness"""

        sources = research_results["sources"]

        if not sources:
            research_results["quality_metrics"] = {
                "overall_score": 0.0,
                "source_count": 0,
                "avg_credibility": 0.0,
                "avg_relevance": 0.0,
                "completeness": 0.0,
            }
            return

        # Calculate quality metrics
        credibility_scores = [s.get("credibility_score", 0.5) for s in sources]
        relevance_scores = [s.get("relevance_score", 0.5) for s in sources]

        avg_credibility = sum(credibility_scores) / len(credibility_scores)
        avg_relevance = sum(relevance_scores) / len(relevance_scores)

        # Source diversity analysis
        source_types = list(set(s.get("source_type", "unknown") for s in sources))
        diversity_score = min(len(source_types) / 5.0, 1.0)

        # Content completeness
        total_words = sum(s.get("word_count", 0) for s in sources)
        completeness = min(total_words / 10000.0, 1.0)

        # Overall quality score
        overall_score = (
            avg_credibility * 0.3
            + avg_relevance * 0.25
            + diversity_score * 0.2
            + completeness * 0.15
            + min(len(sources) / research_plan["max_sources"], 1.0) * 0.1
        )

        research_results["quality_metrics"] = {
            "overall_score": round(overall_score, 3),
            "source_count": len(sources),
            "avg_credibility": round(avg_credibility, 3),
            "avg_relevance": round(avg_relevance, 3),
            "source_diversity": round(diversity_score, 3),
            "completeness": round(completeness, 3),
            "total_words": total_words,
            "source_types": source_types,
        }

        console.print(f"[cyan]📊 Research Quality Score: {overall_score:.2f}[/cyan]")
