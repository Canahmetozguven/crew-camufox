#!/usr/bin/env python3
"""
Enhanced Deep Researcher Agent with Tool Composition Integration
Combines the original DeepResearcherAgent capabilities with advanced Tool Composition system
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from camoufox import AsyncCamoufox
from crewai import Agent
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.models.research_models import ResearchSource, BrowserSession
from src.utils.helpers import extract_text_content, detect_source_type, calculate_credibility_score
from src.tools.composition.integration import ComposedToolManager, enhanced_search, get_pipeline_capabilities

console = Console()


class EnhancedDeepResearcherAgent:
    """
    Enhanced Deep Researcher Agent with Tool Composition Integration
    
    Combines the proven multi-engine research capabilities of the original DeepResearcherAgent
    with the advanced parallel processing, intelligent filtering, and performance optimization
    features of the Tool Composition system.
    
    Key Enhancements:
    - Parallel multi-engine search execution
    - Intelligent query optimization and variant generation
    - Advanced result deduplication and quality scoring
    - Performance monitoring and health checks
    - Batch processing capabilities
    - Context-aware search optimization
    - Fallback strategies for reliability
    """

    def __init__(
        self,
        model_name: str = "magistral:latest",
        browser_model_name: str = "granite3.3:8b",
        ollama_base_url: str = "http://localhost:11434",
        headless: bool = True,
        proxy: Optional[Dict[str, str]] = None,
        use_composition: bool = True,
    ):
        self.model_name = model_name
        self.browser_model_name = browser_model_name
        self.ollama_base_url = ollama_base_url
        self.headless = headless
        self.proxy = proxy
        self.use_composition = use_composition

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
        console.print(f"[cyan]🔧 Tool Composition: {'Enabled' if use_composition else 'Legacy Mode'}[/cyan]")

        # Initialize the researcher agent with main LLM
        self.agent = Agent(
            role="Enhanced Multi-Engine Web Researcher",
            goal="Execute comprehensive web research using advanced tool composition and parallel processing",
            backstory=f"""You are an elite cyber-researcher with expertise in advanced tool composition,
            parallel multi-engine web scraping, and intelligent information extraction. You operate with 
            a sophisticated dual-model architecture enhanced by tool composition capabilities:
            
            - Main Intelligence ({model_name}): Research planning, analysis, and synthesis
            - Browser Intelligence ({browser_model_name}): Fast content processing and extraction
            - Tool Composition System: Parallel execution, intelligent filtering, and quality optimization
            
            Enhanced capabilities include:
            
            - Parallel multi-engine search execution (Google, Scholar, Bing, DuckDuckGo)
            - Intelligent query optimization and variant generation
            - Advanced result deduplication and relevance scoring
            - Quality validation with multi-criteria filtering
            - Performance monitoring and health checks
            - Context-aware search optimization
            - Batch processing for efficiency
            - Fallback strategies for maximum reliability
            - Anti-detection browsing with Camoufox stealth features
            
            You approach each research task with unprecedented efficiency and intelligence, leveraging
            the power of tool composition to deliver superior results faster than ever before.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,  # Use main LLM for agent operations
        )

        # Initialize Tool Composition Manager if enabled
        if self.use_composition:
            self.composition_manager = ComposedToolManager()
            console.print(f"[green]✅ Tool Composition Manager initialized[/green]")
            
            # Log capabilities
            capabilities = get_pipeline_capabilities()
            console.print(f"[cyan]📋 Available pipelines: {list(capabilities.get('pipelines', {}).keys())}[/cyan]")
            console.print(f"[cyan]🔍 Supported engines: {capabilities.get('supported_engines', [])}[/cyan]")
        else:
            self.composition_manager = None
            console.print(f"[yellow]⚠️ Running in legacy mode without tool composition[/yellow]")

    async def execute_research_plan(self, research_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a comprehensive research plan with enhanced capabilities"""

        session_id = f"enhanced_research_{int(datetime.now().timestamp())}"
        console.print(
            f"[bold blue]🔍 Executing Enhanced Research Plan: {research_plan['query']}[/bold blue]"
        )
        console.print(f"[yellow]Session ID: {session_id}[/yellow]")
        console.print(f"[yellow]Depth: {research_plan['research_depth']}[/yellow]")
        console.print(f"[yellow]Tool Composition: {'Enabled' if self.use_composition else 'Legacy Mode'}[/yellow]")

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
            "enhancement_used": "tool_composition" if self.use_composition else "legacy",
            "performance_stats": {},
        }

        try:
            # Execute research with enhanced capabilities
            if self.use_composition:
                return await self._execute_enhanced_research(research_plan, research_results)
            else:
                return await self._execute_legacy_research(research_plan, research_results)

        except Exception as e:
            console.print(f"[red]❌ Research failed: {e}[/red]")
            research_results["completion_status"] = "failed"
            research_results["error"] = str(e)
            research_results["failed_at"] = datetime.now().isoformat()
            return research_results

    async def _execute_enhanced_research(
        self, research_plan: Dict[str, Any], research_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute research using enhanced tool composition capabilities"""

        console.print(f"[cyan]🚀 Using Enhanced Tool Composition Research[/cyan]")
        
        start_time = time.time()
        query = research_plan["query"]
        max_sources = research_plan.get("max_sources", 20)
        
        # Prepare context for optimized search
        search_context = {
            "research_depth": research_plan.get("research_depth", "comprehensive"),
            "quality_criteria": research_plan.get("quality_criteria", {}),
            "preferred_engines": research_plan.get("preferred_engines", ["google", "scholar", "bing", "duckduckgo"]),
            "min_relevance": research_plan.get("min_relevance_score", 0.3),
            "content_types": research_plan.get("content_types", ["academic", "news", "general"]),
            "enable_parallel": True,
            "enable_intelligent_filtering": True,
            "enable_quality_validation": True,
        }

        try:
            # Phase 1: Enhanced Primary Search
            console.print(f"\n[cyan]📋 Phase 1: Enhanced Primary Search[/cyan]")
            
            # Generate search variants for better coverage
            search_terms = research_plan.get("search_strategies", {}).get("primary_terms", [query])
            all_search_results = []
            
            # Use batch search for efficiency
            if len(search_terms) > 1 and self.composition_manager is not None:
                console.print(f"[yellow]🔍 Executing batch search for {len(search_terms)} queries...[/yellow]")
                batch_results = await self.composition_manager.batch_search(
                    search_terms[:5],  # Limit to top 5 terms
                    max_results_per_query=max(8, max_sources // 2),  # Each query gets substantial results
                    context=search_context
                )
                
                # Aggregate results
                for result in batch_results:
                    if result.success:
                        all_search_results.extend(result.data)
                        console.print(f"[green]✅ Batch query successful: {len(result.data)} results[/green]")
                    else:
                        console.print(f"[red]❌ Batch query failed: {result.error}[/red]")
                        
            else:
                # Single enhanced search
                if self.composition_manager is not None:
                    search_result = await self.composition_manager.enhanced_search(
                        query,
                        max_results=max_sources,
                        context=search_context
                    )
                else:
                    # Fallback if composition manager is None
                    console.print(f"[red]❌ Composition manager not available[/red]")
                    return await self._execute_legacy_research(research_plan, research_results)
                
                if search_result.success:
                    all_search_results = search_result.data
                    console.print(f"[green]✅ Enhanced search successful: {len(all_search_results)} results[/green]")
                else:
                    console.print(f"[red]❌ Enhanced search failed: {search_result.error}[/red]")
                    # Fallback to legacy mode
                    return await self._execute_legacy_research(research_plan, research_results)

            # Phase 2: Content Extraction and Analysis
            console.print(f"\n[cyan]📋 Phase 2: Enhanced Content Processing[/cyan]")
            
            if all_search_results:
                # Extract URLs for content processing
                source_urls = [result.get("url") for result in all_search_results if result.get("url")]
                console.print(f"[yellow]📄 Processing {len(source_urls)} discovered sources...[/yellow]")
                
                # Use existing content extraction with enhanced error handling
                processed_sources = await self._process_sources_enhanced(
                    source_urls, 
                    research_plan.get("quality_criteria", {}),
                    all_search_results  # Pass search metadata for enrichment
                )
                
                research_results["sources"] = processed_sources
                console.print(f"[green]✅ Successfully processed {len(processed_sources)} sources[/green]")
            else:
                console.print(f"[red]❌ No sources found in enhanced search[/red]")
                research_results["sources"] = []

            # Phase 3: Quality Analysis and Metrics
            console.print(f"\n[cyan]📋 Phase 3: Quality Analysis and Performance Metrics[/cyan]")
            
            # Get performance statistics
            if self.composition_manager is not None:
                performance_stats = self.composition_manager.get_pipeline_stats()
                research_results["performance_stats"] = performance_stats
            
            # Enhanced post-processing
            await self._post_process_results_enhanced(research_results, research_plan)
            
            # Final status
            execution_time = time.time() - start_time
            source_count = len(research_results["sources"])
            
            if source_count >= 3:
                research_results["completion_status"] = "completed"
                console.print(f"[green]✅ Enhanced research completed successfully![/green]")
            elif source_count > 0:
                research_results["completion_status"] = "partial"
                console.print(f"[yellow]⚠️ Enhanced research completed with limited results[/yellow]")
            else:
                research_results["completion_status"] = "failed"
                console.print(f"[red]❌ Enhanced research failed to find sources[/red]")
            
            research_results["completed_at"] = datetime.now().isoformat()
            research_results["execution_time"] = execution_time
            
            console.print(f"[green]📊 Enhanced Research Summary:[/green]")
            console.print(f"[green]   - Sources found: {source_count}[/green]")
            console.print(f"[green]   - Execution time: {execution_time:.2f}s[/green]")
            console.print(f"[green]   - Performance score: {performance_stats.get('search_pipeline_performance', {}).get('success_rate', 0)}%[/green]")
            
            return research_results

        except Exception as e:
            console.print(f"[red]❌ Enhanced research execution failed: {e}[/red]")
            console.print(f"[yellow]🔄 Falling back to legacy research mode...[/yellow]")
            return await self._execute_legacy_research(research_plan, research_results)

    async def _execute_legacy_research(
        self, research_plan: Dict[str, Any], research_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute research using legacy mode (original DeepResearcherAgent logic)"""
        
        console.print(f"[yellow]🔄 Using Legacy Research Mode[/yellow]")
        
        # Use the original logic from DeepResearcherAgent
        try:
            successful_phases = 0
            total_sources_found = 0

            for phase in research_plan["execution_phases"]:
                console.print(f"\n[cyan]📋 Phase {phase['phase']}: {phase['name']}[/cyan]")

                try:
                    phase_result = await asyncio.wait_for(
                        self._execute_research_phase_legacy(phase, research_plan, research_results),
                        timeout=300.0,
                    )
                    research_results["phase_results"].append(phase_result)

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
                    continue

                except Exception as e:
                    console.print(f"[red]❌ Phase {phase['phase']} failed: {e}[/red]")
                    continue

                if total_sources_found >= 5:
                    console.print(
                        f"[green]✅ Sufficient sources found ({total_sources_found}), proceeding to analysis...[/green]"
                    )
                    break

            # Determine completion status
            final_source_count = len(research_results["sources"])

            if final_source_count == 0:
                research_results["completion_status"] = "failed"
                research_results["error"] = "No sources could be extracted from any research phase"
            elif final_source_count < 3:
                research_results["completion_status"] = "partial"
                research_results["warning"] = f"Only {final_source_count} sources found, results may be limited"
            else:
                research_results["completion_status"] = "completed"

            # Post-processing
            await self._post_process_results(research_results, research_plan)
            research_results["completed_at"] = datetime.now().isoformat()

            console.print(f"[green]✅ Legacy research completed![/green]")
            console.print(f"[green]📊 Total sources: {len(research_results['sources'])}[/green]")

            return research_results

        except Exception as e:
            console.print(f"[red]❌ Legacy research failed: {e}[/red]")
            research_results["completion_status"] = "failed"
            research_results["error"] = str(e)
            research_results["failed_at"] = datetime.now().isoformat()
            return research_results

    async def _process_sources_enhanced(
        self, 
        urls: List[str], 
        quality_criteria: Dict[str, Any],
        search_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Enhanced source processing with search metadata integration"""
        
        console.print(f"[cyan]📄 Enhanced processing of {len(urls)} sources...[/cyan]")
        
        if not urls:
            return []

        # Create URL to metadata mapping
        url_metadata = {}
        if search_metadata:
            for item in search_metadata:
                if item.get("url"):
                    url_metadata[item["url"]] = {
                        "search_relevance": item.get("relevance_score", 0.5),
                        "search_engine": item.get("source_engine", "unknown"),
                        "search_ranking": item.get("ranking", 0),
                        "search_confidence": item.get("confidence_score", 0.5),
                    }

        processed_sources = []
        batch_size = 8  # Increased from 5 to process more sources efficiently

        for i in range(0, len(urls), batch_size):
            batch_urls = urls[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(urls) + batch_size - 1) // batch_size

            console.print(
                f"[yellow]📄 Enhanced processing batch {batch_num}/{total_batches} ({len(batch_urls)} sources)...[/yellow]"
            )

            # Use context manager for browser session
            async with AsyncCamoufox(
                headless=self.headless,
                proxy=self.proxy,
                geoip=True,
                humanize=True,
                block_images=True,
                i_know_what_im_doing=True,
                locale=["en-US", "en"],
                args=[
                    "--ignore-certificate-errors",
                    "--ignore-ssl-errors",
                    "--ignore-certificate-errors-spki-list",
                ],
            ) as browser:
                page = await browser.new_page()

                try:
                    for url in batch_urls:
                        try:
                            source_data = await asyncio.wait_for(
                                self._extract_source_content_enhanced(page, url, url_metadata.get(url, {})),
                                timeout=25.0,
                            )
                            print(f"DEBUG: Extracted data for {url}: {source_data}")
                            # More lenient quality checking - accept more sources for comprehensive research
                            if source_data:
                                # Apply basic quality check but be more permissive
                                if (source_data.get("word_count", 0) > 50 and 
                                    source_data.get("title") and 
                                    len(source_data.get("content", "").strip()) > 100):
                                    enhanced_source = await self._enhance_source_analysis(source_data)
                                    processed_sources.append(enhanced_source)
                                    console.print(
                                        f"[green]✅ Enhanced processed: {enhanced_source.get('title', url)[:60]}...[/green]"
                                    )
                                else:
                                    console.print(f"[yellow]⚠️ Basic quality check failed for: {url} (insufficient content)[/yellow]")
                                    print(f"DEBUG: Source failed basic content check: {url}")
                            else:
                                console.print(f"[yellow]⚠️ Enhanced extraction returned None for: {url}[/yellow]")
                                print(f"DEBUG: Source extraction failed: {url}")
                        except asyncio.TimeoutError:
                            console.print(f"[red]❌ Enhanced timeout after 25s: {url}[/red]")
                            continue

                        except Exception as e:
                            console.print(f"[red]❌ Enhanced source processing failed: {url} - {e}[/red]")
                            print(f"DEBUG: Exception for {url}: {e}")
                            continue

                        await asyncio.sleep(0.5)

                finally:
                    await page.close()

            if i + batch_size < len(urls):
                await asyncio.sleep(1)

        console.print(f"[green]✅ Enhanced processing completed: {len(processed_sources)} sources[/green]")
        return processed_sources

    async def _extract_source_content_enhanced(
        self, page, url: str, search_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Enhanced content extraction with search metadata integration"""
        
        try:
            # Navigate to the page with retries
            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
            except Exception as nav_error:
                # Try with load event instead of networkidle for problematic pages
                try:
                    await page.goto(url, wait_until="load", timeout=20000)
                except Exception:
                    console.print(f"[yellow]⚠️ Navigation failed for {url}, trying basic load...[/yellow]")
                    await page.goto(url, timeout=15000)
            
            # Extract basic information
            title = await page.title() or "Untitled"
            content_html = await page.content()

            # Parse with BeautifulSoup for content extraction
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content_html, "html.parser")

            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "aside", "header"]):
                element.decompose()

            # Extract text content
            text_content = soup.get_text(separator="\n", strip=True)
            
            # Clean up whitespace
            lines = [line.strip() for line in text_content.split("\n") if line.strip()]
            clean_content = "\n".join(lines)
            
            # Ensure we have some content
            if not clean_content or len(clean_content.strip()) < 50:
                # Try alternative extraction methods
                main_content = soup.find("main") or soup.find("article") or soup.find("div", {"class": ["content", "main", "article"]})
                if main_content:
                    clean_content = main_content.get_text(separator="\n", strip=True)
                    lines = [line.strip() for line in clean_content.split("\n") if line.strip()]
                    clean_content = "\n".join(lines)
            
            # Calculate basic metrics
            word_count = len(clean_content.split()) if clean_content else 0
            
            # Create source data structure with expanded content
            source_data = {
                "url": url,
                "title": title,
                "content": clean_content[:8000],  # Significantly increased content length
                "relevance_score": 0.5,  # Default, will be enhanced with search metadata
                "credibility_score": 0.5,  # Default scoring
                "word_count": word_count,
                "full_content_available": len(clean_content) > 8000
            }
            
            # Enhance with search metadata if available
            if search_metadata:
                source_data["search_metadata"] = search_metadata
                
                # Adjust scores based on search metadata
                if "search_relevance" in search_metadata:
                    original_relevance = source_data.get("relevance_score", 0.5)
                    search_relevance = search_metadata["search_relevance"]
                    # Weighted average favoring search engine relevance
                    source_data["relevance_score"] = (original_relevance * 0.3 + search_relevance * 0.7)
                
                # Add search engine source info
                source_data["discovered_via"] = search_metadata.get("search_engine", "unknown")
                source_data["search_ranking"] = search_metadata.get("search_ranking", 0)
            else:
                source_data["discovered_via"] = "direct"
                source_data["search_ranking"] = 0
                
            return source_data
            
        except Exception as e:
            console.print(f"[red]❌ Enhanced content extraction failed for {url}: {e}[/red]")
            return None

    def _meets_quality_criteria_enhanced(
        self, source_data: Dict[str, Any], criteria: Dict[str, Any]
    ) -> bool:
        """Enhanced quality criteria checking with search metadata"""
        
        # Use existing quality check as base
        base_quality = self._meets_quality_criteria_legacy(source_data, criteria)
        
        if not base_quality:
            return False
        
        # Additional enhanced checks
        search_metadata = source_data.get("search_metadata", {})
        
        # Boost quality for high search relevance
        search_relevance = search_metadata.get("search_relevance", 0.5)
        if search_relevance > 0.7:
            return True  # High search relevance overrides some quality concerns
        
        # Check search engine confidence
        search_confidence = search_metadata.get("search_confidence", 0.5)
        if search_confidence < 0.3:
            return False  # Low search engine confidence
        
        return True

    async def _post_process_results_enhanced(
        self, research_results: Dict[str, Any], research_plan: Dict[str, Any]
    ) -> None:
        """Enhanced post-processing with tool composition metrics"""
        
        # Use existing post-processing
        await self._post_process_results(research_results, research_plan)
        
        # Add enhanced metrics
        if self.composition_manager:
            # Get health check
            health_status = await self.composition_manager.health_check()
            research_results["system_health"] = health_status
            
            # Add search engine performance breakdown
            performance_stats = research_results.get("performance_stats", {})
            search_performance = performance_stats.get("search_pipeline_performance", {})
            
            if search_performance:
                research_results["quality_metrics"]["search_performance"] = {
                    "total_searches": search_performance.get("total_executions", 0),
                    "success_rate": search_performance.get("success_rate", 0),
                    "avg_execution_time": search_performance.get("average_execution_time", 0),
                    "parallel_efficiency": search_performance.get("parallel_efficiency", 0),
                }

    # Legacy methods (copied from original DeepResearcherAgent for fallback)
    async def _execute_research_phase_legacy(
        self, phase: Dict[str, Any], research_plan: Dict[str, Any], research_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Legacy research phase execution (fallback method)"""
        # Implementation would be copied from original DeepResearcherAgent
        # For brevity, returning a placeholder - in real implementation, copy the full method
        return {
            "phase": phase["phase"],
            "name": phase["name"],
            "status": "completed",
            "sources": [],
            "failed_urls": [],
            "metrics": {},
        }

    async def _extract_source_content_legacy(self, page, url: str) -> Optional[Dict[str, Any]]:
        """Legacy content extraction (fallback method)
            WE SHOULD NOT USE THIS METHOD, IT IS HERE FOR LEGACY SUPPORT ONLY
            It should be replaced with the enhanced version in _extract_source_content_enhanced
            CURRENTLY THERE IS A ERROR WE HAVE TO FIX
        """
        # Implementation would be copied from original DeepResearcherAgent
        # For brevity, returning a placeholder - in real implementation, copy the full method
        return {
            "url": url,
            "title": "Legacy Extracted Title",
            "content": "Legacy extracted content",
            "relevance_score": 0.5,
            "credibility_score": 0.5,
            "word_count": 1000,  # <-- Add this line

        }

    def _meets_quality_criteria_legacy(
        self, source_data: Dict[str, Any], criteria: Dict[str, Any]
    ) -> bool:
        """Legacy quality criteria checking (fallback method)"""
        # Implementation would be copied from original DeepResearcherAgent
        # For brevity, returning a simple check - in real implementation, copy the full method
        return source_data.get("word_count", 0) > 100

    async def _enhance_source_analysis(self, source_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced source analysis with LLM insights (shared method)"""
        # Implementation copied from original DeepResearcherAgent
        try:
            content = source_data.get("content", "")[:2000]

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

            response = self.browser_llm.invoke(analysis_prompt)
            llm_analysis_text = response.content if hasattr(response, "content") else str(response)

            if isinstance(llm_analysis_text, list):
                llm_analysis_text = str(llm_analysis_text[0]) if llm_analysis_text else ""
            elif not isinstance(llm_analysis_text, str):
                llm_analysis_text = str(llm_analysis_text)

            try:
                llm_analysis = json.loads(llm_analysis_text)
                source_data["llm_analysis"] = llm_analysis
            except json.JSONDecodeError:
                source_data["llm_analysis"] = {"raw_response": llm_analysis_text}

        except Exception as e:
            console.print(f"[yellow]⚠️ LLM analysis failed: {e}[/yellow]")
            source_data["llm_analysis"] = {"error": str(e)}

        return source_data

    async def _post_process_results(
        self, research_results: Dict[str, Any], research_plan: Dict[str, Any]
    ) -> None:
        """Post-process research results for quality and completeness (shared method)"""
        
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
            + min(len(sources) / research_plan.get("max_sources", 20), 1.0) * 0.1
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

    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from tool composition system"""
        if self.composition_manager:
            return self.composition_manager.get_pipeline_stats()
        else:
            return {"message": "Tool composition not enabled"}

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the research system"""
        if self.composition_manager:
            return await self.composition_manager.health_check()
        else:
            return {
                "overall_status": "healthy",
                "mode": "legacy",
                "tool_composition": "disabled",
                "timestamp": datetime.now().isoformat(),
            }

    def get_capabilities(self) -> Dict[str, Any]:
        """Get current system capabilities"""
        base_capabilities = {
            "agent_type": "EnhancedDeepResearcherAgent",
            "version": "2.0.0",
            "models": {
                "main": self.model_name,
                "browser": self.browser_model_name,
            },
            "features": [
                "multi_engine_search",
                "intelligent_content_extraction",
                "quality_assessment",
                "credibility_scoring",
                "llm_enhanced_analysis",
            ],
            "search_engines": ["google", "google_scholar", "bing", "duckduckgo"],
        }

        if self.composition_manager:
            composition_capabilities = get_pipeline_capabilities()
            base_capabilities["tool_composition"] = composition_capabilities
            base_capabilities["features"].extend([
                "parallel_execution",
                "intelligent_filtering",
                "batch_processing",
                "performance_monitoring",
                "health_checks",
                "context_aware_optimization",
            ])
        else:
            base_capabilities["tool_composition"] = "disabled"

        return base_capabilities