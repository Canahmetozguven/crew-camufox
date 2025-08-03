#!/usr/bin/env python3
"""
Enhanced Multi-Agent Research Orchestrator with Tool Composition Integration
Extends the original orchestrator with optional enhanced agent capabilities
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from ..config import get_settings, ResearchConfig, ResearchDepth
from ..utils import get_logger, resilient_operation, CircuitBreakerConfig, RetryConfig
from .research_planner import ResearchPlannerAgent
from .deep_researcher import DeepResearcherAgent
from .enhanced_deep_researcher import EnhancedDeepResearcherAgent
from .final_writer import FinalWriterAgent

# Initialize logger and console
logger = get_logger("enhanced_orchestrator")
console = Console()


class EnhancedMultiAgentResearchOrchestrator:
    """
    Enhanced Multi-Agent Research Orchestrator with Tool Composition Integration
    
    Extends the original orchestrator with:
    - Optional enhanced deep researcher with tool composition
    - Performance comparison capabilities
    - Advanced monitoring and health checks
    - Backward compatibility with original agents
    - Intelligent agent selection based on query complexity
    """

    def __init__(
        self, 
        custom_settings: Optional[Dict[str, Any]] = None,
        use_enhanced_researcher: bool = True,
        enable_performance_comparison: bool = False
    ):
        """Initialize enhanced orchestrator with optional enhanced capabilities"""

        self.settings = get_settings()
        if custom_settings:
            # Override specific settings if provided
            for key, value in custom_settings.items():
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)

        # Configuration options
        self.use_enhanced_researcher = use_enhanced_researcher
        self.enable_performance_comparison = enable_performance_comparison

        # Create output directory
        self.output_dir = getattr(self.settings, 'output_dir', 'research_outputs')
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize agents
        logger.info("Initializing Enhanced Multi-Agent Research System")
        console.print("[bold blue]🚀 Initializing Enhanced Multi-Agent Research System[/bold blue]")

        # Research Planner (unchanged)
        self.planner = ResearchPlannerAgent(
            model_name=self.settings.ollama.model_name,
            ollama_base_url=self.settings.ollama.base_url,
        )

        # Initialize researcher(s) based on configuration
        if self.use_enhanced_researcher:
            console.print("[cyan]🔧 Initializing Enhanced Deep Researcher with Tool Composition...[/cyan]")
            self.researcher = EnhancedDeepResearcherAgent(
                model_name=self.settings.ollama.model_name,
                browser_model_name="granite3.3:8b",  # Fast browser model
                ollama_base_url=self.settings.ollama.base_url,
                headless=self.settings.browser.headless,
                proxy=None,  # TODO: Add proxy support to settings
                use_composition=True
            )
            
            # Also initialize legacy researcher for comparison if enabled
            if self.enable_performance_comparison:
                console.print("[yellow]🔄 Also initializing Legacy Researcher for comparison...[/yellow]")
                self.legacy_researcher = DeepResearcherAgent(
                    model_name=self.settings.ollama.model_name,
                    ollama_base_url=self.settings.ollama.base_url,
                    headless=self.settings.browser.headless,
                    proxy=None,
                )
            else:
                self.legacy_researcher = None
        else:
            console.print("[yellow]🔄 Using Legacy Deep Researcher (enhanced mode disabled)...[/yellow]")
            self.researcher = DeepResearcherAgent(
                model_name=self.settings.ollama.model_name,
                ollama_base_url=self.settings.ollama.base_url,
                headless=self.settings.browser.headless,
                proxy=None,
            )
            self.legacy_researcher = None

        # Final Writer (unchanged)
        self.writer = FinalWriterAgent(
            model_name=self.settings.ollama.model_name,
            ollama_base_url=self.settings.ollama.base_url,
        )

        logger.info("All enhanced agents initialized successfully")
        self._display_initialization_summary()

    def _display_initialization_summary(self) -> None:
        """Display initialization summary with enhanced capabilities"""
        
        # Determine researcher type
        if isinstance(self.researcher, EnhancedDeepResearcherAgent):
            researcher_info = "Enhanced Deep Researcher (Tool Composition Enabled)"
            capabilities = self.researcher.get_capabilities()
            features = capabilities.get("features", [])
        else:
            researcher_info = "Legacy Deep Researcher"
            features = ["multi_engine_search", "content_extraction", "quality_assessment"]

        console.print(f"[green]✅ Research System Initialized:[/green]")
        console.print(f"   • Research Planner: Active")
        console.print(f"   • Deep Researcher: {researcher_info}")
        if self.legacy_researcher:
            console.print(f"   • Legacy Researcher: Available for comparison")
        console.print(f"   • Final Writer: Active")
        console.print(f"   • Performance Comparison: {'Enabled' if self.enable_performance_comparison else 'Disabled'}")
        console.print(f"   • Enhanced Features: {len(features)} available")

    @classmethod
    def create_enhanced(
        cls,
        model_name: str = "magistral:latest",
        ollama_base_url: str = "http://localhost:11434",
        headless: bool = True,
        proxy: Optional[Dict[str, str]] = None,
        output_dir: str = "research_outputs",
        use_enhanced_researcher: bool = True,
        enable_performance_comparison: bool = False,
    ) -> "EnhancedMultiAgentResearchOrchestrator":
        """Create enhanced orchestrator with custom parameters"""

        custom_settings = {
            "ollama": {"model_name": model_name, "base_url": ollama_base_url},
            "browser": {"headless": headless},
            "output_dir": output_dir,
        }

        return cls(
            custom_settings=custom_settings,
            use_enhanced_researcher=use_enhanced_researcher,
            enable_performance_comparison=enable_performance_comparison
        )

    @resilient_operation(
        retry_config=RetryConfig(max_attempts=2, min_wait=5.0),
        circuit_breaker_config=CircuitBreakerConfig(failure_threshold=3, timeout=300),
    )
    async def execute_enhanced_research_mission(
        self,
        query: str,
        research_depth: str = "medium",
        report_type: str = "comprehensive",
        save_outputs: bool = True,
        enable_comparison: Optional[bool] = None,
        max_sources: int = 15,
    ) -> Dict[str, Any]:
        """
        Execute an enhanced research mission with optional performance comparison
        
        Args:
            query: The research question to investigate
            research_depth: "surface", "medium", "deep", or "exhaustive"
            report_type: Type of report to generate
            save_outputs: Whether to save outputs to files
            enable_comparison: Override performance comparison setting for this mission
            
        Returns:
            Enhanced research mission results with optional comparison metrics
        """

        mission_id = f"enhanced_mission_{int(datetime.now().timestamp())}"
        
        # Determine if comparison should be enabled for this mission
        comparison_enabled = enable_comparison if enable_comparison is not None else self.enable_performance_comparison
        comparison_enabled = comparison_enabled and self.legacy_researcher is not None

        console.print(f"\n[bold cyan]🚀 Starting Enhanced Research Mission: {mission_id}[/bold cyan]")
        console.print(f"[yellow]Query: {query}[/yellow]")
        console.print(f"[yellow]Depth: {research_depth}[/yellow]")
        console.print(f"[yellow]Report Type: {report_type}[/yellow]")
        console.print(f"[yellow]Enhanced Mode: {'Yes' if isinstance(self.researcher, EnhancedDeepResearcherAgent) else 'No'}[/yellow]")
        console.print(f"[yellow]Performance Comparison: {'Yes' if comparison_enabled else 'No'}[/yellow]")

        mission_results = {
            "mission_id": mission_id,
            "query": query,
            "research_depth": research_depth,
            "report_type": report_type,
            "started_at": datetime.now().isoformat(),
            "status": "in_progress",
            "enhanced_mode": isinstance(self.researcher, EnhancedDeepResearcherAgent),
            "comparison_enabled": comparison_enabled,
            "stages": {
                "planning": {"status": "pending"},
                "research": {"status": "pending"},
                "writing": {"status": "pending"},
            },
            "outputs": {},
            "performance_metrics": {},
        }

        # Add comparison placeholder if enabled
        if comparison_enabled:
            mission_results["comparison_results"] = {
                "enhanced": {},
                "legacy": {},
                "performance_delta": {}
            }

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:

                # Stage 1: Research Planning (unchanged)
                planning_task = progress.add_task(
                    "[cyan]🧠 Planning enhanced research strategy...", total=None
                )

                console.print("\n[bold blue]Stage 1: Enhanced Research Planning[/bold blue]")
                research_plan = await self._execute_planning_stage(query, research_depth, max_sources)

                mission_results["stages"]["planning"] = {
                    "status": "completed",
                    "completed_at": datetime.now().isoformat(),
                    "plan_id": research_plan.get("id"),
                }
                mission_results["outputs"]["research_plan"] = research_plan

                progress.update(planning_task, description="[green]✅ Enhanced research plan completed")

                # Stage 2: Enhanced Research Execution
                research_task = progress.add_task(
                    "[yellow]🔍 Executing enhanced deep research...", total=None
                )

                console.print("\n[bold blue]Stage 2: Enhanced Deep Research Execution[/bold blue]")
                
                if comparison_enabled:
                    # Execute both enhanced and legacy research for comparison
                    research_results = await self._execute_comparison_research_stage(research_plan, mission_results)
                else:
                    # Execute only the primary researcher
                    research_results = await self._execute_research_stage(research_plan)

                mission_results["stages"]["research"] = {
                    "status": "completed",
                    "completed_at": datetime.now().isoformat(),
                    "sources_found": len(research_results.get("sources", [])),
                    "quality_score": research_results.get("quality_metrics", {}).get("overall_score", 0),
                    "enhancement_used": research_results.get("enhancement_used", "unknown"),
                }
                mission_results["outputs"]["research_results"] = research_results

                progress.update(research_task, description="[green]✅ Enhanced research execution completed")

                # Stage 3: Report Generation (unchanged but with enhanced data)
                writing_task = progress.add_task(
                    "[magenta]📝 Generating enhanced comprehensive report...", total=None
                )

                console.print("\n[bold blue]Stage 3: Enhanced Report Generation[/bold blue]")

                report_config = {
                    "report_type": report_type,
                    "include_appendices": True,
                    "format_options": ["markdown", "text", "json"],
                    "include_performance_metrics": isinstance(self.researcher, EnhancedDeepResearcherAgent),
                    "include_comparison_data": comparison_enabled,
                }

                final_report = await self._execute_writing_stage(
                    research_results, research_plan, report_config
                )

                mission_results["stages"]["writing"] = {
                    "status": "completed",
                    "completed_at": datetime.now().isoformat(),
                    "report_id": final_report.get("report_id"),
                    "quality_score": final_report.get("quality_assessment", {}).get("overall_score", 0),
                }
                mission_results["outputs"]["final_report"] = final_report

                progress.update(writing_task, description="[green]✅ Enhanced report generation completed")

            # Mission completion
            mission_results["status"] = "completed"
            mission_results["completed_at"] = datetime.now().isoformat()

            # Calculate enhanced metrics
            await self._calculate_enhanced_mission_metrics(mission_results)

            # Display enhanced results summary
            self._display_enhanced_mission_summary(mission_results)

            # Save outputs with enhanced metadata
            console.print(f"\n[cyan]💾 Auto-saving enhanced research outputs...[/cyan]")
            await self._save_enhanced_mission_outputs(mission_results, save_outputs)

            console.print(f"\n[bold green]🎉 Enhanced Research Mission Completed Successfully![/bold green]")

            return mission_results

        except Exception as e:
            mission_results["status"] = "failed"
            mission_results["error"] = str(e)
            mission_results["failed_at"] = datetime.now().isoformat()

            console.print(f"\n[bold red]❌ Enhanced Research Mission Failed: {e}[/bold red]")
            return mission_results

    async def _execute_planning_stage(self, query: str, research_depth: str, max_sources: int = 15) -> Dict[str, Any]:
        """Execute the research planning stage (unchanged from original)"""

        console.print(
            "[cyan]🧠 Research Planner Agent: Creating comprehensive research strategy...[/cyan]"
        )

        research_plan = self.planner.create_comprehensive_plan(
            query=query, research_depth=research_depth, max_sources=max_sources
        )

        if research_plan.get("status") == "failed":
            raise Exception(f"Planning failed: {research_plan.get('error', 'Unknown error')}")

        # Display plan summary
        config = research_plan.get("config", {})
        console.print(f"[green]✅ Research plan created:[/green]")
        console.print(f"   • Time Limit: {config.get('max_time_minutes', 'Unknown')} minutes")
        console.print(f"   • Max Sources: {config.get('max_sources', 'Unknown')}")
        console.print(f"   • Execution Phases: {len(research_plan.get('execution_phases', []))}")
        console.print(
            f"   • Search Terms: {len(research_plan.get('search_strategies', {}).get('primary_terms', []))}"
        )

        return research_plan

    async def _execute_research_stage(self, research_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the research stage with the primary researcher"""

        researcher_type = "Enhanced" if isinstance(self.researcher, EnhancedDeepResearcherAgent) else "Legacy"
        console.print(
            f"[yellow]🔍 {researcher_type} Deep Researcher Agent: Executing comprehensive web research...[/yellow]"
        )

        # Execute research plan
        research_results = await self.researcher.execute_research_plan(research_plan)

        if research_results.get("completion_status") == "failed":
            raise Exception(f"Research failed: {research_results.get('error', 'Unknown error')}")

        # Display research summary
        quality_metrics = research_results.get("quality_metrics", {})
        console.print(f"[green]✅ {researcher_type} research execution completed:[/green]")
        console.print(f"   • Sources Found: {len(research_results.get('sources', []))}")
        console.print(f"   • Quality Score: {quality_metrics.get('overall_score', 0):.2f}")
        console.print(f"   • Avg Credibility: {quality_metrics.get('avg_credibility', 0):.2f}")
        console.print(f"   • Total Words: {quality_metrics.get('total_words', 0):,}")

        # Add performance stats if available (enhanced researcher only)
        if isinstance(self.researcher, EnhancedDeepResearcherAgent):
            try:
                performance_stats = await self.researcher.get_performance_stats()
                search_performance = performance_stats.get("search_pipeline_performance", {})
                if search_performance:
                    console.print(f"   • Search Success Rate: {search_performance.get('success_rate', 0)}%")
                    console.print(f"   • Avg Search Time: {search_performance.get('average_execution_time', 0):.2f}s")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not get performance stats: {e}[/yellow]")

        return research_results

    async def _execute_comparison_research_stage(
        self, research_plan: Dict[str, Any], mission_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute research with both enhanced and legacy researchers for comparison"""

        console.print("[cyan]🔬 Executing Comparison Research (Enhanced vs Legacy)...[/cyan]")

        # Execute enhanced research
        console.print("\n[blue]🚀 Enhanced Researcher Execution:[/blue]")
        start_time = datetime.now()
        enhanced_results = await self.researcher.execute_research_plan(research_plan)
        enhanced_duration = (datetime.now() - start_time).total_seconds()

        # Execute legacy research
        console.print("\n[yellow]🔄 Legacy Researcher Execution:[/yellow]")
        start_time = datetime.now()
        
        if self.legacy_researcher is not None:
            legacy_results = await self.legacy_researcher.execute_research_plan(research_plan)
            legacy_duration = (datetime.now() - start_time).total_seconds()
        else:
            # Fallback if legacy researcher is not available
            console.print("[red]❌ Legacy researcher not available for comparison[/red]")
            return enhanced_results

        # Store comparison data
        mission_results["comparison_results"] = {
            "enhanced": {
                "sources_found": len(enhanced_results.get("sources", [])),
                "execution_time": enhanced_duration,
                "quality_score": enhanced_results.get("quality_metrics", {}).get("overall_score", 0),
                "completion_status": enhanced_results.get("completion_status", "unknown"),
            },
            "legacy": {
                "sources_found": len(legacy_results.get("sources", [])),
                "execution_time": legacy_duration,
                "quality_score": legacy_results.get("quality_metrics", {}).get("overall_score", 0),
                "completion_status": legacy_results.get("completion_status", "unknown"),
            },
            "performance_delta": {}
        }

        # Calculate performance improvements
        if enhanced_duration > 0 and legacy_duration > 0:
            time_improvement = ((legacy_duration - enhanced_duration) / legacy_duration) * 100
            mission_results["comparison_results"]["performance_delta"] = {
                "time_improvement_percent": round(time_improvement, 2),
                "sources_difference": enhanced_results.get("sources", []) - legacy_results.get("sources", []),
                "quality_improvement": enhanced_results.get("quality_metrics", {}).get("overall_score", 0) - 
                                     legacy_results.get("quality_metrics", {}).get("overall_score", 0),
            }

        # Display comparison summary
        self._display_comparison_summary(mission_results["comparison_results"])

        # Return the enhanced results as primary (can be configured differently)
        return enhanced_results

    def _display_comparison_summary(self, comparison_results: Dict[str, Any]) -> None:
        """Display performance comparison summary"""

        console.print("\n[bold cyan]📊 Performance Comparison Summary[/bold cyan]")

        comparison_table = Table(show_header=True, header_style="bold cyan")
        comparison_table.add_column("Metric")
        comparison_table.add_column("Enhanced", style="green")
        comparison_table.add_column("Legacy", style="yellow")
        comparison_table.add_column("Improvement", style="blue")

        enhanced = comparison_results.get("enhanced", {})
        legacy = comparison_results.get("legacy", {})
        delta = comparison_results.get("performance_delta", {})

        comparison_table.add_row(
            "Sources Found",
            str(enhanced.get("sources_found", 0)),
            str(legacy.get("sources_found", 0)),
            f"+{delta.get('sources_difference', 0)}"
        )

        comparison_table.add_row(
            "Execution Time",
            f"{enhanced.get('execution_time', 0):.2f}s",
            f"{legacy.get('execution_time', 0):.2f}s",
            f"{delta.get('time_improvement_percent', 0):+.1f}%"
        )

        comparison_table.add_row(
            "Quality Score",
            f"{enhanced.get('quality_score', 0):.3f}",
            f"{legacy.get('quality_score', 0):.3f}",
            f"{delta.get('quality_improvement', 0):+.3f}"
        )

        console.print(comparison_table)

    async def _execute_writing_stage(
        self,
        research_results: Dict[str, Any],
        research_plan: Dict[str, Any],
        report_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the report writing stage (enhanced with additional metadata)"""

        console.print(
            "[magenta]📝 Final Writer Agent: Synthesizing findings into enhanced comprehensive report...[/magenta]"
        )

        # Add enhanced metadata to research results if available
        if isinstance(self.researcher, EnhancedDeepResearcherAgent):
            # Get additional performance stats
            performance_stats = await self.researcher.get_performance_stats()
            research_results["enhanced_performance_stats"] = performance_stats

        # Generate comprehensive report
        final_report = await self.writer.generate_comprehensive_report(
            research_results=research_results,
            research_plan=research_plan,
            report_config=report_config,
        )

        if final_report.get("status") == "failed":
            raise Exception(
                f"Report generation failed: {final_report.get('error', 'Unknown error')}"
            )

        # Display writing summary
        quality_assessment = final_report.get("quality_assessment", {})
        console.print(f"[green]✅ Enhanced report generation completed:[/green]")
        console.print(f"   • Report ID: {final_report.get('report_id', 'Unknown')}")
        console.print(f"   • Quality Score: {quality_assessment.get('overall_score', 0):.2f}")
        console.print(
            f"   • Reading Time: {quality_assessment.get('estimated_reading_time', 0)} minutes"
        )
        console.print(f"   • Sections: {len(final_report.get('sections', {}))}")

        return final_report

    async def _calculate_enhanced_mission_metrics(self, mission_results: Dict[str, Any]) -> None:
        """Calculate enhanced mission metrics with additional performance data"""

        # Calculate basic metrics (from original)
        if mission_results.get("started_at") and mission_results.get("completed_at"):
            try:
                start = datetime.fromisoformat(mission_results["started_at"])
                end = datetime.fromisoformat(mission_results["completed_at"])
                duration = end - start
                mission_results["total_duration"] = str(duration)
                mission_results["total_duration_seconds"] = int(duration.total_seconds())
            except Exception:
                mission_results["total_duration"] = "Unknown"

        # Enhanced metrics
        research_results = mission_results.get("outputs", {}).get("research_results", {})
        
        # Performance metrics from enhanced researcher
        if "enhanced_performance_stats" in research_results:
            performance_stats = research_results["enhanced_performance_stats"]
            mission_results["performance_metrics"]["tool_composition_stats"] = performance_stats

        # System health metrics
        if isinstance(self.researcher, EnhancedDeepResearcherAgent):
            try:
                health_status = await self.researcher.health_check()
                mission_results["performance_metrics"]["system_health"] = health_status
            except Exception as e:
                console.print(f"[yellow]Warning: Could not get health status: {e}[/yellow]")

        # Comparison metrics (if available)
        if "comparison_results" in mission_results:
            comparison = mission_results["comparison_results"]
            performance_delta = comparison.get("performance_delta", {})
            
            mission_results["performance_metrics"]["comparison_summary"] = {
                "enhancement_enabled": True,
                "time_improvement": performance_delta.get("time_improvement_percent", 0),
                "quality_improvement": performance_delta.get("quality_improvement", 0),
                "sources_improvement": performance_delta.get("sources_difference", 0),
            }

        # Aggregate quality scores
        research_quality = research_results.get("quality_metrics", {}).get("overall_score", 0)
        report_quality = (
            mission_results.get("outputs", {})
            .get("final_report", {})
            .get("quality_assessment", {})
            .get("overall_score", 0)
        )

        mission_results["overall_quality_score"] = (research_quality + report_quality) / 2

        # Enhanced efficiency metrics
        sources = research_results.get("sources", [])
        total_words = sum(source.get("word_count", 0) for source in sources)

        mission_results["total_sources_analyzed"] = len(sources)
        mission_results["total_content_words"] = total_words

        if mission_results.get("total_duration_seconds", 0) > 0:
            duration_minutes = mission_results["total_duration_seconds"] / 60
            mission_results["sources_per_minute"] = round(len(sources) / duration_minutes, 2)
            mission_results["words_per_minute"] = round(total_words / duration_minutes, 0)

    def _display_enhanced_mission_summary(self, mission_results: Dict[str, Any]) -> None:
        """Display enhanced mission summary with performance metrics"""

        # Create enhanced summary table
        table = Table(
            title="📊 Enhanced Research Mission Summary", 
            show_header=True, 
            header_style="bold cyan"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        # Basic info
        table.add_row("Mission ID", mission_results.get("mission_id", "Unknown"))
        table.add_row("Query", mission_results.get("query", "Unknown"))
        table.add_row("Enhanced Mode", "Yes" if mission_results.get("enhanced_mode", False) else "No")
        table.add_row("Status", mission_results.get("status", "Unknown"))

        # Performance metrics
        table.add_row("Total Duration", mission_results.get("total_duration", "Unknown"))
        table.add_row("Overall Quality Score", f"{mission_results.get('overall_quality_score', 0):.2f}")
        table.add_row("Sources Analyzed", str(mission_results.get("total_sources_analyzed", 0)))
        table.add_row("Content Words", f"{mission_results.get('total_content_words', 0):,}")

        # Enhanced metrics
        if mission_results.get("sources_per_minute"):
            table.add_row("Processing Efficiency", f"{mission_results['sources_per_minute']} sources/min")
        if mission_results.get("words_per_minute"):
            table.add_row("Content Efficiency", f"{mission_results['words_per_minute']:,} words/min")

        # Comparison metrics
        performance_metrics = mission_results.get("performance_metrics", {})
        if "comparison_summary" in performance_metrics:
            comparison = performance_metrics["comparison_summary"]
            table.add_row("Time Improvement", f"{comparison.get('time_improvement', 0):+.1f}%")
            table.add_row("Quality Improvement", f"{comparison.get('quality_improvement', 0):+.3f}")

        console.print("\n")
        console.print(table)

        # System health panel (if available)
        if "system_health" in performance_metrics:
            health = performance_metrics["system_health"]
            health_status = health.get("overall_status", "unknown")
            health_color = "green" if health_status == "healthy" else "yellow"
            
            health_panel = Panel(
                f"Overall Status: {health_status}\n"
                f"Tool Composition: {'Active' if health.get('search_pipeline', {}).get('status') == 'healthy' else 'Inactive'}\n"
                f"Performance Monitoring: {'Enabled' if 'search_pipeline_performance' in performance_metrics.get('tool_composition_stats', {}) else 'Disabled'}",
                title="🏥 System Health",
                border_style=health_color
            )
            console.print(health_panel)

    async def _save_enhanced_mission_outputs(
        self, mission_results: Dict[str, Any], save_outputs: bool = True
    ) -> None:
        """Save enhanced mission outputs with additional metadata"""

        mission_id = mission_results.get("mission_id", "unknown")

        console.print(f"[cyan]📋 Processing enhanced research outputs for {mission_id}...[/cyan]")

        # Save main JSON with enhanced data
        await self._save_json_file(mission_results, mission_id)

        # Extract/generate all formats
        await self._ensure_all_formats_exist(mission_results, mission_id)

        # Save additional enhanced components
        if save_outputs:
            await self._save_enhanced_components(mission_results, mission_id)

        # Display final summary
        self._display_saved_files_summary(mission_id)

    async def _save_enhanced_components(
        self, mission_results: Dict[str, Any], mission_id: str
    ) -> None:
        """Save additional enhanced research components"""

        try:
            # Save performance metrics separately
            performance_metrics = mission_results.get("performance_metrics", {})
            if performance_metrics:
                metrics_file = os.path.join(self.output_dir, f"{mission_id}_performance.json")
                with open(metrics_file, "w", encoding="utf-8") as f:
                    json.dump(performance_metrics, f, indent=2, ensure_ascii=False, default=str)
                console.print(f"[green]✅ Saved performance metrics: {os.path.basename(metrics_file)}[/green]")

            # Save comparison results if available
            if "comparison_results" in mission_results:
                comparison_file = os.path.join(self.output_dir, f"{mission_id}_comparison.json")
                with open(comparison_file, "w", encoding="utf-8") as f:
                    json.dump(mission_results["comparison_results"], f, indent=2, ensure_ascii=False, default=str)
                console.print(f"[green]✅ Saved comparison results: {os.path.basename(comparison_file)}[/green]")

            # Save original components
            await self._save_additional_components(mission_results, mission_id)

        except Exception as e:
            console.print(f"[red]❌ Error saving enhanced components: {e}[/red]")

    async def clean_for_json(self, obj, seen: Optional[set] = None) -> Any:
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return None  # Break the cycle
        seen.add(obj_id)
        if isinstance(obj, dict):
            return {k: await self.clean_for_json(v, seen) for k, v in obj.items() if not callable(v) and not hasattr(v, '__dict__')}
        elif isinstance(obj, list):
            return [await self.clean_for_json(i, seen) for i in obj]
        elif isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        else:
            return str(obj)  # fallback for unserializable objects


    # Import methods from original orchestrator (for compatibility)
    async def _save_json_file(self, mission_results: Dict[str, Any], mission_id: str) -> None:
        """Save the complete JSON research file"""
        try:
            # Clean the mission results for JSON serialization
            serializable_results = await self.clean_for_json(mission_results)
            mission_file = os.path.join(self.output_dir, f"{mission_id}_complete.json")
            markdown_file = os.path.join(self.output_dir, f"{mission_id}_report.md")
            text_file = os.path.join(self.output_dir, f"{mission_id}_report.txt")
            with open(mission_file, "w", encoding="utf-8") as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)
            console.print(f"[green]✅ Saved Enhanced JSON: {os.path.basename(mission_file)}[/green]")
            # Also save Markdown and Text formats
            with open(markdown_file, "w", encoding="utf-8") as f:
                f.write(mission_results.get("outputs", {}).get("final_report", {}).get("formatted_outputs", "").get("markdown", ""))
            console.print(f"[green]✅ Saved Markdown Report: {os.path.basename(markdown_file)}[/green]")
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(mission_results.get("outputs", {}).get("final_report", {}).get("formatted_outputs", "").get("text", ""))
            console.print(f"[green]✅ Saved Text Report: {os.path.basename(text_file)}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error saving JSON: {e}[/red]")

    async def _ensure_all_formats_exist(self, mission_results: Dict[str, Any], mission_id: str) -> None:
        """Ensure both Markdown and Text formats exist"""
        # Implementation would mirror the original orchestrator method
        pass  # Placeholder - would copy from original

    async def _save_additional_components(self, mission_results: Dict[str, Any], mission_id: str) -> None:
        """Save additional research components"""
        # Implementation would mirror the original orchestrator method
        pass  # Placeholder - would copy from original

    def _display_saved_files_summary(self, mission_id: str) -> None:
        """Display a summary of all saved files"""
        console.print(f"\n[bold green]📁 Enhanced Auto-Save Complete for {mission_id}:[/bold green]")
        
        files_to_check = [
            (f"{mission_id}_complete.json", "Enhanced JSON data"),
            (f"{mission_id}_report.md", "Markdown report"),
            (f"{mission_id}_report.txt", "Text report"),
            (f"{mission_id}_performance.json", "Performance metrics"),
            (f"{mission_id}_comparison.json", "Comparison results"),
            (f"{mission_id}_plan.json", "Research plan"),
            (f"{mission_id}_research.json", "Research data"),
        ]

        for filename, description in files_to_check:
            filepath = os.path.join(self.output_dir, filename)
            if os.path.exists(filepath):
                file_size = self._get_file_size(filepath)
                console.print(f"   • {description}: {filename} ({file_size})")

        console.print(f"[cyan]📂 All enhanced files saved in: {self.output_dir}/[/cyan]")

    def _get_file_size(self, file_path: str) -> str:
        """Get human-readable file size"""
        try:
            size = os.path.getsize(file_path)
            if size < 1024:
                return f"{size}B"
            elif size < 1024 * 1024:
                return f"{size/1024:.1f}KB"
            else:
                return f"{size/(1024*1024):.1f}MB"
        except Exception:
            return "Unknown"

    def get_enhanced_capabilities(self) -> Dict[str, Any]:
        """Get enhanced system capabilities"""
        
        base_capabilities = {
            "orchestrator_type": "EnhancedMultiAgentResearchOrchestrator",
            "version": "2.0.0",
            "enhanced_mode": isinstance(self.researcher, EnhancedDeepResearcherAgent),
            "performance_comparison": self.enable_performance_comparison,
            "agents": {
                "planner": "ResearchPlannerAgent",
                "researcher": type(self.researcher).__name__,
                "writer": "FinalWriterAgent",
            },
            "features": [
                "comprehensive_research_planning",
                "multi_agent_coordination",
                "automated_report_generation",
                "multi_format_outputs",
                "quality_assessment",
            ]
        }

        # Add enhanced features if enhanced researcher is used
        if isinstance(self.researcher, EnhancedDeepResearcherAgent):
            researcher_capabilities = self.researcher.get_capabilities()
            base_capabilities["enhanced_features"] = researcher_capabilities.get("features", [])
            base_capabilities["tool_composition"] = researcher_capabilities.get("tool_composition", {})
            
            # Add performance monitoring features
            base_capabilities["features"].extend([
                "performance_monitoring",
                "health_checks",
                "system_optimization",
                "comparative_analysis",
            ])

        return base_capabilities

    def display_enhanced_capabilities(self) -> None:
        """Display enhanced system capabilities"""

        enhanced_info = "🚀 **Enhanced Multi-Agent Research System**" if self.use_enhanced_researcher else "🤖 **Multi-Agent Research System**"
        
        capabilities_text = f"""{enhanced_info}

**Research Planner Agent:**
• Creates comprehensive research strategies
• Configures search terms and methodologies  
• Plans multi-phase execution workflows
• Sets quality criteria and fact-checking protocols

**{"Enhanced " if self.use_enhanced_researcher else ""}Deep Researcher Agent:**
• Advanced stealth web browsing with Camoufox
• {"Parallel multi-engine search execution" if self.use_enhanced_researcher else "Multi-engine search execution"}
• {"Intelligent query optimization and filtering" if self.use_enhanced_researcher else "Content extraction and quality analysis"}
• {"Real-time performance monitoring" if self.use_enhanced_researcher else "Source credibility assessment"}
• {"Tool composition with fallback strategies" if self.use_enhanced_researcher else "LLM-enhanced content analysis"}

**Final Writer Agent:**
• Comprehensive report synthesis
• Executive summary generation
• Multi-format output (Markdown, Text, JSON)
• Source bibliography and appendices
• Quality assessment and metrics

**{"Enhanced " if self.use_enhanced_researcher else ""}Features:**
• {"Performance comparison (Enhanced vs Legacy)" if self.enable_performance_comparison else "Research depth configuration"}
• {"System health monitoring and diagnostics" if self.use_enhanced_researcher else "Automated file management"}
• {"Advanced error handling with fallback strategies" if self.use_enhanced_researcher else "Quality scoring and validation"}
• {"Real-time performance optimization" if self.use_enhanced_researcher else "Multi-format report generation"}

**Research Depth Levels:**
• Surface: 15 min, 8 sources - Quick overview
• Medium: 30 min, 12 sources - Balanced analysis  
• Deep: 60 min, 15 sources - Thorough investigation
• Exhaustive: 120 min, 20 sources - Comprehensive study

**Auto-Save Output Formats:**
• Complete JSON research data (always saved)
• Professional Markdown reports (always saved)
• Plain text summaries (always saved)
• {"Performance metrics and system health" if self.use_enhanced_researcher else "Source bibliographies and appendices"}
• {"Comparison analysis results" if self.enable_performance_comparison else "Research metrics and quality assessments"}"""

        capabilities_panel = Panel(
            capabilities_text,
            title="🔬 Enhanced System Overview" if self.use_enhanced_researcher else "🔬 System Overview",
            border_style="cyan",
        )

        console.print(capabilities_panel)

    # Backward compatibility method
    async def execute_research_mission(self, *args, **kwargs) -> Dict[str, Any]:
        """Backward compatibility wrapper for enhanced research mission"""
        return await self.execute_enhanced_research_mission(*args, **kwargs)


# Demonstration function
async def main():
    """Demo of the enhanced multi-agent research system"""

    # Initialize enhanced orchestrator
    orchestrator = EnhancedMultiAgentResearchOrchestrator(
        use_enhanced_researcher=True,
        enable_performance_comparison=True
    )

    # Display enhanced capabilities
    orchestrator.display_enhanced_capabilities()

    # Example enhanced research mission
    query = "What are the latest developments in quantum computing applications for cryptography?"

    results = await orchestrator.execute_enhanced_research_mission(
        query=query, 
        research_depth="medium", 
        report_type="comprehensive", 
        save_outputs=True,
        enable_comparison=True
    )

    console.print("\n[bold green]🎯 Enhanced research mission completed![/bold green]")
    return results


if __name__ == "__main__":
    asyncio.run(main())