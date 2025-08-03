#!/usr/bin/env python3
"""
Multi-Agent Research Orchestrator
Coordinates Research Planner, Deep Researcher, and Final Writer agents
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
from .final_writer import FinalWriterAgent

# Initialize logger and console
logger = get_logger("orchestrator")
console = Console()


class MultiAgentResearchOrchestrator:
    """
    Orchestrates the complete research workflow using three specialized agents:
    1. Research Planner - Creates comprehensive research plans
    2. Deep Researcher - Executes research and gathers sources
    3. Final Writer - Synthesizes findings into comprehensive reports
    """

    def __init__(self, custom_settings: Optional[Dict[str, Any]] = None):
        """Initialize orchestrator with settings-based configuration"""

        self.settings = get_settings()
        if custom_settings:
            # Override specific settings if provided
            for key, value in custom_settings.items():
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)

        # Create output directory (use research_outputs by default)
        self.output_dir = getattr(self.settings, 'output_dir', 'research_outputs')
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize agents with settings
        logger.info("Initializing Multi-Agent Research System")
        console.print("[bold blue]🤖 Initializing Multi-Agent Research System[/bold blue]")

        self.planner = ResearchPlannerAgent(
            model_name=self.settings.ollama.model_name,
            ollama_base_url=self.settings.ollama.base_url,
        )

        self.researcher = DeepResearcherAgent(
            model_name=self.settings.ollama.model_name,
            ollama_base_url=self.settings.ollama.base_url,
            headless=self.settings.browser.headless,
            proxy=None,  # TODO: Add proxy support to settings
        )

        self.writer = FinalWriterAgent(
            model_name=self.settings.ollama.model_name,
            ollama_base_url=self.settings.ollama.base_url,
        )

        logger.info("All agents initialized successfully")
        console.print("[green]✅ All agents initialized successfully![/green]")

    # For backward compatibility
    @classmethod
    def create_legacy(
        cls,
        model_name: str = "magistral:latest",
        ollama_base_url: str = "http://localhost:11434",
        headless: bool = True,
        proxy: Optional[Dict[str, str]] = None,
        output_dir: str = "research_outputs",
    ) -> "MultiAgentResearchOrchestrator":
        """Create orchestrator with legacy parameters (backward compatibility)"""

        custom_settings = {
            "ollama": {"model_name": model_name, "base_url": ollama_base_url},
            "browser": {"headless": headless},
            "output_dir": output_dir,
        }

        return cls(custom_settings=custom_settings)

    @resilient_operation(
        retry_config=RetryConfig(max_attempts=2, min_wait=5.0),
        circuit_breaker_config=CircuitBreakerConfig(failure_threshold=3, timeout=300),
    )
    async def execute_research_mission(
        self,
        query: str,
        research_depth: str = "medium",
        report_type: str = "comprehensive",
        save_outputs: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a complete research mission from planning to final report

        Args:
            query: The research question to investigate
            research_depth: "surface", "medium", "deep", or "exhaustive"
            report_type: Type of report to generate ("comprehensive", "executive", "technical")
            save_outputs: Whether to save outputs to files

        Returns:
            Complete research mission results including plan, research, and report
        """

        mission_id = f"mission_{int(datetime.now().timestamp())}"

        console.print(f"\n[bold cyan]🚀 Starting Research Mission: {mission_id}[/bold cyan]")
        console.print(f"[yellow]Query: {query}[/yellow]")
        console.print(f"[yellow]Depth: {research_depth}[/yellow]")
        console.print(f"[yellow]Report Type: {report_type}[/yellow]")

        mission_results = {
            "mission_id": mission_id,
            "query": query,
            "research_depth": research_depth,
            "report_type": report_type,
            "started_at": datetime.now().isoformat(),
            "status": "in_progress",
            "stages": {
                "planning": {"status": "pending"},
                "research": {"status": "pending"},
                "writing": {"status": "pending"},
            },
            "outputs": {},
        }

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:

                # Stage 1: Research Planning
                planning_task = progress.add_task(
                    "[cyan]🧠 Planning research strategy...", total=None
                )

                console.print("\n[bold blue]Stage 1: Research Planning[/bold blue]")
                research_plan = await self._execute_planning_stage(query, research_depth)

                mission_results["stages"]["planning"] = {
                    "status": "completed",
                    "completed_at": datetime.now().isoformat(),
                    "plan_id": research_plan.get("id"),
                }
                mission_results["outputs"]["research_plan"] = research_plan

                progress.update(planning_task, description="[green]✅ Research plan completed")

                # Stage 2: Deep Research Execution
                research_task = progress.add_task(
                    "[yellow]🔍 Executing deep research...", total=None
                )

                console.print("\n[bold blue]Stage 2: Deep Research Execution[/bold blue]")
                research_results = await self._execute_research_stage(research_plan)

                mission_results["stages"]["research"] = {
                    "status": "completed",
                    "completed_at": datetime.now().isoformat(),
                    "sources_found": len(research_results.get("sources", [])),
                    "quality_score": research_results.get("quality_metrics", {}).get(
                        "overall_score", 0
                    ),
                }
                mission_results["outputs"]["research_results"] = research_results

                progress.update(research_task, description="[green]✅ Research execution completed")

                # Stage 3: Report Generation
                writing_task = progress.add_task(
                    "[magenta]📝 Generating comprehensive report...", total=None
                )

                console.print("\n[bold blue]Stage 3: Report Generation[/bold blue]")

                report_config = {
                    "report_type": report_type,
                    "include_appendices": True,
                    "format_options": ["markdown", "text", "json"],
                }

                final_report = await self._execute_writing_stage(
                    research_results, research_plan, report_config
                )

                mission_results["stages"]["writing"] = {
                    "status": "completed",
                    "completed_at": datetime.now().isoformat(),
                    "report_id": final_report.get("report_id"),
                    "quality_score": final_report.get("quality_assessment", {}).get(
                        "overall_score", 0
                    ),
                }
                mission_results["outputs"]["final_report"] = final_report

                progress.update(writing_task, description="[green]✅ Report generation completed")

            # Mission completion
            mission_results["status"] = "completed"
            mission_results["completed_at"] = datetime.now().isoformat()

            # Calculate overall metrics
            await self._calculate_mission_metrics(mission_results)

            # Display results summary
            self._display_mission_summary(mission_results)

            # Always save outputs automatically after research completion
            console.print(f"\n[cyan]💾 Auto-saving research outputs in all formats...[/cyan]")
            await self._save_mission_outputs_with_extraction(mission_results, save_outputs)

            console.print(f"\n[bold green]🎉 Research Mission Completed Successfully![/bold green]")

            return mission_results

        except Exception as e:
            mission_results["status"] = "failed"
            mission_results["error"] = str(e)
            mission_results["failed_at"] = datetime.now().isoformat()

            console.print(f"\n[bold red]❌ Research Mission Failed: {e}[/bold red]")
            return mission_results

    async def _execute_planning_stage(self, query: str, research_depth: str) -> Dict[str, Any]:
        """Execute the research planning stage"""

        console.print(
            "[cyan]🧠 Research Planner Agent: Creating comprehensive research strategy...[/cyan]"
        )

        # Create research plan
        research_plan = self.planner.create_comprehensive_plan(
            query=query, research_depth=research_depth
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
        """Execute the deep research stage"""

        console.print(
            "[yellow]🔍 Deep Researcher Agent: Executing comprehensive web research...[/yellow]"
        )

        # Execute research plan
        research_results = await self.researcher.execute_research_plan(research_plan)

        if research_results.get("completion_status") == "failed":
            raise Exception(f"Research failed: {research_results.get('error', 'Unknown error')}")

        # Display research summary
        quality_metrics = research_results.get("quality_metrics", {})
        console.print(f"[green]✅ Research execution completed:[/green]")
        console.print(f"   • Sources Found: {len(research_results.get('sources', []))}")
        console.print(f"   • Quality Score: {quality_metrics.get('overall_score', 0):.2f}")
        console.print(f"   • Avg Credibility: {quality_metrics.get('avg_credibility', 0):.2f}")
        console.print(f"   • Total Words: {quality_metrics.get('total_words', 0):,}")

        return research_results

    async def _execute_writing_stage(
        self,
        research_results: Dict[str, Any],
        research_plan: Dict[str, Any],
        report_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the report writing stage"""

        console.print(
            "[magenta]📝 Final Writer Agent: Synthesizing findings into comprehensive report...[/magenta]"
        )

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
        console.print(f"[green]✅ Report generation completed:[/green]")
        console.print(f"   • Report ID: {final_report.get('report_id', 'Unknown')}")
        console.print(f"   • Quality Score: {quality_assessment.get('overall_score', 0):.2f}")
        console.print(
            f"   • Reading Time: {quality_assessment.get('estimated_reading_time', 0)} minutes"
        )
        console.print(f"   • Sections: {len(final_report.get('sections', {}))}")

        return final_report

    async def _calculate_mission_metrics(self, mission_results: Dict[str, Any]) -> None:
        """Calculate overall mission metrics"""

        # Calculate total duration
        if mission_results.get("started_at") and mission_results.get("completed_at"):
            try:
                start = datetime.fromisoformat(mission_results["started_at"])
                end = datetime.fromisoformat(mission_results["completed_at"])
                duration = end - start
                mission_results["total_duration"] = str(duration)
                mission_results["total_duration_seconds"] = int(duration.total_seconds())
            except Exception:
                mission_results["total_duration"] = "Unknown"

        # Aggregate quality scores
        research_quality = (
            mission_results.get("outputs", {})
            .get("research_results", {})
            .get("quality_metrics", {})
            .get("overall_score", 0)
        )
        report_quality = (
            mission_results.get("outputs", {})
            .get("final_report", {})
            .get("quality_assessment", {})
            .get("overall_score", 0)
        )

        mission_results["overall_quality_score"] = (research_quality + report_quality) / 2

        # Count total resources
        sources = mission_results.get("outputs", {}).get("research_results", {}).get("sources", [])
        total_words = sum(source.get("word_count", 0) for source in sources)

        mission_results["total_sources_analyzed"] = len(sources)
        mission_results["total_content_words"] = total_words

        # Calculate efficiency metrics
        if mission_results.get("total_duration_seconds", 0) > 0:
            sources_per_minute = len(sources) / (mission_results["total_duration_seconds"] / 60)
            mission_results["sources_per_minute"] = round(sources_per_minute, 2)

    def _display_mission_summary(self, mission_results: Dict[str, Any]) -> None:
        """Display comprehensive mission summary"""

        # Create summary table
        table = Table(
            title="📊 Research Mission Summary", show_header=True, header_style="bold cyan"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        # Basic info
        table.add_row("Mission ID", mission_results.get("mission_id", "Unknown"))
        table.add_row("Query", mission_results.get("query", "Unknown"))
        table.add_row("Research Depth", mission_results.get("research_depth", "Unknown"))
        table.add_row("Status", mission_results.get("status", "Unknown"))

        # Performance metrics
        table.add_row("Total Duration", mission_results.get("total_duration", "Unknown"))
        table.add_row(
            "Overall Quality Score", f"{mission_results.get('overall_quality_score', 0):.2f}"
        )
        table.add_row("Sources Analyzed", str(mission_results.get("total_sources_analyzed", 0)))
        table.add_row("Content Words", f"{mission_results.get('total_content_words', 0):,}")

        if mission_results.get("sources_per_minute"):
            table.add_row("Efficiency", f"{mission_results['sources_per_minute']} sources/min")

        console.print("\n")
        console.print(table)

        # Stage status panel
        stages = mission_results.get("stages", {})
        stage_status = []

        for stage_name, stage_data in stages.items():
            status = stage_data.get("status", "unknown")
            status_icon = "✅" if status == "completed" else "❌" if status == "failed" else "⏳"
            stage_status.append(f"{status_icon} {stage_name.title()}: {status}")

        stage_panel = Panel(
            "\n".join(stage_status), title="🔄 Stage Completion Status", border_style="blue"
        )

        console.print(stage_panel)

    async def _save_mission_outputs_with_extraction(
        self, mission_results: Dict[str, Any], save_outputs: bool = True
    ) -> None:
        """
        Save mission outputs and automatically extract/generate all formats
        This method ALWAYS runs after research completion to ensure all formats are created
        """

        mission_id = mission_results.get("mission_id", "unknown")

        console.print(f"[cyan]📋 Processing research outputs for {mission_id}...[/cyan]")

        # Always save the complete JSON file first
        await self._save_json_file(mission_results, mission_id)

        # Always extract or generate Markdown and Text formats
        await self._ensure_all_formats_exist(mission_results, mission_id)

        # Save additional component files if full save is requested
        if save_outputs:
            await self._save_additional_components(mission_results, mission_id)

        # Display final summary
        self._display_saved_files_summary(mission_id)
    
    async def fill_missing_json_fields(self, data, dummy="dummy"):
        if isinstance(data, dict):
            return {k: await self.fill_missing_json_fields(v, dummy) if v not in [None, ""] else dummy for k, v in data.items()}
        elif isinstance(data, list):
            return [await self.fill_missing_json_fields(item, dummy) for item in data]
        else:
            return data

    async def _save_json_file(self, mission_results: Dict[str, Any], mission_id: str) -> None:
        """Save the complete JSON research file
        THIS IS A TEMP SOLUTION THAT WILL BE FIXED IN THE FUTURE
        """

        try:
            mission_results = await self.fill_missing_json_fields(mission_results)
            mission_file = os.path.join(self.output_dir, f"{mission_id}_complete.json")
            with open(mission_file, "w", encoding="utf-8") as f:
                json.dump(mission_results, f, indent=2, ensure_ascii=False, default=str)
            console.print(f"[green]✅ Saved JSON: {os.path.basename(mission_file)}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Error saving JSON: {e}[/red]")

    async def _ensure_all_formats_exist(
        self, mission_results: Dict[str, Any], mission_id: str
    ) -> None:
        """Ensure both Markdown and Text formats exist, generating them if needed"""

        final_report = mission_results.get("outputs", {}).get("final_report", {})
        formatted_outputs = final_report.get("formatted_outputs", {})

        # Handle Markdown format
        if "markdown" in formatted_outputs and formatted_outputs["markdown"]:
            # Extract existing Markdown
            md_file = os.path.join(self.output_dir, f"{mission_id}_report.md")
            try:
                with open(md_file, "w", encoding="utf-8") as f:
                    f.write(formatted_outputs["markdown"])
                console.print(f"[green]✅ Extracted Markdown: {os.path.basename(md_file)}[/green]")
            except Exception as e:
                console.print(f"[red]❌ Error saving Markdown: {e}[/red]")
                await self._generate_fallback_markdown(mission_results, mission_id)
        else:
            # Generate fallback Markdown
            console.print(f"[yellow]⚠️ No formatted Markdown found, generating fallback...[/yellow]")
            await self._generate_fallback_markdown(mission_results, mission_id)

        # Handle Text format
        if "text" in formatted_outputs and formatted_outputs["text"]:
            # Extract existing Text
            txt_file = os.path.join(self.output_dir, f"{mission_id}_report.txt")
            try:
                with open(txt_file, "w", encoding="utf-8") as f:
                    f.write(formatted_outputs["text"])
                console.print(f"[green]✅ Extracted Text: {os.path.basename(txt_file)}[/green]")
            except Exception as e:
                console.print(f"[red]❌ Error saving Text: {e}[/red]")
                await self._generate_fallback_text(mission_results, mission_id)
        else:
            # Generate fallback Text
            console.print(f"[yellow]⚠️ No formatted Text found, generating fallback...[/yellow]")
            await self._generate_fallback_text(mission_results, mission_id)

    async def _save_additional_components(
        self, mission_results: Dict[str, Any], mission_id: str
    ) -> None:
        """Save additional research components (research plan, research data)"""

        try:
            # Save research plan separately
            research_plan = mission_results.get("outputs", {}).get("research_plan", {})
            if research_plan:
                plan_file = os.path.join(self.output_dir, f"{mission_id}_plan.json")
                with open(plan_file, "w", encoding="utf-8") as f:
                    json.dump(research_plan, f, indent=2, ensure_ascii=False, default=str)
                console.print(
                    f"[green]✅ Saved research plan: {os.path.basename(plan_file)}[/green]"
                )

            # Save research results separately
            research_results = mission_results.get("outputs", {}).get("research_results", {})
            if research_results:
                results_file = os.path.join(self.output_dir, f"{mission_id}_research.json")
                with open(results_file, "w", encoding="utf-8") as f:
                    json.dump(research_results, f, indent=2, ensure_ascii=False, default=str)
                console.print(
                    f"[green]✅ Saved research data: {os.path.basename(results_file)}[/green]"
                )

        except Exception as e:
            console.print(f"[red]❌ Error saving additional components: {e}[/red]")

    def _display_saved_files_summary(self, mission_id: str) -> None:
        """Display a summary of all saved files"""

        console.print(f"\n[bold green]📁 Auto-Save Complete for {mission_id}:[/bold green]")

        files_to_check = [
            (f"{mission_id}_complete.json", "Complete JSON data"),
            (f"{mission_id}_report.md", "Markdown report"),
            (f"{mission_id}_report.txt", "Text report"),
            (f"{mission_id}_plan.json", "Research plan"),
            (f"{mission_id}_research.json", "Research data"),
        ]

        for filename, description in files_to_check:
            filepath = os.path.join(self.output_dir, filename)
            if os.path.exists(filepath):
                file_size = self._get_file_size(filepath)
                console.print(f"   • {description}: {filename} ({file_size})")

        console.print(f"[cyan]📂 All files saved in: {self.output_dir}/[/cyan]")

    async def _save_mission_outputs(self, mission_results: Dict[str, Any]) -> None:
        """Save mission outputs to files - Always saves JSON, Markdown, and Text formats"""

        mission_id = mission_results.get("mission_id", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        console.print(f"\n[cyan]💾 Saving mission outputs in all formats...[/cyan]")

        saved_files = []

        try:
            # 1. ALWAYS Save complete mission results as JSON (main file)
            mission_file = os.path.join(self.output_dir, f"{mission_id}_complete.json")
            with open(mission_file, "w", encoding="utf-8") as f:
                json.dump(mission_results, f, indent=2, ensure_ascii=False, default=str)
            console.print(f"[green]✅ Saved complete mission: {mission_file}[/green]")
            saved_files.append(("JSON", mission_file))

            # 2. ALWAYS Extract and save formatted outputs (Markdown & Text)
            final_report = mission_results.get("outputs", {}).get("final_report", {})
            formatted_outputs = final_report.get("formatted_outputs", {})

            # Save Markdown format
            if "markdown" in formatted_outputs and formatted_outputs["markdown"]:
                md_file = os.path.join(self.output_dir, f"{mission_id}_report.md")
                with open(md_file, "w", encoding="utf-8") as f:
                    f.write(formatted_outputs["markdown"])
                console.print(f"[green]✅ Saved Markdown report: {md_file}[/green]")
                saved_files.append(("Markdown", md_file))
            else:
                console.print(
                    f"[yellow]⚠️ Warning: No markdown output found in formatted_outputs[/yellow]"
                )
                # Try to generate fallback markdown from sections
                await self._generate_fallback_markdown(mission_results, mission_id)

            # Save Text format
            if "text" in formatted_outputs and formatted_outputs["text"]:
                txt_file = os.path.join(self.output_dir, f"{mission_id}_report.txt")
                with open(txt_file, "w", encoding="utf-8") as f:
                    f.write(formatted_outputs["text"])
                console.print(f"[green]✅ Saved text report: {txt_file}[/green]")
                saved_files.append(("Text", txt_file))
            else:
                console.print(
                    f"[yellow]⚠️ Warning: No text output found in formatted_outputs[/yellow]"
                )
                # Try to generate fallback text from sections
                await self._generate_fallback_text(mission_results, mission_id)

            # 3. Save additional component files (optional)

            # Save research plan separately
            research_plan = mission_results.get("outputs", {}).get("research_plan", {})
            if research_plan:
                plan_file = os.path.join(self.output_dir, f"{mission_id}_plan.json")
                with open(plan_file, "w", encoding="utf-8") as f:
                    json.dump(research_plan, f, indent=2, ensure_ascii=False, default=str)
                console.print(f"[green]✅ Saved research plan: {plan_file}[/green]")
                saved_files.append(("Research Plan", plan_file))

            # Save research results separately
            research_results = mission_results.get("outputs", {}).get("research_results", {})
            if research_results:
                results_file = os.path.join(self.output_dir, f"{mission_id}_research.json")
                with open(results_file, "w", encoding="utf-8") as f:
                    json.dump(research_results, f, indent=2, ensure_ascii=False, default=str)
                console.print(f"[green]✅ Saved research results: {results_file}[/green]")
                saved_files.append(("Research Data", results_file))

            # 4. Display summary of saved files
            console.print(f"\n[bold green]📁 Output Summary:[/bold green]")
            for file_type, file_path in saved_files:
                file_size = self._get_file_size(file_path)
                console.print(f"   • {file_type}: {os.path.basename(file_path)} ({file_size})")

        except Exception as e:
            console.print(f"[red]❌ Error saving outputs: {e}[/red]")
            # Try to save at least the JSON file as backup
            try:
                backup_file = os.path.join(self.output_dir, f"{mission_id}_backup.json")
                with open(backup_file, "w", encoding="utf-8") as f:
                    json.dump(mission_results, f, indent=2, ensure_ascii=False, default=str)
                console.print(f"[yellow]⚡ Saved backup file: {backup_file}[/yellow]")
            except Exception as backup_error:
                console.print(f"[red]❌ Failed to save backup: {backup_error}[/red]")

    async def _generate_fallback_markdown(
        self, mission_results: Dict[str, Any], mission_id: str
    ) -> None:
        """Generate fallback markdown if formatted output is missing"""

        try:
            final_report = mission_results.get("outputs", {}).get("final_report", {})
            sections = final_report.get("sections", {})
            metadata = final_report.get("metadata", {})

            if not sections:
                console.print(f"[yellow]⚠️ No sections available for fallback markdown[/yellow]")
                return

            # Generate basic markdown structure
            markdown = f"""# Research Report: {mission_results.get('query', 'Research Mission')}

**Mission ID:** {mission_id}  
**Generated:** {mission_results.get('completed_at', 'Unknown')}  
**Sources Analyzed:** {metadata.get('total_sources', 0)}  

---

## Executive Summary

{sections.get('executive_summary', 'Executive summary not available.')}

---

## Introduction

{sections.get('introduction', 'Introduction not available.')}

---

## Methodology

{sections.get('methodology', 'Methodology not available.')}

---

## Findings

{sections.get('findings', 'Findings not available.')}

---

## Analysis

{sections.get('analysis', 'Analysis not available.')}

---

## Recommendations

{sections.get('recommendations', 'Recommendations not available.')}

---

*Generated by Multi-Agent Research System*
"""

            md_file = os.path.join(self.output_dir, f"{mission_id}_report.md")
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(markdown)
            console.print(f"[green]✅ Generated fallback Markdown: {md_file}[/green]")

        except Exception as e:
            console.print(f"[red]❌ Failed to generate fallback markdown: {e}[/red]")

    async def _generate_fallback_text(
        self, mission_results: Dict[str, Any], mission_id: str
    ) -> None:
        """Generate fallback text if formatted output is missing"""

        try:
            final_report = mission_results.get("outputs", {}).get("final_report", {})
            sections = final_report.get("sections", {})
            metadata = final_report.get("metadata", {})

            if not sections:
                console.print(f"[yellow]⚠️ No sections available for fallback text[/yellow]")
                return

            # Generate basic text structure
            text = f"""RESEARCH REPORT: {mission_results.get('query', 'RESEARCH MISSION').upper()}

Mission ID: {mission_id}
Generated: {mission_results.get('completed_at', 'Unknown')}
Sources Analyzed: {metadata.get('total_sources', 0)}

{'='*80}

EXECUTIVE SUMMARY

{sections.get('executive_summary', 'Executive summary not available.')}

{'='*80}

INTRODUCTION

{sections.get('introduction', 'Introduction not available.')}

{'='*80}

METHODOLOGY

{sections.get('methodology', 'Methodology not available.')}

{'='*80}

FINDINGS

{sections.get('findings', 'Findings not available.')}

{'='*80}

ANALYSIS

{sections.get('analysis', 'Analysis not available.')}

{'='*80}

RECOMMENDATIONS

{sections.get('recommendations', 'Recommendations not available.')}

{'='*80}

Generated by Multi-Agent Research System
"""

            txt_file = os.path.join(self.output_dir, f"{mission_id}_report.txt")
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(text)
            console.print(f"[green]✅ Generated fallback text: {txt_file}[/green]")

        except Exception as e:
            console.print(f"[red]❌ Failed to generate fallback text: {e}[/red]")

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

    def get_available_depth_levels(self) -> Dict[str, Dict[str, Any]]:
        """Get available research depth configurations"""
        # Convert enum keys to string keys
        return {depth.value: config for depth, config in ResearchConfig.DEPTH_CONFIGS.items()}

    async def extract_formatted_outputs_from_json(self, json_file_path: str) -> bool:
        """
        Extract and save Markdown and Text outputs from an existing JSON research file

        Args:
            json_file_path: Path to the JSON file containing research results

        Returns:
            bool: True if extraction successful, False otherwise
        """

        try:
            console.print(f"\n[cyan]📄 Extracting formatted outputs from: {json_file_path}[/cyan]")

            # Load the JSON file
            with open(json_file_path, "r", encoding="utf-8") as f:
                mission_results = json.load(f)

            # Extract mission ID from the data or filename
            mission_id = mission_results.get("mission_id")
            if not mission_id:
                # Try to extract from filename
                filename = os.path.basename(json_file_path)
                if "_complete.json" in filename:
                    mission_id = filename.replace("_complete.json", "")
                else:
                    mission_id = f"extracted_{int(datetime.now().timestamp())}"

            # Extract formatted outputs
            final_report = mission_results.get("outputs", {}).get("final_report", {})
            formatted_outputs = final_report.get("formatted_outputs", {})

            extracted_files = []

            # Extract Markdown
            if "markdown" in formatted_outputs and formatted_outputs["markdown"]:
                md_file = os.path.join(self.output_dir, f"{mission_id}_report.md")
                with open(md_file, "w", encoding="utf-8") as f:
                    f.write(formatted_outputs["markdown"])
                console.print(f"[green]✅ Extracted Markdown: {md_file}[/green]")
                extracted_files.append(("Markdown", md_file))
            else:
                # Generate fallback
                await self._generate_fallback_markdown(mission_results, mission_id)
                md_file = os.path.join(self.output_dir, f"{mission_id}_report.md")
                extracted_files.append(("Markdown (Fallback)", md_file))

            # Extract Text
            if "text" in formatted_outputs and formatted_outputs["text"]:
                txt_file = os.path.join(self.output_dir, f"{mission_id}_report.txt")
                with open(txt_file, "w", encoding="utf-8") as f:
                    f.write(formatted_outputs["text"])
                console.print(f"[green]✅ Extracted Text: {txt_file}[/green]")
                extracted_files.append(("Text", txt_file))
            else:
                # Generate fallback
                await self._generate_fallback_text(mission_results, mission_id)
                txt_file = os.path.join(self.output_dir, f"{mission_id}_report.txt")
                extracted_files.append(("Text (Fallback)", txt_file))

            # Display summary
            console.print(f"\n[bold green]📁 Extraction Summary:[/bold green]")
            for file_type, file_path in extracted_files:
                file_size = self._get_file_size(file_path)
                console.print(f"   • {file_type}: {os.path.basename(file_path)} ({file_size})")

            return True

        except Exception as e:
            console.print(f"[red]❌ Failed to extract formatted outputs: {e}[/red]")
            return False

    def ensure_auto_save_enabled(self) -> None:
        """
        Ensure that save_outputs is always enabled for research missions.
        This guarantees that JSON, Markdown, and Text files are always created.
        """

        console.print(
            f"[green]✅ Auto-save is enabled - All research outputs will be saved automatically[/green]"
        )
        console.print(f"[cyan]📁 Output directory: {self.output_dir}[/cyan]")
        console.print(
            f"[yellow]💡 Files saved: JSON + Markdown + Text for every research mission[/yellow]"
        )

    def list_saved_research_missions(self) -> List[Dict[str, str]]:
        """List all saved research missions with their file types"""

        missions = []

        try:
            if not os.path.exists(self.output_dir):
                return missions

            # Find all JSON mission files
            for filename in os.listdir(self.output_dir):
                if filename.endswith("_complete.json"):
                    mission_id = filename.replace("_complete.json", "")

                    mission_info = {
                        "mission_id": mission_id,
                        "json_file": os.path.join(self.output_dir, filename),
                        "markdown_file": None,
                        "text_file": None,
                    }

                    # Check for corresponding markdown and text files
                    md_file = os.path.join(self.output_dir, f"{mission_id}_report.md")
                    txt_file = os.path.join(self.output_dir, f"{mission_id}_report.txt")

                    if os.path.exists(md_file):
                        mission_info["markdown_file"] = md_file
                    if os.path.exists(txt_file):
                        mission_info["text_file"] = txt_file

                    missions.append(mission_info)

            return sorted(missions, key=lambda x: x["mission_id"], reverse=True)

        except Exception as e:
            console.print(f"[red]❌ Error listing missions: {e}[/red]")
            return []

    def display_capabilities(self) -> None:
        """Display system capabilities"""

        capabilities_panel = Panel(
            """🤖 **Multi-Agent Research System Capabilities**

**Research Planner Agent:**
• Creates comprehensive research strategies
• Configures search terms and methodologies  
• Plans multi-phase execution workflows
• Sets quality criteria and fact-checking protocols

**Deep Researcher Agent:**
• Advanced stealth web browsing with Camoufox
• Multi-phase search execution
• Content extraction and quality analysis
• Source credibility assessment
• LLM-enhanced content analysis

**Final Writer Agent:**
• Comprehensive report synthesis
• Executive summary generation
• Multi-format output (Markdown, Text, JSON)
• Source bibliography and appendices
• Quality assessment and metrics

**Research Depth Levels:**
• Surface: 15 min, 8 sources - Quick overview
• Medium: 30 min, 12 sources - Balanced analysis  
• Deep: 60 min, 15 sources - Thorough investigation
• Exhaustive: 120 min, 20 sources - Comprehensive study

**Auto-Save Output Formats:**
• Complete JSON research data (always saved)
• Professional Markdown reports (always saved)
• Plain text summaries (always saved)
• Source bibliographies and appendices
• Research metrics and quality assessments

**File Management:**
• Automatic extraction of formatted outputs from JSON
• Fallback generation if formatted outputs missing
• File size tracking and output summaries
• Mission listing and management utilities""",
            title="🔬 System Overview",
            border_style="cyan",
        )

        console.print(capabilities_panel)


async def main():
    """Demo of the multi-agent research system"""

    # Initialize orchestrator with new settings-based approach
    orchestrator = MultiAgentResearchOrchestrator()

    # Display capabilities
    orchestrator.display_capabilities()

    # Example research mission
    query = "What are the latest developments in quantum computing applications for cryptography?"

    results = await orchestrator.execute_research_mission(
        query=query, research_depth="medium", report_type="comprehensive", save_outputs=True
    )

    console.print("\n[bold green]🎯 Research mission completed![/bold green]")
    return results


if __name__ == "__main__":
    asyncio.run(main())
