#!/usr/bin/env python3
"""
Deep Web Research Tool with CrewAI and Camoufox
Main application entry point
"""

import asyncio
import os
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from agents.research_coordinator import ResearchCoordinator
from tools.research_tools import ResearchToolkit
from models.research_models import ResearchQuery, ResearchReport, ResearchDepth


# Load environment variables
load_dotenv()

console = Console()


class DeepWebResearcher:
    """Main research orchestrator using CrewAI and Camoufox"""

    def __init__(
        self,
        model_name: str = None,
        ollama_base_url: str = None,
        headless: bool = None,
        proxy: Optional[Dict[str, str]] = None,
    ):

        # Load configuration from environment or use defaults
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "magistral:24b")
        self.ollama_base_url = ollama_base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.headless = (
            headless
            if headless is not None
            else os.getenv("CAMOUFOX_HEADLESS", "true").lower() == "true"
        )

        # Proxy configuration
        self.proxy = proxy
        if not self.proxy and os.getenv("PROXY_ENABLED", "false").lower() == "true":
            proxy_server = os.getenv("PROXY_SERVER")
            if proxy_server:
                self.proxy = {
                    "server": proxy_server,
                    "username": os.getenv("PROXY_USERNAME"),
                    "password": os.getenv("PROXY_PASSWORD"),
                }

        # Other settings
        self.request_delay = float(os.getenv("REQUEST_DELAY", "1.0"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.timeout = int(os.getenv("TIMEOUT", "30"))

        # Initialize components
        self.coordinator = ResearchCoordinator(
            model_name=self.model_name, ollama_base_url=self.ollama_base_url
        )

        self.toolkit = ResearchToolkit(
            headless=self.headless, proxy=self.proxy, request_delay=self.request_delay
        )

    async def research(
        self,
        query: str,
        focus_areas: List[str] = None,
        max_sources: int = None,
        depth: str = None,
        fact_check: bool = None,
        exclude_domains: List[str] = None,
    ) -> ResearchReport:
        """Conduct comprehensive research on a query"""

        # Set defaults from environment
        max_sources = max_sources or int(os.getenv("MAX_SOURCES", "15"))
        depth = depth or os.getenv("RESEARCH_DEPTH", "medium")
        fact_check = (
            fact_check
            if fact_check is not None
            else os.getenv("ENABLE_FACT_CHECK", "true").lower() == "true"
        )
        focus_areas = focus_areas or []
        exclude_domains = exclude_domains or []

        console.print(Panel(f"🔍 Starting Deep Web Research", style="bold blue"))
        console.print(f"Query: {query}")
        console.print(f"Max Sources: {max_sources}")
        console.print(f"Depth: {depth}")
        console.print(f"Fact Check: {fact_check}")

        try:
            # Phase 1: Create research plan
            with console.status("[bold green]Creating research plan...", spinner="dots"):
                research_plan = self.coordinator.create_research_plan(
                    query=query, focus_areas=focus_areas, max_sources=max_sources, depth=depth
                )

            console.print("✅ Research plan created")

            # Phase 2: Comprehensive search and content extraction
            with console.status(
                "[bold green]Gathering information from web sources...", spinner="dots"
            ):
                search_results = await self.toolkit.search_comprehensive(
                    query=query,
                    max_sources=max_sources,
                    enable_deep_links=True,
                    max_depth=research_plan["config"]["deep_link_depth"],
                )

            console.print(f"✅ Collected {len(search_results)} sources")

            # Phase 3: Filter excluded domains
            if exclude_domains:
                original_count = len(search_results)
                search_results = [
                    result
                    for result in search_results
                    if not any(
                        domain.lower() in result.get("url", "").lower()
                        for domain in exclude_domains
                    )
                ]
                filtered_count = original_count - len(search_results)
                if filtered_count > 0:
                    console.print(f"🚫 Filtered out {filtered_count} sources from excluded domains")

            # Phase 4: Quality evaluation
            with console.status("[bold green]Evaluating research quality...", spinner="dots"):
                quality_eval = self.coordinator.evaluate_research_quality(search_results)

            console.print(f"✅ Quality Score: {quality_eval['overall_score']:.2f}/1.0")

            # Phase 5: Generate comprehensive report
            with console.status("[bold green]Generating research report...", spinner="dots"):
                summary = self.coordinator.generate_research_summary(
                    query=query, results=search_results, quality_eval=quality_eval
                )

            # Phase 6: Fact checking (if enabled)
            fact_checks = []
            if fact_check and search_results:
                with console.status("[bold green]Performing fact checks...", spinner="dots"):
                    fact_checks = await self._perform_fact_checks(search_results[:5])
                console.print(f"✅ Completed {len(fact_checks)} fact checks")

            # Create final report
            report = self._create_research_report(
                query=query,
                search_results=search_results,
                quality_eval=quality_eval,
                summary=summary,
                fact_checks=fact_checks,
                research_plan=research_plan,
            )

            console.print(Panel("✅ Research Complete!", style="bold green"))

            return report

        except Exception as e:
            console.print(Panel(f"❌ Research failed: {str(e)}", style="bold red"))
            # Return minimal report on error
            return self._create_error_report(query, str(e))

    async def _perform_fact_checks(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform basic fact checking on key claims"""
        fact_checks = []

        for source in sources:
            try:
                # Extract key claims from content
                content = source.get("content", "")
                if len(content) < 100:  # Skip sources with minimal content
                    continue

                # Simple claim extraction (can be enhanced with NLP)
                sentences = content.split(".")
                potential_claims = [
                    s.strip()
                    for s in sentences
                    if len(s.strip()) > 50
                    and any(
                        word in s.lower()
                        for word in ["is", "are", "was", "were", "has", "have", "will", "according"]
                    )
                ]

                if potential_claims:
                    # Take first significant claim
                    claim = potential_claims[0]

                    fact_check = {
                        "claim": claim,
                        "source_url": source.get("url", ""),
                        "source_credibility": source.get("credibility_score", 0.5),
                        "verification_status": "needs_verification",
                        "confidence_score": source.get("credibility_score", 0.5),
                        "details": f"Claim from {source.get('source_type', 'unknown')} source",
                    }
                    fact_checks.append(fact_check)

            except Exception as e:
                console.print(f"Fact check error for {source.get('url', 'unknown')}: {str(e)}")
                continue

        return fact_checks

    def _create_research_report(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        quality_eval: Dict[str, Any],
        summary: str,
        fact_checks: List[Dict[str, Any]],
        research_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a comprehensive research report"""

        return {
            "query": query,
            "executive_summary": summary,
            "research_methodology": f"Comprehensive web research using {len(search_results)} sources",
            "key_findings": self._extract_key_findings(search_results),
            "sources": search_results,
            "fact_checks": fact_checks,
            "quality_metrics": quality_eval,
            "research_plan": research_plan,
            "limitations": self._identify_limitations(search_results, quality_eval),
            "recommendations": quality_eval.get("recommendations", []),
            "generated_at": datetime.now(),
            "total_sources_analyzed": len(search_results),
            "research_depth": research_plan.get("depth", "medium"),
        }

    def _extract_key_findings(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract key findings from research results"""
        findings = []

        # Group by source type
        source_types = {}
        for result in results:
            source_type = result.get("source_type", "unknown")
            if source_type not in source_types:
                source_types[source_type] = []
            source_types[source_type].append(result)

        findings.append(
            f"Analyzed {len(results)} sources across {len(source_types)} different source types"
        )

        # Top credible sources
        high_credibility = [r for r in results if r.get("credibility_score", 0) > 0.7]
        if high_credibility:
            findings.append(f"Found {len(high_credibility)} high-credibility sources")

        # Content volume
        total_words = sum(r.get("word_count", 0) for r in results)
        findings.append(f"Collected {total_words:,} words of content for analysis")

        return findings

    def _identify_limitations(
        self, results: List[Dict[str, Any]], quality_eval: Dict[str, Any]
    ) -> str:
        """Identify research limitations"""
        limitations = []

        if quality_eval["overall_score"] < 0.6:
            limitations.append("Overall research quality could be improved")

        if quality_eval["credibility"] < 0.6:
            limitations.append("Some sources may have lower credibility")

        if quality_eval["diversity"] < 0.6:
            limitations.append("Limited diversity in source types")

        if len(results) < 10:
            limitations.append("Relatively small number of sources analyzed")

        if not limitations:
            limitations.append("No significant limitations identified")

        return "; ".join(limitations)

    def _create_error_report(self, query: str, error_message: str) -> Dict[str, Any]:
        """Create error report when research fails"""
        return {
            "query": query,
            "executive_summary": f"Research failed: {error_message}",
            "research_methodology": "Error occurred during research process",
            "key_findings": [],
            "sources": [],
            "fact_checks": [],
            "quality_metrics": {"overall_score": 0.0},
            "research_plan": {},
            "limitations": "Research could not be completed due to technical issues",
            "recommendations": [
                "Try again with different search terms",
                "Check system configuration",
            ],
            "generated_at": datetime.now(),
            "total_sources_analyzed": 0,
            "research_depth": "none",
        }

    def display_results(self, report: Dict[str, Any]):
        """Display research results in a formatted way"""

        # Main header
        console.print("\n" + "=" * 80)
        console.print(Panel(f"Research Report: {report['query']}", style="bold cyan"))

        # Quality metrics table
        metrics_table = Table(title="Research Quality Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Score", style="green")

        quality = report.get("quality_metrics", {})
        metrics_table.add_row("Overall Score", f"{quality.get('overall_score', 0):.2f}/1.0")
        metrics_table.add_row("Credibility", f"{quality.get('credibility', 0):.2f}/1.0")
        metrics_table.add_row("Source Diversity", f"{quality.get('diversity', 0):.2f}/1.0")
        metrics_table.add_row("Total Sources", str(quality.get("total_sources", 0)))

        console.print(metrics_table)

        # Executive summary
        console.print(
            Panel(
                report.get("executive_summary", "No summary available"),
                title="Executive Summary",
                style="blue",
            )
        )

        # Key findings
        findings = report.get("key_findings", [])
        if findings:
            console.print("\n🔍 Key Findings:")
            for finding in findings:
                console.print(f"  • {finding}")

        # Top sources
        sources = report.get("sources", [])[:5]  # Show top 5
        if sources:
            console.print("\n📚 Top Sources:")
            for i, source in enumerate(sources, 1):
                console.print(f"  {i}. {source.get('title', 'Unknown Title')}")
                console.print(f"     URL: {source.get('url', 'No URL')}")
                console.print(
                    f"     Type: {source.get('source_type', 'Unknown')} | "
                    f"Credibility: {source.get('credibility_score', 0):.2f}"
                )
                console.print()

        # Recommendations
        recommendations = report.get("recommendations", [])
        if recommendations:
            console.print("💡 Recommendations:")
            for rec in recommendations:
                console.print(f"  • {rec}")

        console.print("\n" + "=" * 80)


async def main():
    """Main function for command-line usage"""

    console.print(Panel("Deep Web Research Tool", style="bold magenta"))
    console.print("Powered by CrewAI and Camoufox\n")

    if len(sys.argv) < 2:
        console.print('Usage: python main.py "your research query"')
        console.print('Example: python main.py "artificial intelligence latest developments 2024"')
        return

    query = " ".join(sys.argv[1:])

    # Initialize researcher
    researcher = DeepWebResearcher()

    # Conduct research
    report = await researcher.research(query)

    # Display results
    researcher.display_results(report)


if __name__ == "__main__":
    asyncio.run(main())
