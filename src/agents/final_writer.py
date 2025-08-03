#!/usr/bin/env python3
"""
Final Writer Agent
Specialized in comprehensive report generation and synthesis
"""

import json
import re
import time
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import Counter

from crewai import Agent
from langchain_ollama import ChatOllama
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

console = Console()


class FinalWriterAgent:
    """Agent responsible for synthesizing research into comprehensive reports"""

    def __init__(
        self,
        model_name: str = "magistral:latest",
        content_model_name: str = "granite3.3:8b",
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.content_model_name = content_model_name
        self.ollama_base_url = ollama_base_url

        # Main LLM for comprehensive report generation (magistral)
        self.llm = ChatOllama(
            model=model_name,
            base_url=ollama_base_url,
            temperature=0.2,  # Slightly higher for creative writing
            num_ctx=40000,
        )

        # Content processing LLM for quick analysis tasks (granite3.3:8b)
        self.content_llm = ChatOllama(
            model=content_model_name,
            base_url=ollama_base_url,
            temperature=0.1,
            num_ctx=40000,  # Extended context for better content processing
        )

        console.print(f"[cyan]📝 Main Writing Model: {model_name}[/cyan]")
        console.print(f"[cyan]⚡ Content Processing Model: {content_model_name}[/cyan]")

        # Initialize the writer agent
        self.agent = Agent(
            role="Research Report Writer",
            goal="Create comprehensive, well-structured research reports that synthesize findings into actionable insights",
            backstory="""You are an expert research analyst and technical writer with advanced degrees
            in data science and journalism. With over a decade of experience in academic and
            investigative research, you excel at:
            
            - Synthesizing complex information from multiple sources
            - Creating compelling narratives from data and findings
            - Structuring comprehensive reports with clear sections
            - Identifying key insights and actionable recommendations
            - Writing for diverse audiences and technical levels
            - Fact-checking and source verification
            - Creating executive summaries and detailed analyses
            
            Your reports are known for their clarity, depth, and practical value. You approach
            each writing task with rigorous methodology while maintaining engaging, accessible prose.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )

    async def generate_comprehensive_report(
        self,
        research_results: Dict[str, Any],
        research_plan: Dict[str, Any],
        report_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a comprehensive research report from research results with performance optimizations"""

        report_config = report_config or {}

        # Debug: Check if research_results is actually a dictionary
        if not isinstance(research_results, dict):
            console.print(
                f"[red]❌ Error: research_results is not a dictionary, it's a {type(research_results)}[/red]"
            )
            console.print(f"[red]Content preview: {str(research_results)[:200]}...[/red]")
            return {
                "status": "failed",
                "error": f"Invalid research_results type: expected dict, got {type(research_results)}",
                "report_id": f"error_{int(datetime.now().timestamp())}",
                "generated_at": datetime.now().isoformat(),
            }

        console.print(f"[bold blue]📝 Generating Comprehensive Report[/bold blue]")
        console.print(f"[yellow]Query: {research_results.get('query', 'Unknown')}[/yellow]")
        console.print(f"[yellow]Sources: {len(research_results.get('sources', []))}[/yellow]")

        # Start performance timer
        start_time = time.time()

        # Validate research_results structure
        if not research_results.get("query"):
            console.print(f"[red]❌ Warning: No query found in research_results[/red]")
        if not research_results.get("sources"):
            console.print(f"[red]❌ Warning: No sources found in research_results[/red]")

        report_data = {
            "report_id": f"report_{int(datetime.now().timestamp())}",
            "generated_at": datetime.now().isoformat(),
            "query": research_results.get("query", "Unknown Query"),
            "research_session_id": research_results.get("session_id"),
            "report_type": report_config.get("report_type", "comprehensive"),
            "sections": {},
            "metadata": {},
            "quality_assessment": {},
        }

        try:
            # Use fast analysis for large datasets
            sources_count = (
                len(research_results.get("sources", []))
                if isinstance(research_results, dict)
                else 0
            )
            if sources_count > 20:
                console.print(
                    f"[cyan]⚡ Large dataset detected ({sources_count} sources), using fast generation mode[/cyan]"
                )
                analysis = await self._fast_analyze_research_data(research_results, research_plan)
            else:
                console.print(
                    f"[cyan]🔍 Standard generation mode for {sources_count} sources[/cyan]"
                )
                analysis = await self._analyze_research_data(research_results, research_plan)

            console.print(f"[green]✅ Analysis completed ({time.time() - start_time:.1f}s)[/green]")
            report_data["analysis_metadata"] = analysis

            # Generate report sections with parallel processing
            console.print(f"[cyan]📝 Generating report sections in parallel...[/cyan]")
            sections_start = time.time()

            sections = await self._generate_report_sections_parallel(
                research_results, research_plan, analysis, report_config
            )
            report_data["sections"] = sections

            console.print(
                f"[green]✅ Sections generated ({time.time() - sections_start:.1f}s)[/green]"
            )

            # Generate executive summary (keep this last as it needs other sections)
            console.print(f"[cyan]📋 Generating executive summary...[/cyan]")
            executive_start = time.time()

            executive_summary = await self._generate_executive_summary_fast(
                research_results, analysis, sections
            )
            report_data["sections"]["executive_summary"] = executive_summary

            console.print(
                f"[green]✅ Executive summary completed ({time.time() - executive_start:.1f}s)[/green]"
            )

            # Quick metadata and quality assessment (no regeneration for speed)
            metadata = self._generate_metadata(research_results, research_plan, report_config)
            quality_assessment = await self._quick_quality_assessment(research_results, sections)

            report_data["metadata"] = metadata
            report_data["quality_assessment"] = quality_assessment

            # Generate formatted outputs
            console.print(f"[cyan]📄 Generating formatted outputs...[/cyan]")
            formatted_outputs = await self._generate_formatted_outputs(report_data)
            report_data["formatted_outputs"] = formatted_outputs

            total_time = time.time() - start_time
            console.print(
                f"[bold green]✅ Report generated successfully in {total_time:.1f} seconds![/bold green]"
            )
            console.print(
                f"[green]📊 Quality Score: {quality_assessment.get('overall_score', 0):.2f}[/green]"
            )

            return report_data

        except Exception as e:
            console.print(f"[red]❌ Report generation failed: {e}[/red]")
            report_data["error"] = str(e)
            report_data["status"] = "failed"
            return report_data

    async def _fast_analyze_research_data(
        self, research_results: Dict[str, Any], research_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fast analysis for large datasets using granite3.3:8b"""

        console.print(f"[cyan]⚡ Fast analysis mode activated[/cyan]")

        # Type safety check
        if not isinstance(research_results, dict):
            console.print(
                f"[red]❌ Error in _fast_analyze_research_data: research_results is {type(research_results)}, not dict[/red]"
            )
            return {
                "source_analysis": {"total_sources": 0, "error": "Invalid input type"},
                "content_themes": {"main_themes": ["error"], "error": True},
                "credibility_distribution": {},
                "key_findings": [],
                "data_gaps": ["Invalid research results format"],
                "fast_mode": True,
                "error": f"Expected dict, got {type(research_results)}",
            }

        sources = research_results.get("sources", [])

        analysis = {
            "source_analysis": {},
            "content_themes": {},
            "credibility_distribution": {},
            "key_findings": [],
            "data_gaps": [],
            "fast_mode": True,
        }

        if not sources:
            return analysis

        # Quick source analysis
        source_types = [s.get("source_type", "unknown") for s in sources]
        analysis["source_analysis"] = {
            "total_sources": len(sources),
            "source_types": dict(Counter(source_types)),
            "avg_credibility": sum(s.get("credibility_score", 0.5) for s in sources) / len(sources),
            "avg_relevance": sum(s.get("relevance_score", 0.5) for s in sources) / len(sources),
        }

        # Fast content themes (using first few sources only)
        sample_sources = sources[:5]  # Limit for speed
        sample_content = " ".join([s.get("content", "")[:500] for s in sample_sources])

        if sample_content:
            try:
                themes = await self._extract_content_themes_fast(sample_content)
                analysis["content_themes"] = themes
            except:
                analysis["content_themes"] = {
                    "themes": ["analysis", "research"],
                    "fast_fallback": True,
                }

        # Quick credibility distribution
        credibility_scores = [s.get("credibility_score", 0.5) for s in sources]
        analysis["credibility_distribution"] = {
            "high_credibility": len([s for s in credibility_scores if s >= 0.8]),
            "medium_credibility": len([s for s in credibility_scores if 0.5 <= s < 0.8]),
            "low_credibility": len([s for s in credibility_scores if s < 0.5]),
        }

        # Fast key findings (from high-quality sources only)
        high_quality_sources = [s for s in sources if s.get("credibility_score", 0) > 0.7][:3]
        analysis["key_findings"] = await self._identify_key_findings_fast(high_quality_sources)

        console.print(f"[green]✅ Fast analysis completed for {len(sources)} sources[/green]")
        return analysis

    async def _extract_content_themes_fast(self, content: str) -> Dict[str, Any]:
        """Fast theme extraction using granite3.3:8b"""

        try:
            theme_prompt = f"""
            Quickly identify 3-5 main themes from this research content:
            
            Content (sample): {content[:1000]}
            
            Return JSON:
            {{
                "main_themes": ["theme1", "theme2", "theme3"],
                "key_concepts": ["concept1", "concept2"]
            }}
            """

            response = self.content_llm.invoke(theme_prompt)
            response_text = response.content if hasattr(response, "content") else str(response)

            if isinstance(response_text, list):
                response_text = str(response_text[0]) if response_text else ""

            try:
                themes = json.loads(response_text)
                return themes
            except:
                return {"main_themes": ["research", "analysis", "findings"], "fast_fallback": True}

        except:
            return {"main_themes": ["research", "analysis", "findings"], "fast_fallback": True}

    async def _identify_key_findings_fast(
        self, sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fast key findings identification"""

        if not sources:
            return []

        findings = []
        for source in sources[:2]:  # Limit to 2 sources for speed
            content_preview = source.get("content", "")[:500]
            finding = {
                "finding_type": "summary",
                "description": f"Key insights from {source.get('title', 'source')}",
                "source_title": source.get("title", "Unknown"),
                "confidence_level": "medium",
                "significance": "moderate",
            }
            findings.append(finding)

        return findings

    async def _generate_report_sections_parallel(
        self,
        research_results: Dict[str, Any],
        research_plan: Dict[str, Any],
        analysis: Dict[str, Any],
        report_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate report sections in parallel for faster processing"""

        console.print(f"[cyan]🚀 Parallel section generation started[/cyan]")

        # Define section generation tasks
        section_tasks = [
            ("introduction", self._generate_introduction_fast(research_results, research_plan)),
            (
                "methodology",
                self._generate_methodology_section_fast(research_plan, research_results),
            ),
            ("findings", self._generate_findings_section_fast(analysis, research_results)),
            ("analysis", self._generate_analysis_section_fast(analysis, research_results)),
            (
                "recommendations",
                self._generate_recommendations_section_fast(
                    analysis, research_results, research_plan
                ),
            ),
        ]

        # Execute all section generation in parallel
        section_results = await asyncio.gather(
            *[task[1] for task in section_tasks], return_exceptions=True
        )

        # Combine results
        sections = {}
        for i, (section_name, _) in enumerate(section_tasks):
            result = section_results[i]
            if isinstance(result, Exception):
                console.print(f"[red]❌ Failed to generate {section_name}: {result}[/red]")
                sections[section_name] = f"Error generating {section_name} section."
            else:
                sections[section_name] = result

        # Generate appendices (non-critical, can be simple)
        sections["appendices"] = await self._generate_appendices_fast(research_results, analysis)

        console.print(f"[green]✅ All sections generated in parallel[/green]")
        return sections

    async def _generate_introduction_fast(
        self, research_results: Dict[str, Any], research_plan: Dict[str, Any]
    ) -> str:
        """Generate introduction quickly"""

        query = research_results.get("query", "Unknown")
        sources_count = len(research_results.get("sources", []))

        intro_prompt = f"""
        Write a brief introduction (200-300 words) for this research:
        
        Research Question: "{query}"
        Sources Analyzed: {sources_count}
        
        Include: research question importance, brief context, objectives, methodology overview.
        Be specific and avoid templates.
        """

        response = self.content_llm.invoke(intro_prompt)
        return self._extract_content(response)

    async def _generate_methodology_section_fast(
        self, research_plan: Dict[str, Any], research_results: Dict[str, Any]
    ) -> str:
        """Generate methodology section quickly"""

        sources_count = len(research_results.get("sources", []))

        methodology_prompt = f"""
        Write a brief methodology section (150-250 words) describing:
        
        - Web-based research using multiple search engines
        - {sources_count} sources analyzed
        - Quality assessment and filtering applied
        - Data processing and synthesis methods
        
        Be concise and specific.
        """

        response = self.content_llm.invoke(methodology_prompt)
        return self._extract_content(response)

    async def _generate_findings_section_fast(
        self, analysis: Dict[str, Any], research_results: Dict[str, Any]
    ) -> str:
        """Generate findings section quickly"""

        key_findings = analysis.get("key_findings", [])
        themes = analysis.get("content_themes", {}).get("main_themes", [])

        findings_prompt = f"""
        Write a findings section (250-350 words) based on:
        
        Key Findings: {json.dumps(key_findings[:3])}
        Main Themes: {themes[:3]}
        Sources: {len(research_results.get("sources", []))} analyzed
        
        Present specific discoveries with supporting evidence.
        """

        response = self.content_llm.invoke(findings_prompt)
        return self._extract_content(response)

    async def _generate_analysis_section_fast(
        self, analysis: Dict[str, Any], research_results: Dict[str, Any]
    ) -> str:
        """Generate analysis section quickly"""

        analysis_prompt = f"""
        Write an analysis section (200-300 words) interpreting the research findings.
        
        Research Topic: {research_results.get("query", "Unknown")}
        Sources Quality: {analysis.get("credibility_distribution", {})}
        
        Focus on: significance of findings, reliability assessment, broader implications.
        """

        response = self.content_llm.invoke(analysis_prompt)
        return self._extract_content(response)

    async def _generate_recommendations_section_fast(
        self,
        analysis: Dict[str, Any],
        research_results: Dict[str, Any],
        research_plan: Dict[str, Any],
    ) -> str:
        """Generate recommendations section quickly"""

        recommendations_prompt = f"""
        Write actionable recommendations (200-250 words) based on:
        
        Research Question: {research_results.get("query", "")}
        Key Findings Available: {len(analysis.get("key_findings", []))}
        
        Provide specific, implementable recommendations with priorities.
        """

        response = self.content_llm.invoke(recommendations_prompt)
        return self._extract_content(response)

    async def _generate_appendices_fast(
        self, research_results: Dict[str, Any], analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate appendices quickly"""

        sources = research_results.get("sources", [])

        appendices = {
            "source_bibliography": [
                {
                    "id": i + 1,
                    "title": s.get("title", "Unknown"),
                    "url": s.get("url", ""),
                    "credibility_score": s.get("credibility_score", 0.5),
                }
                for i, s in enumerate(sources[:20])  # Limit for speed
            ],
            "research_statistics": {
                "total_sources": len(sources),
                "source_analysis": analysis.get("source_analysis", {}),
            },
        }

        return appendices

    async def _generate_executive_summary_fast(
        self, research_results: Dict[str, Any], analysis: Dict[str, Any], sections: Dict[str, Any]
    ) -> str:
        """Generate executive summary quickly"""

        key_findings = analysis.get("key_findings", [])

        summary_prompt = f"""
        Write an executive summary (200-250 words) for:
        
        Research: {research_results.get("query", "")}
        Sources: {len(research_results.get("sources", []))}
        Key Findings: {len(key_findings)}
        
        Include: research question, main findings, key conclusions, top recommendations.
        Write for decision-makers.
        """

        response = self.content_llm.invoke(summary_prompt)
        return self._extract_content(response)

    async def _quick_quality_assessment(
        self, research_results: Dict[str, Any], sections: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Quick quality assessment without regeneration"""

        sources_count = len(research_results.get("sources", []))
        sections_completed = len([s for s in sections.values() if s and len(str(s).strip()) > 50])

        # Simple scoring
        source_score = min(sources_count / 10, 1.0)
        sections_score = sections_completed / 6  # 6 expected sections

        overall_score = (source_score * 0.4 + sections_score * 0.6) * 10

        return {
            "overall_score": min(overall_score, 10.0),
            "source_count": sources_count,
            "sections_completed": sections_completed,
            "assessment_mode": "quick",
            "timestamp": datetime.now().isoformat(),
        }

    async def _quick_content_assessment(self, content: str) -> Dict[str, Any]:
        """Quick content assessment using granite3.3:8b"""

        try:
            # Limit content for fast processing
            content_preview = content[:500] if content else ""

            assessment_prompt = f"""
            Quickly assess this content quality (1-10 scale):
            
            Content: {content_preview}
            
            Return JSON: {{"overall_score": 0-10, "quality": "high/medium/low"}}
            """

            response = self.content_llm.invoke(assessment_prompt)
            response_text = response.content if hasattr(response, "content") else str(response)

            if isinstance(response_text, list):
                response_text = str(response_text[0]) if response_text else ""

            try:
                assessment = json.loads(response_text)
                return assessment
            except:
                # Fallback assessment
                word_count = len(content.split()) if content else 0
                score = min(word_count / 100, 10)  # Simple word-count based scoring
                return {"overall_score": score, "quality": "medium"}

        except:
            return {"overall_score": 5.0, "quality": "medium"}

    async def _smart_section_improvement(self, content: str, section_name: str) -> str:
        """Smart section improvement using granite3.3:8b"""

        try:
            improvement_prompt = f"""
            Improve this {section_name} section content:
            
            Current content: {content[:800]}
            
            Make it more specific, clear, and professional. Return improved version (200-400 words).
            """

            response = self.content_llm.invoke(improvement_prompt)
            improved_content = self._extract_content(response)

            # Return improved content if it's substantially different and better
            if len(improved_content) > len(content) * 0.8:
                return improved_content
            else:
                return content  # Return original if improvement isn't substantial

        except:
            return content  # Return original on error

    async def _analyze_research_data(
        self, research_results: Dict[str, Any], research_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze research data to prepare for report generation"""

        # Type safety check
        if not isinstance(research_results, dict):
            console.print(
                f"[red]❌ Error in _analyze_research_data: research_results is {type(research_results)}, not dict[/red]"
            )
            return {
                "source_analysis": {"total_sources": 0, "error": "Invalid input type"},
                "content_themes": {"main_themes": ["error"], "error": True},
                "credibility_distribution": {},
                "temporal_analysis": {},
                "key_findings": [],
                "data_gaps": ["Invalid research results format"],
                "conflicting_information": [],
                "error": f"Expected dict, got {type(research_results)}",
            }

        sources = research_results.get("sources", [])

        analysis = {
            "source_analysis": {},
            "content_themes": {},
            "credibility_distribution": {},
            "temporal_analysis": {},
            "key_findings": [],
            "data_gaps": [],
            "conflicting_information": [],
        }

        if not sources:
            return analysis

        # Source type analysis
        source_types = [s.get("source_type", "unknown") for s in sources]
        analysis["source_analysis"] = {
            "total_sources": len(sources),
            "source_types": dict(Counter(source_types)),
            "avg_credibility": sum(s.get("credibility_score", 0.5) for s in sources) / len(sources),
            "avg_relevance": sum(s.get("relevance_score", 0.5) for s in sources) / len(sources),
        }

        # Content theme analysis
        all_content = " ".join([s.get("content", "") for s in sources])
        themes = await self._extract_content_themes(all_content)
        analysis["content_themes"] = themes

        # Credibility distribution
        credibility_scores = [s.get("credibility_score", 0.5) for s in sources]
        analysis["credibility_distribution"] = {
            "high_credibility": len([s for s in credibility_scores if s >= 0.8]),
            "medium_credibility": len([s for s in credibility_scores if 0.5 <= s < 0.8]),
            "low_credibility": len([s for s in credibility_scores if s < 0.5]),
        }

        # Identify key findings
        key_findings = await self._identify_key_findings(sources)
        analysis["key_findings"] = key_findings

        # Identify potential data gaps
        data_gaps = await self._identify_data_gaps(research_plan, sources)
        analysis["data_gaps"] = data_gaps

        return analysis

    async def _extract_content_themes(self, content: str) -> Dict[str, Any]:
        """Extract major themes from research content using fast content processing LLM"""

        try:
            # Limit content size for LLM processing
            content_sample = content[:4000] if len(content) > 4000 else content

            theme_prompt = f"""
            Analyze this research content and identify the main themes and topics:
            
            Content: {content_sample}
            
            Please identify:
            1. Top 5 most prominent themes
            2. Key concepts and terminology
            3. Main arguments or viewpoints presented
            4. Statistical or factual claims
            
            Respond in JSON format:
            {{
                "main_themes": ["theme1", "theme2", ...],
                "key_concepts": ["concept1", "concept2", ...],
                "main_arguments": ["argument1", "argument2", ...],
                "statistical_claims": ["claim1", "claim2", ...]
            }}
            """

            # Use content_llm (granite3.3:8b) for fast theme extraction
            response = self.content_llm.invoke(theme_prompt)
            response_text = response.content if hasattr(response, "content") else str(response)

            # Ensure we have a string
            if isinstance(response_text, list):
                response_text = str(response_text[0]) if response_text else ""
            elif not isinstance(response_text, str):
                response_text = str(response_text)

            try:
                themes = json.loads(response_text)
                return themes
            except json.JSONDecodeError:
                return {"raw_analysis": response_text}

        except Exception as e:
            console.print(f"[yellow]⚠️ Theme extraction failed: {e}[/yellow]")
            return {"error": str(e)}

    async def _identify_key_findings(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify key findings from research sources"""

        findings = []

        try:
            # Prepare content for analysis
            source_summaries = []
            for i, source in enumerate(sources[:10]):  # Limit to top 10 sources
                content = source.get("content", "")[:1000]  # Limit content
                summary = {
                    "source_id": i,
                    "title": source.get("title", ""),
                    "domain": source.get("domain", ""),
                    "content_preview": content,
                    "credibility": source.get("credibility_score", 0.5),
                }
                source_summaries.append(summary)

            findings_prompt = f"""
            Analyze these research sources and identify the most significant findings:
            
            Sources: {json.dumps(source_summaries, indent=2)[:3000]}
            
            Please identify:
            1. Most important factual findings
            2. Statistical insights or data points
            3. Expert opinions or conclusions
            4. Trends or patterns identified
            5. Contradictions or conflicting information
            
            For each finding, provide:
            - finding_type: "fact" | "statistic" | "opinion" | "trend" | "contradiction"
            - description: detailed description
            - source_references: list of source IDs
            - confidence_level: "high" | "medium" | "low"
            - significance: "critical" | "important" | "moderate"
            
            Return as JSON array of findings.
            """

            response = self.llm.invoke(findings_prompt)
            response_text = response.content if hasattr(response, "content") else str(response)

            # Ensure string format
            if isinstance(response_text, list):
                response_text = str(response_text[0]) if response_text else ""
            elif not isinstance(response_text, str):
                response_text = str(response_text)

            try:
                findings = json.loads(response_text)
                if not isinstance(findings, list):
                    findings = [findings]  # Wrap single finding in list
                return findings[:15]  # Limit findings
            except json.JSONDecodeError:
                return [{"raw_analysis": response_text, "type": "analysis_error"}]

        except Exception as e:
            console.print(f"[yellow]⚠️ Key findings identification failed: {e}[/yellow]")
            return [{"error": str(e), "type": "processing_error"}]

    async def _identify_data_gaps(
        self, research_plan: Dict[str, Any], sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify potential gaps in research coverage"""

        gaps = []

        # Check against research plan objectives
        planned_topics = research_plan.get("search_strategies", {}).get("primary_terms", [])
        covered_content = " ".join([s.get("content", "") for s in sources]).lower()

        for topic in planned_topics:
            if topic.lower() not in covered_content:
                gaps.append(
                    {
                        "gap_type": "topic_coverage",
                        "description": f"Limited coverage of planned topic: {topic}",
                        "severity": "medium",
                    }
                )

        # Check source diversity
        source_types = [s.get("source_type", "unknown") for s in sources]
        type_counts = Counter(source_types)

        if len(type_counts) < 3:
            gaps.append(
                {
                    "gap_type": "source_diversity",
                    "description": "Limited source type diversity",
                    "severity": "low",
                }
            )

        # Check temporal coverage
        dates = []
        for source in sources:
            pub_date = source.get("metadata", {}).get("publication_date")
            if pub_date:
                dates.append(pub_date)

        if len(dates) < len(sources) * 0.5:
            gaps.append(
                {
                    "gap_type": "temporal_information",
                    "description": "Many sources lack publication dates",
                    "severity": "medium",
                }
            )

        return gaps

    async def _generate_report_sections(
        self,
        research_results: Dict[str, Any],
        research_plan: Dict[str, Any],
        analysis: Dict[str, Any],
        report_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate all report sections"""

        sections = {}

        # Introduction
        sections["introduction"] = await self._generate_introduction(
            research_results, research_plan
        )

        # Methodology
        sections["methodology"] = await self._generate_methodology_section(
            research_plan, research_results
        )

        # Findings
        sections["findings"] = await self._generate_findings_section(analysis, research_results)

        # Analysis
        sections["analysis"] = await self._generate_analysis_section(analysis, research_results)

        # Recommendations
        sections["recommendations"] = await self._generate_recommendations_section(
            analysis, research_results, research_plan
        )

        # Appendices
        sections["appendices"] = await self._generate_appendices(research_results, analysis)

        return sections

    async def _generate_introduction(
        self, research_results: Dict[str, Any], research_plan: Dict[str, Any]
    ) -> str:
        """Generate introduction section with enhanced content"""

        try:
            sources = research_results.get("sources", [])
            sources_count = len(sources)

            intro_prompt = f"""
            Write a comprehensive introduction for a research report. Use specific details, not templates.
            
            Research Query: "{research_results['query']}"
            Sources Analyzed: {sources_count}
            Research Depth: {research_plan.get('research_depth', 'medium')}
            
            Requirements:
            1. Start with the specific research question and why it matters
            2. Provide concrete background context (no generic statements)
            3. State clear, specific objectives (not template objectives)
            4. Briefly describe the methodology used
            5. Preview the report structure
            
            Write 350-400 words of substantial, specific content. Avoid phrases like "this report examines" or "the following sections." Be direct and informative.
            """

            response = self.llm.invoke(intro_prompt)
            response_text = self._extract_content(response)

            # Clean and validate
            cleaned_content = self._remove_template_artifacts(response_text)

            # If content is too generic or short, try again with more specific prompt
            if len(cleaned_content.strip()) < 200 or self._is_too_generic(cleaned_content):
                specific_prompt = f"""
                Create a specific introduction for research on: "{research_results['query']}"
                
                Include specific context about why this topic is important, what gap in knowledge exists, and what specific insights this research provides. Use concrete language and avoid generic academic phrases.
                
                Write 300+ words with specific details, not templates.
                """
                response = self.llm.invoke(specific_prompt)
                response_text = self._extract_content(response)
                cleaned_content = self._remove_template_artifacts(response_text)

            return (
                cleaned_content
                if len(cleaned_content.strip()) > 100
                else "Introduction content needs manual review."
            )

        except Exception as e:
            console.print(f"[red]Error generating introduction: {e}[/red]")
            return f"Error generating introduction. Please review research data and regenerate."

    def _extract_content(self, response) -> str:
        """Extract content from LLM response consistently"""
        if hasattr(response, "content"):
            content = response.content
        else:
            content = str(response)

        if isinstance(content, list):
            content = str(content[0]) if content else ""

        return str(content)

    def _is_too_generic(self, content: str) -> bool:
        """Check if content is too generic or template-like"""
        generic_phrases = [
            "this report examines",
            "this study explores",
            "the following sections",
            "comprehensive analysis",
            "detailed examination",
            "thorough investigation",
        ]

        content_lower = content.lower()
        generic_count = sum(1 for phrase in generic_phrases if phrase in content_lower)

        return generic_count > 2

    async def _generate_methodology_section(
        self, research_plan: Dict[str, Any], research_results: Dict[str, Any]
    ) -> str:
        """Generate enhanced methodology section"""

        try:
            sources = research_results.get("sources", [])
            search_strategies = research_plan.get("search_strategies", {})
            quality_criteria = research_plan.get("quality_criteria", {})

            # Extract specific details for the methodology
            methodology_data = {
                "sources_count": len(sources),
                "search_terms": search_strategies.get("primary_terms", []),
                "quality_requirements": quality_criteria.get("content_quality", {}),
                "research_phases": len(research_results.get("phase_results", [])),
                "credibility_threshold": quality_criteria.get("source_credibility", {}).get(
                    "minimum_score", "Not specified"
                ),
            }

            methodology_prompt = f"""
            Write a detailed methodology section describing how this research was conducted:
            
            Research Details:
            - {methodology_data['sources_count']} sources analyzed
            - Search terms: {methodology_data['search_terms'][:5]}
            - Research conducted in {methodology_data['research_phases']} phases
            - Quality criteria applied: {methodology_data['quality_requirements']}
            
            Create 6 specific subsections:
            1. Research Design and Approach - Describe web-based research with automation
            2. Data Collection Methods - Detail how sources were gathered
            3. Search Strategies Employed - Explain search term selection and refinement
            4. Quality Assessment Criteria - Define credibility and relevance standards
            5. Analysis Procedures - Describe how data was processed and synthesized
            6. Limitations and Considerations - Address potential biases and constraints
            
            Write 300-400 words with specific details about this research process.
            """

            response = self.llm.invoke(methodology_prompt)
            response_text = self._extract_content(response)
            cleaned_content = self._remove_template_artifacts(response_text)

            return (
                cleaned_content
                if len(cleaned_content.strip()) > 150
                else "Methodology section requires manual review."
            )

        except Exception as e:
            console.print(f"[red]Error generating methodology: {e}[/red]")
            return "Error generating methodology section. Please review research parameters."

    async def _generate_findings_section(
        self, analysis: Dict[str, Any], research_results: Dict[str, Any]
    ) -> str:
        """Generate enhanced findings section with specific discoveries"""

        try:
            key_findings = analysis.get("key_findings", [])[:8]  # Limit to top findings
            content_themes = analysis.get("content_themes", {})
            sources = research_results.get("sources", [])
            quality_metrics = research_results.get("quality_metrics", {})

            # Prepare specific data for findings
            findings_data = {
                "findings_count": len(key_findings),
                "themes": content_themes.get("main_themes", [])[:5],
                "high_credibility_sources": len(
                    [s for s in sources if s.get("credibility_score", 0) > 0.7]
                ),
                "total_sources": len(sources),
                "overall_quality": quality_metrics.get("overall_score", "Not calculated"),
            }

            findings_prompt = f"""
            Write a comprehensive findings section presenting these research discoveries:
            
            Research Results:
            - {findings_data['findings_count']} key findings identified
            - {findings_data['high_credibility_sources']} high-credibility sources (out of {findings_data['total_sources']})
            - Main themes: {findings_data['themes']}
            - Overall quality score: {findings_data['overall_quality']}
            
            Key Findings to Present:
            {json.dumps(key_findings, indent=2)[:1500]}
            
            Structure your findings section with:
            1. Introduction stating scope of findings
            2. Major discoveries organized by themes
            3. Statistical highlights and data points
            4. Expert conclusions from credible sources  
            5. Patterns and trends identified
            6. Any contradictions or conflicting information found
            
            Write 400-600 words with specific evidence and source references. Use concrete data, not generic statements.
            """

            response = self.llm.invoke(findings_prompt)
            response_text = self._extract_content(response)
            cleaned_content = self._remove_template_artifacts(response_text)

            # Validate findings quality
            if len(cleaned_content.strip()) < 200 or cleaned_content.count("finding") > 10:
                # Try more specific approach
                specific_prompt = f"""
                Present the actual research findings discovered:
                
                Research Topic: {research_results.get('query', 'Unknown')}
                Key Discoveries: {json.dumps(key_findings[:3], indent=2)}
                
                Write specific findings with evidence, statistics, and source credibility notes. Avoid generic language.
                """
                response = self.llm.invoke(specific_prompt)
                response_text = self._extract_content(response)
                cleaned_content = self._remove_template_artifacts(response_text)

            return (
                cleaned_content
                if len(cleaned_content.strip()) > 100
                else "Findings section needs manual review with source data."
            )

        except Exception as e:
            console.print(f"[red]Error generating findings: {e}[/red]")
            return "Error generating findings section. Please review analysis data."

    async def _generate_analysis_section(
        self, analysis: Dict[str, Any], research_results: Dict[str, Any]
    ) -> str:
        """Generate enhanced analysis section with critical evaluation"""

        try:
            key_findings = analysis.get("key_findings", [])[:6]
            content_themes = analysis.get("content_themes", {})
            quality_metrics = research_results.get("quality_metrics", {})
            sources = research_results.get("sources", [])

            # Prepare analysis data
            analysis_data = {
                "high_quality_sources": len(
                    [s for s in sources if s.get("credibility_score", 0) > 0.75]
                ),
                "source_diversity": quality_metrics.get("source_diversity", 0),
                "overall_quality": quality_metrics.get("overall_score", 0),
                "themes_identified": len(content_themes.get("main_themes", [])),
                "contradictions": [
                    f for f in key_findings if f.get("finding_type") == "contradiction"
                ],
            }

            analysis_prompt = f"""
            Write a thorough analysis section interpreting these research findings:
            
            Research Topic: {research_results.get('query', 'Unknown')}
            Analysis Context:
            - {analysis_data['high_quality_sources']} high-quality sources analyzed
            - Source diversity score: {analysis_data['source_diversity']}
            - {analysis_data['themes_identified']} main themes identified
            - {len(analysis_data['contradictions'])} contradictions found
            
            Key Findings to Analyze:
            {json.dumps(key_findings, indent=2)[:1000]}
            
            Create 5 analysis subsections:
            1. **Interpretation of Key Findings** - What the findings mean and their significance
            2. **Connections Between Themes** - How different findings relate to each other
            3. **Reliability and Validity Assessment** - Evaluate source credibility and data quality
            4. **Implications and Broader Context** - What these findings mean in the bigger picture  
            5. **Limitations and Gaps** - What's missing and potential biases
            
            Write 400-600 words of critical analysis with specific references to the data. Be analytical, not descriptive.
            """

            response = self.llm.invoke(analysis_prompt)
            response_text = self._extract_content(response)
            cleaned_content = self._remove_template_artifacts(response_text)

            return (
                cleaned_content
                if len(cleaned_content.strip()) > 200
                else "Analysis section needs manual review."
            )

        except Exception as e:
            console.print(f"[red]Error generating analysis: {e}[/red]")
            return "Error generating analysis section. Please review research findings."

    async def _generate_recommendations_section(
        self,
        analysis: Dict[str, Any],
        research_results: Dict[str, Any],
        research_plan: Dict[str, Any],
    ) -> str:
        """Generate enhanced recommendations with actionable advice"""

        try:
            key_findings = analysis.get("key_findings", [])[:5]
            sources_count = len(research_results.get("sources", []))
            query = research_results.get("query", "")

            # Extract actionable elements from findings
            actionable_items = []
            for finding in key_findings:
                if finding.get("significance") in ["critical", "important"]:
                    actionable_items.append(
                        {
                            "finding": finding.get("description", ""),
                            "type": finding.get("finding_type", ""),
                            "confidence": finding.get("confidence_level", ""),
                        }
                    )

            recommendations_prompt = f"""
            Based on research findings, write specific actionable recommendations:
            
            Research Question: "{query}"
            Sources Analyzed: {sources_count}
            
            Critical Findings for Recommendations:
            {json.dumps(actionable_items, indent=2)[:1500]}
            
            Create recommendations in these categories:
            1. **Immediate Actions** - Based on high-confidence findings
            2. **Strategic Improvements** - Medium-term recommendations  
            3. **Further Investigation** - Areas needing more research
            4. **Implementation Considerations** - How to apply these recommendations
            5. **Priority Ranking** - Order by importance and feasibility
            
            Write 300-450 words with specific, implementable recommendations. Include timelines and success metrics where possible.
            """

            response = self.llm.invoke(recommendations_prompt)
            response_text = self._extract_content(response)
            cleaned_content = self._remove_template_artifacts(response_text)

            # Validate recommendations quality
            if len(cleaned_content.strip()) < 150 or not self._contains_actionable_language(
                cleaned_content
            ):
                specific_prompt = f"""
                Create specific recommendations to address: "{query}"
                
                Based on the research findings, what should be done? Give concrete steps, not generic advice.
                Include who should act, what they should do, and when.
                """
                response = self.llm.invoke(specific_prompt)
                response_text = self._extract_content(response)
                cleaned_content = self._remove_template_artifacts(response_text)

            return (
                cleaned_content
                if len(cleaned_content.strip()) > 100
                else "Recommendations section needs manual review."
            )

        except Exception as e:
            console.print(f"[red]Error generating recommendations: {e}[/red]")
            return "Error generating recommendations. Please review analysis and findings."

    def _contains_actionable_language(self, content: str) -> bool:
        """Check if content contains actionable language"""
        action_words = [
            "implement",
            "develop",
            "create",
            "establish",
            "conduct",
            "should",
            "recommend",
            "propose",
            "suggest",
            "consider",
            "prioritize",
            "focus on",
            "invest in",
            "address",
            "improve",
        ]

        content_lower = content.lower()
        action_count = sum(1 for word in action_words if word in content_lower)

        return action_count >= 3

    async def _generate_executive_summary(
        self, research_results: Dict[str, Any], analysis: Dict[str, Any], sections: Dict[str, Any]
    ) -> str:
        """Generate enhanced executive summary for decision makers"""

        try:
            key_findings = analysis.get("key_findings", [])[:4]  # Top 4 findings
            themes = analysis.get("content_themes", {}).get("main_themes", [])[:4]
            quality_score = research_results.get("quality_metrics", {}).get("overall_score", 0)
            sources_count = len(research_results.get("sources", []))

            # Extract actionable insights
            critical_findings = [f for f in key_findings if f.get("significance") == "critical"][:3]

            summary_prompt = f"""
            Write a compelling executive summary for busy decision-makers:
            
            Research Topic: "{research_results.get('query', '')}"
            Research Scope: {sources_count} sources analyzed, quality score: {quality_score:.2f}
            
            Critical Findings (most important results):
            {json.dumps(critical_findings, indent=2)[:800]}
            
            Key Themes Discovered:
            {themes}
            
            Write an executive summary that:
            1. Opens with the specific research question and why it matters
            2. States 3-4 most important findings with supporting evidence
            3. Draws clear conclusions about implications
            4. Provides 2-3 top-priority actionable recommendations
            5. Notes key limitations that affect decision-making
            
            Write 250-300 words in clear, business-focused language. Use bullet points for findings and recommendations.
            """

            response = self.llm.invoke(summary_prompt)
            response_text = self._extract_content(response)
            cleaned_content = self._remove_template_artifacts(response_text)

            # Validate summary quality
            if len(cleaned_content.strip()) < 150 or not self._contains_decision_language(
                cleaned_content
            ):
                focused_prompt = f"""
                Create an executive summary answering: "{research_results.get('query', '')}"
                
                Key results: {json.dumps(critical_findings[:2], indent=2)}
                
                What should executives know and do based on this research? Write concisely with specific recommendations.
                """
                response = self.llm.invoke(focused_prompt)
                response_text = self._extract_content(response)
                cleaned_content = self._remove_template_artifacts(response_text)

            return (
                cleaned_content
                if len(cleaned_content.strip()) > 100
                else "Executive summary needs manual review."
            )

        except Exception as e:
            console.print(f"[red]Error generating executive summary: {e}[/red]")
            return "Error generating executive summary. Please review research findings."

    def _contains_decision_language(self, content: str) -> bool:
        """Check if content contains decision-oriented language"""
        decision_words = [
            "recommend",
            "should",
            "must",
            "critical",
            "important",
            "action",
            "implement",
            "decision",
            "priority",
            "impact",
            "result",
            "outcome",
            "benefit",
            "risk",
            "opportunity",
        ]

        content_lower = content.lower()
        decision_count = sum(1 for word in decision_words if word in content_lower)

        return decision_count >= 4

    async def _generate_appendices(
        self, research_results: Dict[str, Any], analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate appendices with supporting information"""

        appendices = {}

        # Source bibliography
        sources = research_results.get("sources", [])
        bibliography = []

        for i, source in enumerate(sources, 1):
            bib_entry = {
                "id": i,
                "title": source.get("title", "Unknown Title"),
                "url": source.get("url", ""),
                "domain": source.get("domain", ""),
                "access_date": source.get("extracted_at", ""),
                "credibility_score": source.get("credibility_score", 0.0),
                "word_count": source.get("word_count", 0),
            }
            bibliography.append(bib_entry)

        appendices["source_bibliography"] = bibliography

        # Research statistics
        appendices["research_statistics"] = {
            "total_sources": len(sources),
            "quality_metrics": research_results.get("quality_metrics", {}),
            "source_analysis": analysis.get("source_analysis", {}),
            "credibility_distribution": analysis.get("credibility_distribution", {}),
        }

        # Search terms and strategies
        appendices["methodology_details"] = {
            "phase_results": research_results.get("phase_results", []),
            "failed_urls_count": len(research_results.get("failed_urls", [])),
            "research_duration": self._calculate_research_duration(research_results),
        }

        return appendices

    async def _generate_formatted_outputs(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate formatted versions of the report"""

        outputs = {}

        # Markdown format
        markdown_content = self._format_as_markdown(report_data)
        outputs["markdown"] = markdown_content

        # Plain text format
        text_content = self._format_as_text(report_data)
        outputs["text"] = text_content

        # Structured JSON (already available in report_data)
        outputs["json"] = report_data

        return outputs

    def _format_as_markdown(self, report_data: Dict[str, Any]) -> str:
        """Format report as enhanced Markdown with quality indicators"""

        sections = report_data.get("sections", {})
        metadata = report_data.get("metadata", {})
        quality = report_data.get("quality_assessment", {})

        # Quality indicators
        quality_score = quality.get("overall_score", 0)
        quality_badge = (
            "🟢 High"
            if quality_score >= 8
            else "🟡 Medium" if quality_score >= 6 else "🔴 Needs Review"
        )

        markdown = f"""# Research Report: {report_data.get('query', 'Research Query')}

> **Quality Assessment:** {quality_badge} (Score: {quality_score:.1f}/10)
> **Report Generated:** {report_data.get('generated_at', 'Unknown Date')}
> **Sources Analyzed:** {metadata.get('total_sources', 0)} sources

---

## 📋 Executive Summary

{self._clean_section_content(sections.get('executive_summary', 'Executive summary pending review.'))}

---

## 🎯 Introduction

{self._clean_section_content(sections.get('introduction', 'Introduction content not available.'))}

---

## 🔬 Research Methodology

{self._clean_section_content(sections.get('methodology', 'Methodology details not available.'))}

---

## 📊 Key Findings

{self._clean_section_content(sections.get('findings', 'Research findings not available.'))}

---

## 📈 Analysis & Insights

{self._clean_section_content(sections.get('analysis', 'Analysis content not available.'))}

---

## 🎯 Recommendations

{self._clean_section_content(sections.get('recommendations', 'Recommendations not available.'))}

---

## 📚 Source Bibliography

"""

        # Enhanced bibliography with quality indicators
        appendices = sections.get("appendices", {})
        bibliography = appendices.get("source_bibliography", [])

        if bibliography:
            # Group sources by quality
            high_quality = [s for s in bibliography if s.get("credibility_score", 0) >= 0.8]
            medium_quality = [s for s in bibliography if 0.6 <= s.get("credibility_score", 0) < 0.8]
            lower_quality = [s for s in bibliography if s.get("credibility_score", 0) < 0.6]

            if high_quality:
                markdown += "\n### 🟢 High Quality Sources\n\n"
                for source in high_quality[:10]:
                    markdown += f"{source['id']}. **{source['title']}**  \n"
                    markdown += f"   📍 {source.get('domain', 'Unknown')} | 📊 Credibility: {source['credibility_score']:.2f}  \n"
                    markdown += (
                        f"   🔗 [{source.get('url', 'N/A')}]({source.get('url', '#')})  \n\n"
                    )

            if medium_quality:
                markdown += "\n### 🟡 Standard Quality Sources\n\n"
                for source in medium_quality[:8]:
                    markdown += f"{source['id']}. {source['title']} (Score: {source['credibility_score']:.2f})  \n"
                    markdown += f"   🔗 {source.get('url', 'N/A')}  \n\n"

            if lower_quality and len(high_quality) + len(medium_quality) < 15:
                markdown += "\n### 🔴 Additional Sources\n\n"
                for source in lower_quality[:5]:
                    markdown += f"{source['id']}. {source['title']}  \n"
        else:
            markdown += "\nNo bibliography available.\n"

        # Add research statistics
        stats = appendices.get("research_statistics", {})
        if stats:
            markdown += f"\n---\n\n## 📊 Research Statistics\n\n"
            markdown += f"- **Total Sources Processed:** {stats.get('total_sources', 0)}\n"
            if stats.get("quality_metrics"):
                qm = stats["quality_metrics"]
                markdown += f"- **Average Source Quality:** {qm.get('avg_credibility', 0):.2f}\n"
                markdown += f"- **Content Coverage:** {qm.get('content_coverage', 0):.1%}\n"

        return markdown

    def _clean_section_content(self, content: str) -> str:
        """Clean section content for final formatting"""
        if not content or content.strip() == "":
            return "_Content not available._"

        # Remove any remaining template artifacts
        cleaned = self._remove_template_artifacts(content)

        # Ensure content doesn't end abruptly
        if cleaned and not cleaned.rstrip().endswith((".", "!", "?", ":")):
            cleaned = cleaned.rstrip() + "."

        return cleaned if cleaned.strip() else "_Content needs review._"

    def _format_as_text(self, report_data: Dict[str, Any]) -> str:
        """Format report as clean, readable plain text"""

        sections = report_data.get("sections", {})
        metadata = report_data.get("metadata", {})
        quality = report_data.get("quality_assessment", {})

        # Quality indicator
        quality_score = quality.get("overall_score", 0)
        quality_text = (
            "HIGH QUALITY"
            if quality_score >= 8
            else "MEDIUM QUALITY" if quality_score >= 6 else "NEEDS REVIEW"
        )

        text = f"""RESEARCH REPORT: {report_data.get('query', 'Research Query').upper()}

QUALITY ASSESSMENT: {quality_text} (Score: {quality_score:.1f}/10)
REPORT GENERATED: {report_data.get('generated_at', 'Unknown Date')}
SOURCES ANALYZED: {metadata.get('total_sources', 0)} sources

{'='*80}

EXECUTIVE SUMMARY

{self._clean_section_for_text(sections.get('executive_summary', 'Executive summary pending review.'))}

{'='*80}

INTRODUCTION

{self._clean_section_for_text(sections.get('introduction', 'Introduction content not available.'))}

{'='*80}

RESEARCH METHODOLOGY

{self._clean_section_for_text(sections.get('methodology', 'Methodology details not available.'))}

{'='*80}

KEY FINDINGS

{self._clean_section_for_text(sections.get('findings', 'Research findings not available.'))}

{'='*80}

ANALYSIS & INSIGHTS

{self._clean_section_for_text(sections.get('analysis', 'Analysis content not available.'))}

{'='*80}

RECOMMENDATIONS

{self._clean_section_for_text(sections.get('recommendations', 'Recommendations not available.'))}

{'='*80}

SOURCE BIBLIOGRAPHY

"""

        # Add clean bibliography
        appendices = sections.get("appendices", {})
        bibliography = appendices.get("source_bibliography", [])

        if bibliography:
            # Prioritize high-quality sources
            sorted_sources = sorted(
                bibliography, key=lambda x: x.get("credibility_score", 0), reverse=True
            )

            for i, source in enumerate(sorted_sources[:15], 1):
                text += f"{i}. {source.get('title', 'Untitled')}\n"
                text += f"   URL: {source.get('url', 'N/A')}\n"
                text += f"   Domain: {source.get('domain', 'Unknown')} | "
                text += f"Quality Score: {source.get('credibility_score', 0):.2f}\n\n"
        else:
            text += "No bibliography available.\n"

        # Add research statistics
        stats = appendices.get("research_statistics", {})
        if stats:
            text += f"{'='*80}\n\n"
            text += f"RESEARCH STATISTICS\n\n"
            text += f"Total Sources Processed: {stats.get('total_sources', 0)}\n"

            if stats.get("quality_metrics"):
                qm = stats["quality_metrics"]
                text += f"Average Source Quality: {qm.get('avg_credibility', 0):.2f}\n"
                text += f"Content Coverage: {qm.get('content_coverage', 0):.1%}\n"

        return text

    def _clean_section_for_text(self, content: str) -> str:
        """Clean section content specifically for text format"""
        if not content or content.strip() == "":
            return "Content not available."

        # Remove template artifacts and clean content
        cleaned = self._remove_template_artifacts(content)

        # Remove markdown formatting for plain text
        cleaned = cleaned.replace("**", "").replace("*", "")
        cleaned = cleaned.replace("##", "").replace("#", "")

        # Ensure proper ending
        if cleaned and not cleaned.rstrip().endswith((".", "!", "?", ":")):
            cleaned = cleaned.rstrip() + "."

        return cleaned if cleaned.strip() else "Content needs review."

    def _calculate_research_duration(self, research_results: Dict[str, Any]) -> str:
        """Calculate research duration"""

        started_at = research_results.get("started_at")
        completed_at = research_results.get("completed_at")

        if not started_at or not completed_at:
            return "Unknown"

        try:
            start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            end = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
            duration = end - start

            total_seconds = int(duration.total_seconds())
            minutes = total_seconds // 60
            seconds = total_seconds % 60

            return f"{minutes}m {seconds}s"
        except Exception:
            return "Unknown"

    def _estimate_total_word_count(self, research_results: Dict[str, Any]) -> int:
        """Estimate total word count from sources"""

        sources = research_results.get("sources", [])
        return sum(s.get("word_count", 0) for s in sources)

    async def _clean_and_validate_content(self, sections: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate report content to remove artifacts and improve quality"""

        cleaned_sections = {}

        for section_name, section_content in sections.items():
            if not section_content:
                continue

            # Convert to string if not already
            content = str(section_content)

            # Clean common artifacts
            content = self._remove_template_artifacts(content)
            content = self._remove_debug_content(content)
            content = self._improve_formatting(content)

            # Validate content quality
            if self._validate_section_content(content):
                cleaned_sections[section_name] = content
            else:
                console.print(
                    f"[yellow]⚠️ Section '{section_name}' failed validation, regenerating...[/yellow]"
                )
                # Keep original but mark for improvement
                cleaned_sections[section_name] = content

        return cleaned_sections

    def _remove_template_artifacts(self, content: str) -> str:
        """Remove template artifacts and placeholder content"""

        # Remove <think> blocks
        import re

        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

        # Remove placeholder patterns
        content = re.sub(r"\[placeholder[^\]]*\]", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\{[^}]*placeholder[^}]*\}", "", content, flags=re.IGNORECASE)

        # Remove template markers
        content = re.sub(r"---\s*template\s*---", "", content, flags=re.IGNORECASE)
        content = re.sub(r"template:\s*[^\n]*", "", content, flags=re.IGNORECASE)

        # Remove "Note:" prefixes for placeholders
        content = re.sub(r"Note:\s*The actual content.*?[\.\n]", "", content, flags=re.IGNORECASE)

        # Remove boxed placeholder content
        content = re.sub(r"\\boxed\{[^}]*\}", "", content)

        return content.strip()

    def _remove_debug_content(self, content: str) -> str:
        """Remove debugging and meta-content"""

        # Remove word count mentions
        import re

        content = re.sub(r"\(word count[^)]*\)", "", content, flags=re.IGNORECASE)
        content = re.sub(r"word count estimate[^\n]*", "", content, flags=re.IGNORECASE)

        # Remove meta commentary about the content
        content = re.sub(
            r"this (summary|section|analysis)[^\.]*provides[^\.]*\.",
            "",
            content,
            flags=re.IGNORECASE,
        )
        content = re.sub(
            r"the above (summary|section|analysis)[^\.]*\.", "", content, flags=re.IGNORECASE
        )

        # Remove instruction artifacts
        content = re.sub(r"final (version|draft)[^\.]*:", "", content, flags=re.IGNORECASE)

        return content.strip()

    def _improve_formatting(self, content: str) -> str:
        """Improve content formatting and readability"""

        # Clean up excessive whitespace
        import re

        content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)
        content = re.sub(r" +", " ", content)

        # Fix common formatting issues
        content = re.sub(r"([\.!?])\s*([A-Z])", r"\1 \2", content)

        return content.strip()

    def _validate_section_content(self, content: str) -> bool:
        """Validate section content quality"""

        if not content or len(content.strip()) < 50:
            return False

        # Check for too many placeholder patterns
        import re

        placeholder_count = len(re.findall(r"\[.*?\]|\{.*?\}", content))
        if placeholder_count > 3:
            return False

        # Check for substantial content
        sentences = content.split(".")
        meaningful_sentences = [s for s in sentences if len(s.strip()) > 20]

        return len(meaningful_sentences) >= 2

    async def _assess_report_quality_enhanced(
        self, research_results: Dict[str, Any], sections: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enhanced quality assessment using dual-model approach"""

        sources = research_results.get("sources", [])
        quality_metrics = research_results.get("quality_metrics", {})

        # Calculate section completeness
        required_sections = [
            "introduction",
            "methodology",
            "findings",
            "analysis",
            "recommendations",
        ]
        completed_sections = sum(
            1
            for section in required_sections
            if sections.get(section) and len(str(sections[section]).strip()) > 100
        )
        section_completeness = completed_sections / len(required_sections)

        # Use content_llm (granite3.3:8b) for quick content assessments
        content_assessments = []
        for section_name, section_content in sections.items():
            if section_content and len(str(section_content).strip()) > 50:
                assessment = await self._quick_content_assessment(str(section_content))
                content_assessments.append(assessment.get("overall_score", 0.0))

        avg_content_quality = (
            sum(content_assessments) / len(content_assessments) if content_assessments else 0
        )

        # Calculate source utilization
        total_sources = len(sources)
        source_utilization = min(total_sources / 5, 1.0)  # Target: at least 5 sources

        # Calculate specificity using fast model
        specificity_score = await self._calculate_specificity_score_fast(sections)

        # Overall score with enhanced weighting
        overall_score = (
            section_completeness * 0.25
            + avg_content_quality * 0.35
            + source_utilization * 0.2
            + specificity_score * 0.2
        )

        return {
            "overall_score": round(overall_score, 3),
            "section_completeness": round(section_completeness, 3),
            "content_quality": round(avg_content_quality, 3),
            "source_utilization": round(source_utilization, 3),
            "specificity_score": round(specificity_score, 3),
            "source_count": total_sources,
            "assessment_timestamp": datetime.now().isoformat(),
            "content_assessments_count": len(content_assessments),
        }

    async def _calculate_specificity_score_fast(self, sections: Dict[str, Any]) -> float:
        """Calculate specificity score using granite3.3:8b for fast processing"""

        try:
            # Combine first 2000 chars from each section for analysis
            combined_content = ""
            for section_name, section_content in sections.items():
                content_str = str(section_content)
                if content_str:
                    combined_content += f"{section_name}: {content_str[:800]}...\n\n"

            if not combined_content.strip():
                return 0.0

            # Limit total content for fast processing
            analysis_content = combined_content[:4000]

            specificity_prompt = f"""
            Rate the specificity of this content (1-10 scale):
            - High specificity (8-10): Concrete data, specific examples, exact numbers, named entities
            - Medium specificity (5-7): Some specific details, general examples, approximate data
            - Low specificity (1-4): Generic language, vague terms, no specific details
            
            Content to analyze:
            {analysis_content}
            
            Return just a number (1-10) representing the specificity score.
            """

            response = self.content_llm.invoke(specificity_prompt)
            response_text = response.content if hasattr(response, "content") else str(response)

            if isinstance(response_text, list):
                response_text = str(response_text[0]) if response_text else ""
            elif not isinstance(response_text, str):
                response_text = str(response_text)

            # Extract numeric score
            import re

            score_match = re.search(r"\b([1-9]|10)(?:\.\d+)?\b", response_text)
            if score_match:
                score = float(score_match.group(1))
                return min(score / 10.0, 1.0)
            else:
                # Fallback to rule-based calculation
                return self._calculate_specificity_score(sections)

        except Exception as e:
            console.print(
                f"[yellow]⚠️ Fast specificity calculation failed, using fallback: {e}[/yellow]"
            )
            return self._calculate_specificity_score(sections)

    def _score_content_quality(self, content: str) -> float:
        """Score individual content quality"""

        if not content or len(content.strip()) < 50:
            return 0.0

        score = 0.0

        # Length bonus (but not excessive)
        length_score = min(len(content) / 500, 1.0)
        score += length_score * 0.3

        # Structure bonus (lists, paragraphs)
        import re

        if re.search(r"^\d+\.|\*|-", content, re.MULTILINE):
            score += 0.2
        if content.count("\n\n") > 0:
            score += 0.1

        # Specificity bonus (numbers, proper nouns, specific terms)
        if re.search(r"\d+%|\$\d+|\d+ (percent|million|billion|thousand)", content):
            score += 0.2
        if re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", content):  # Proper nouns
            score += 0.1

        # Penalty for template language
        template_patterns = [
            r"this (section|analysis) (provides|covers|examines)",
            r"the following (sections?|points?|items?)",
            r"as (mentioned|noted|discussed) (above|previously|earlier)",
        ]
        for pattern in template_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                score -= 0.1

        return max(0.0, min(score, 1.0))

    def _calculate_specificity_score(self, sections: Dict[str, Any]) -> float:
        """Calculate how specific (vs generic) the content is"""

        total_content = " ".join(str(section) for section in sections.values())

        if not total_content:
            return 0.0

        # Count specific indicators
        import re

        specific_indicators = 0

        # Numbers and data points
        specific_indicators += len(re.findall(r"\d+(?:\.\d+)?%", total_content))
        specific_indicators += len(re.findall(r"\$\d+(?:,\d{3})*(?:\.\d+)?", total_content))
        specific_indicators += len(re.findall(r"\d{4}", total_content))  # Years

        # Specific names and organizations
        specific_indicators += len(re.findall(r"[A-Z][a-z]+ [A-Z][a-z]+", total_content))

        # Technical terms and proper nouns
        specific_indicators += len(re.findall(r"\b[A-Z]{2,}\b", total_content))

        # Normalize by content length
        specificity = specific_indicators / (len(total_content.split()) / 100)

        return min(specificity, 1.0)

    async def _improve_low_quality_sections(
        self,
        research_results: Dict[str, Any],
        research_plan: Dict[str, Any],
        analysis: Dict[str, Any],
        sections: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Improve sections using dual-model approach for enhanced quality"""

        improved_sections = sections.copy()

        for section_name, section_content in sections.items():
            # First, quick assessment with granite3.3:8b
            quick_assessment = await self._quick_content_assessment(str(section_content))
            content_score = quick_assessment.get("overall_score", 0.0)

            if content_score < 5.0:  # Low quality threshold (out of 10)
                console.print(
                    f"[yellow]🔧 Improving section: {section_name} (Score: {content_score:.1f})[/yellow]"
                )

                # Try smart improvement with granite3.3:8b first
                improved_content = await self._smart_section_improvement(
                    str(section_content), section_name
                )

                # If still not good enough, use main LLM (magistral) for comprehensive regeneration
                if improved_content == section_content or content_score < 3.0:
                    console.print(
                        f"[cyan]🔄 Using main model for comprehensive regeneration of {section_name}[/cyan]"
                    )
                    comprehensive_content = await self._generate_improved_section(
                        section_name, research_results, research_plan, analysis
                    )

                    if comprehensive_content and len(str(comprehensive_content).strip()) > 100:
                        improved_sections[section_name] = comprehensive_content
                    elif improved_content != section_content:
                        improved_sections[section_name] = improved_content
                else:
                    improved_sections[section_name] = improved_content

        return improved_sections

    async def _generate_improved_section(
        self,
        section_name: str,
        research_results: Dict[str, Any],
        research_plan: Dict[str, Any],
        analysis: Dict[str, Any],
    ) -> str:
        """Generate improved content for a specific section"""

        sources = research_results.get("sources", [])
        query = research_results.get("query", "")

        # Create focused research data for this section
        section_data = self._extract_relevant_data_for_section(section_name, sources, analysis)

        # Enhanced prompts for each section type
        prompt_templates = {
            "executive_summary": f"""
Based on the research findings, write a concise executive summary that:

1. States the research question clearly
2. Highlights 3-5 specific key findings with data/evidence
3. Provides concrete conclusions
4. Lists actionable recommendations
5. Notes significant limitations

Research Query: {query}

Key Data Points:
{section_data}

Write a professional, specific executive summary (200-300 words) without templates or placeholders.
""",
            "introduction": """
Write a comprehensive introduction that:

1. Clearly states the research question and its importance
2. Provides relevant background context
3. Outlines the scope and methodology
4. Previews the report structure

Research Query: {query}
Background Data: {section_data}

Create a detailed introduction (300-400 words) with specific context and clear objectives.
""",
            "findings": """
Present detailed research findings organized by themes:

1. Summarize major discoveries with supporting evidence
2. Include relevant statistics and data points
3. Organize by clear themes/categories
4. Reference source credibility appropriately
5. Highlight patterns and trends

Research Data:
{section_data}

Create comprehensive findings section (400-600 words) with specific insights and evidence.
""",
            "analysis": """
Provide analytical interpretation of findings:

1. Interpret key findings and their significance
2. Draw connections between different themes
3. Assess reliability and validity of sources
4. Discuss broader implications
5. Address limitations and gaps

Research Data:
{section_data}

Write thorough analysis (400-500 words) with critical evaluation and insights.
""",
            "recommendations": """
Develop actionable recommendations based on findings:

1. Provide specific, implementable advice
2. Address the original research question
3. Suggest areas for further investigation
4. Prioritize by importance/feasibility
5. Include implementation considerations

Research Findings:
{section_data}

Create practical recommendations (300-400 words) with clear action items.
""",
        }

        template = prompt_templates.get(section_name, prompt_templates.get("findings", ""))

        if template:
            prompt = template.format(
                query=query, section_data=section_data[:2000]  # Limit data size
            )

            try:
                response = self.llm.invoke(prompt)
                content = response.content if hasattr(response, "content") else str(response)
                return self._remove_template_artifacts(str(content))
            except Exception as e:
                console.print(f"[red]Error improving section {section_name}: {e}[/red]")
                return ""

        return ""

    def _extract_relevant_data_for_section(
        self, section_name: str, sources: List[Dict[str, Any]], analysis: Dict[str, Any]
    ) -> str:
        """Extract relevant data for a specific section"""

        # Combine relevant information
        relevant_data = []

        # Add source summaries
        for source in sources[:5]:  # Top 5 sources
            if source.get("content"):
                relevant_data.append(
                    f"Source: {source.get('title', 'Unknown')} - {source.get('content', '')[:200]}..."
                )

        # Add analysis insights
        if analysis.get("key_themes"):
            relevant_data.append(f"Key Themes: {', '.join(analysis['key_themes'][:5])}")

        if analysis.get("source_quality_analysis"):
            relevant_data.append(f"Source Quality: {analysis['source_quality_analysis']}")

        return "\n".join(relevant_data)

    def _generate_metadata(
        self,
        research_results: Dict[str, Any],
        research_plan: Dict[str, Any],
        report_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate report metadata"""

        sources = research_results.get("sources", [])

        return {
            "total_sources": len(sources),
            "research_depth": research_plan.get("research_depth", "unknown"),
            "research_duration": self._calculate_research_duration(research_results),
            "report_type": report_config.get("report_type", "comprehensive"),
            "word_count_estimate": self._estimate_total_word_count(research_results),
            "source_types": list(set(s.get("source_type", "unknown") for s in sources)),
            "domains_analyzed": list(set(s.get("domain", "unknown") for s in sources))[:10],
        }

    def _assess_report_quality(
        self, research_results: Dict[str, Any], analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess overall report quality"""

        sources = research_results.get("sources", [])
        quality_metrics = research_results.get("quality_metrics", {})

        # Calculate quality score
        source_quality = sum(s.get("credibility_score", 0) for s in sources) / max(len(sources), 1)
        coverage_score = quality_metrics.get("content_coverage", 0.0)
        depth_score = min(len(sources) / 10, 1.0)  # Normalize to 1.0
        analysis_quality = min(len(analysis.get("key_findings", [])) / 5, 1.0)

        overall_score = (
            source_quality * 0.3
            + coverage_score * 0.25
            + depth_score * 0.25
            + analysis_quality * 0.2
        ) * 10

        return {
            "overall_score": min(overall_score, 10.0),
            "source_quality_score": source_quality * 10,
            "coverage_score": coverage_score * 10,
            "depth_score": depth_score * 10,
            "analysis_quality_score": analysis_quality * 10,
            "total_sources": len(sources),
            "high_quality_sources": len(
                [s for s in sources if s.get("credibility_score", 0) >= 0.8]
            ),
        }
