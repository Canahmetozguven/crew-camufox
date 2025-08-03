#!/usr/bin/env python3
"""
Extract Formatted Outputs Utility
Extracts Markdown and Text reports from research mission JSON files
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

# Add the parent directory to the path to import the orchestrator
sys.path.append(str(Path(__file__).parent.parent))

from rich.console import Console

console = Console()


class OutputExtractor:
    """Utility class for extracting formatted outputs from research JSON files"""

    def __init__(self, output_dir: str = "research_outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def extract_from_json_file(self, json_file_path: str) -> bool:
        """
        Extract and save Markdown and Text outputs from a JSON research file

        Args:
            json_file_path: Path to the JSON file containing research results

        Returns:
            bool: True if extraction successful, False otherwise
        """

        try:
            console.print(f"\n[cyan]📄 Processing: {json_file_path}[/cyan]")

            # Load the JSON file with error handling
            try:
                with open(json_file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Try to parse the JSON
                mission_results = json.loads(content)

            except json.JSONDecodeError as e:
                console.print(
                    f"[red]❌ JSON parsing error at line {e.lineno}, column {e.colno}: {e.msg}[/red]"
                )
                console.print(f"[yellow]🔧 Attempting to fix common JSON issues...[/yellow]")

                # Try to fix common JSON issues
                try:
                    # Remove trailing commas and fix incomplete structures
                    fixed_content = self._fix_json_content(content)
                    mission_results = json.loads(fixed_content)
                    console.print(f"[green]✅ JSON successfully repaired and parsed[/green]")
                except Exception as fix_error:
                    console.print(f"[red]❌ Could not repair JSON: {fix_error}[/red]")
                    return False

            # Extract mission ID from the data or filename
            mission_id = mission_results.get("mission_id")
            if not mission_id:
                # Try to extract from filename
                filename = os.path.basename(json_file_path)
                if "_complete.json" in filename:
                    mission_id = filename.replace("_complete.json", "")
                else:
                    mission_id = f"extracted_{int(datetime.now().timestamp())}"

            console.print(f"[yellow]🔍 Mission ID: {mission_id}[/yellow]")

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
                md_file = self._generate_fallback_markdown(mission_results, mission_id)
                if md_file:
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
                txt_file = self._generate_fallback_text(mission_results, mission_id)
                if txt_file:
                    extracted_files.append(("Text (Fallback)", txt_file))

            # Display summary
            if extracted_files:
                console.print(f"\n[bold green]📁 Extraction Summary:[/bold green]")
                for file_type, file_path in extracted_files:
                    file_size = self._get_file_size(file_path)
                    console.print(f"   • {file_type}: {os.path.basename(file_path)} ({file_size})")

            return True

        except Exception as e:
            console.print(f"[red]❌ Failed to extract formatted outputs: {e}[/red]")
            return False

    def _fix_json_content(self, content: str) -> str:
        """
        Attempt to fix common JSON formatting issues

        Args:
            content: Raw JSON content as string

        Returns:
            str: Fixed JSON content
        """

        # Common fixes for JSON issues
        fixed = content

        # Remove trailing commas before closing brackets/braces
        import re

        fixed = re.sub(r",(\s*[}\]])", r"\1", fixed)

        # Try to complete incomplete strings that might cause parsing errors
        # This is a basic fix - for more complex issues, manual intervention might be needed

        # If the JSON ends abruptly, try to close it properly
        if not fixed.rstrip().endswith("}"):
            # Count opening and closing braces to determine how many to add
            open_braces = fixed.count("{")
            close_braces = fixed.count("}")
            missing_braces = open_braces - close_braces

            if missing_braces > 0:
                # Add missing closing braces
                fixed = fixed.rstrip() + "}" * missing_braces

        return fixed

    def _generate_fallback_markdown(
        self, mission_results: Dict[str, Any], mission_id: str
    ) -> str | None:
        """Generate fallback markdown if formatted output is missing"""

        try:
            final_report = mission_results.get("outputs", {}).get("final_report", {})
            sections = final_report.get("sections", {})
            metadata = final_report.get("metadata", {})

            if not sections:
                console.print(f"[yellow]⚠️ No sections available for fallback markdown[/yellow]")
                return None

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

            return md_file

        except Exception as e:
            console.print(f"[red]❌ Failed to generate fallback markdown: {e}[/red]")
            return None

    def _generate_fallback_text(
        self, mission_results: Dict[str, Any], mission_id: str
    ) -> str | None:
        """Generate fallback text if formatted output is missing"""

        try:
            final_report = mission_results.get("outputs", {}).get("final_report", {})
            sections = final_report.get("sections", {})
            metadata = final_report.get("metadata", {})

            if not sections:
                console.print(f"[yellow]⚠️ No sections available for fallback text[/yellow]")
                return None

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

            return txt_file

        except Exception as e:
            console.print(f"[red]❌ Failed to generate fallback text: {e}[/red]")
            return None

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

    def batch_extract_from_directory(self, directory: str) -> int:
        """
        Extract outputs from all JSON research files in a directory

        Args:
            directory: Directory containing JSON research files

        Returns:
            int: Number of files successfully processed
        """

        processed_count = 0

        try:
            if not os.path.exists(directory):
                console.print(f"[red]❌ Directory not found: {directory}[/red]")
                return 0

            json_files = [f for f in os.listdir(directory) if f.endswith("_complete.json")]

            if not json_files:
                console.print(f"[yellow]⚠️ No research JSON files found in: {directory}[/yellow]")
                return 0

            console.print(f"\n[cyan]📁 Found {len(json_files)} research files to process[/cyan]")

            for json_file in json_files:
                json_path = os.path.join(directory, json_file)
                console.print(f"\n[blue]Processing: {json_file}[/blue]")

                if self.extract_from_json_file(json_path):
                    processed_count += 1
                    console.print(f"[green]✅ Successfully processed: {json_file}[/green]")
                else:
                    console.print(f"[red]❌ Failed to process: {json_file}[/red]")

            console.print(
                f"\n[bold green]🎯 Batch processing complete: {processed_count}/{len(json_files)} files processed[/bold green]"
            )

        except Exception as e:
            console.print(f"[red]❌ Error in batch processing: {e}[/red]")

        return processed_count


def main():
    """Main function for command line usage"""

    parser = argparse.ArgumentParser(
        description="Extract Markdown and Text outputs from research mission JSON files"
    )

    parser.add_argument("input", help="Input JSON file or directory containing research files")

    parser.add_argument(
        "-o",
        "--output",
        default="research_outputs",
        help="Output directory for extracted files (default: research_outputs)",
    )

    parser.add_argument(
        "-b", "--batch", action="store_true", help="Process all JSON files in the input directory"
    )

    args = parser.parse_args()

    # Initialize extractor
    extractor = OutputExtractor(output_dir=args.output)

    console.print("[bold cyan]🔧 Research Output Extractor[/bold cyan]")

    if args.batch or os.path.isdir(args.input):
        # Batch processing
        console.print(f"[yellow]📁 Batch processing directory: {args.input}[/yellow]")
        processed = extractor.batch_extract_from_directory(args.input)

        if processed > 0:
            console.print(
                f"\n[bold green]🎉 Successfully processed {processed} files![/bold green]"
            )
        else:
            console.print(f"\n[bold red]❌ No files were processed[/bold red]")
            sys.exit(1)

    else:
        # Single file processing
        if not os.path.exists(args.input):
            console.print(f"[red]❌ File not found: {args.input}[/red]")
            sys.exit(1)

        console.print(f"[yellow]📄 Processing single file: {args.input}[/yellow]")

        if extractor.extract_from_json_file(args.input):
            console.print(f"\n[bold green]🎉 Successfully extracted outputs![/bold green]")
        else:
            console.print(f"\n[bold red]❌ Failed to extract outputs[/bold red]")
            sys.exit(1)

    console.print(f"[cyan]📁 Output directory: {args.output}[/cyan]")


if __name__ == "__main__":
    main()
