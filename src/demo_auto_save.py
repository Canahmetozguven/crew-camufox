#!/usr/bin/env python3
"""
Demo: Enhanced Auto-Save Research System
Demonstrates automatic saving of JSON, Markdown, and Text outputs
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from agents.multi_agent_orchestrator import MultiAgentResearchOrchestrator

console = Console()


async def demo_auto_save_system():
    """Demonstrate the enhanced auto-save research system"""

    console.print(
        Panel(
            """🚀 **Enhanced Multi-Agent Research System Demo**
        
✅ **Auto-Save Features:**
• JSON research data (always saved)
• Markdown reports (always saved)  
• Text summaries (always saved)
• Fallback generation if outputs missing
• File size tracking and summaries
• Mission management utilities

🎯 **Demo Mission:**
We'll run a quick research mission and show how all three formats 
are automatically saved every time.""",
            title="🔬 Auto-Save Demo",
            border_style="cyan",
        )
    )

    # Initialize orchestrator
    console.print("\n[bold blue]🤖 Initializing Multi-Agent Research System...[/bold blue]")

    orchestrator = MultiAgentResearchOrchestrator(headless=True, output_dir="research_outputs")

    # Show that auto-save is enabled
    orchestrator.ensure_auto_save_enabled()

    # Display capabilities
    console.print("\n")
    orchestrator.display_capabilities()

    # Demo query
    query = "What are the latest advancements in renewable energy storage technology?"

    console.print(f"\n[bold yellow]🔍 Demo Research Query:[/bold yellow]")
    console.print(f"[italic]{query}[/italic]")

    console.print(f"\n[cyan]⏰ Running research mission with auto-save enabled...[/cyan]")
    console.print(f"[yellow]💾 This will automatically save: JSON + Markdown + Text files[/yellow]")

    # Execute research mission
    results = await orchestrator.execute_research_mission(
        query=query,
        research_depth="surface",  # Quick demo
        report_type="comprehensive",
        save_outputs=True,  # This ensures all formats are saved
    )

    # Show saved files
    console.print(f"\n[bold green]📁 Auto-Saved Files Summary:[/bold green]")

    mission_id = results.get("mission_id", "unknown")
    output_dir = orchestrator.output_dir

    import os

    saved_files = []

    # Check for JSON file
    json_file = os.path.join(output_dir, f"{mission_id}_complete.json")
    if os.path.exists(json_file):
        size = orchestrator._get_file_size(json_file)
        saved_files.append(f"   • JSON: {os.path.basename(json_file)} ({size})")

    # Check for Markdown file
    md_file = os.path.join(output_dir, f"{mission_id}_report.md")
    if os.path.exists(md_file):
        size = orchestrator._get_file_size(md_file)
        saved_files.append(f"   • Markdown: {os.path.basename(md_file)} ({size})")

    # Check for Text file
    txt_file = os.path.join(output_dir, f"{mission_id}_report.txt")
    if os.path.exists(txt_file):
        size = orchestrator._get_file_size(txt_file)
        saved_files.append(f"   • Text: {os.path.basename(txt_file)} ({size})")

    for file_info in saved_files:
        console.print(file_info)

    # Show extraction utility
    console.print(f"\n[bold blue]🔧 Extraction Utility Available:[/bold blue]")
    console.print(f"[cyan]For existing JSON files, you can use:[/cyan]")
    console.print(f"[code]python src/utils/extract_outputs.py {json_file}[/code]")
    console.print(f"[code]python src/utils/extract_outputs.py research_outputs/ --batch[/code]")

    # List all missions
    missions = orchestrator.list_saved_research_missions()
    if missions:
        console.print(f"\n[bold green]📚 All Saved Research Missions:[/bold green]")
        for i, mission in enumerate(missions[:5]):  # Show latest 5
            console.print(f"   {i+1}. {mission['mission_id']}")
            if mission["markdown_file"] and mission["text_file"]:
                console.print(f"      ✅ Complete (JSON + Markdown + Text)")
            elif mission["markdown_file"] or mission["text_file"]:
                console.print(f"      ⚠️ Partial (some formats missing)")
            else:
                console.print(f"      📄 JSON only")

    console.print(f"\n[bold green]🎉 Auto-Save Demo Complete![/bold green]")
    console.print(
        f"[yellow]💡 Every research mission automatically saves all three formats![/yellow]"
    )

    return results


async def demo_extraction_utility():
    """Demonstrate the extraction utility for existing JSON files"""

    console.print(
        Panel(
            """🔧 **Output Extraction Utility Demo**
        
If you have existing JSON research files without Markdown/Text outputs,
you can use the extraction utility to generate them.

The utility can:
• Extract from single JSON files
• Batch process entire directories
• Generate fallback outputs if missing
• Handle multiple file formats""",
            title="🛠️ Extraction Demo",
            border_style="blue",
        )
    )

    # Initialize orchestrator to get extraction functionality
    orchestrator = MultiAgentResearchOrchestrator(headless=True, output_dir="research_outputs")

    # Check for existing JSON files
    import os

    output_dir = "research_outputs"

    if os.path.exists(output_dir):
        json_files = [f for f in os.listdir(output_dir) if f.endswith("_complete.json")]

        if json_files:
            console.print(f"\n[cyan]📁 Found {len(json_files)} research JSON files[/cyan]")

            # Extract from the first JSON file as demo
            json_file_path = os.path.join(output_dir, json_files[0])
            console.print(f"[yellow]🔧 Demonstrating extraction from: {json_files[0]}[/yellow]")

            success = await orchestrator.extract_formatted_outputs_from_json(json_file_path)

            if success:
                console.print(f"[green]✅ Extraction successful![/green]")
            else:
                console.print(f"[red]❌ Extraction failed[/red]")
        else:
            console.print(f"[yellow]⚠️ No research JSON files found in {output_dir}[/yellow]")
            console.print(f"[cyan]💡 Run the main demo first to generate research files[/cyan]")
    else:
        console.print(f"[yellow]⚠️ Output directory not found: {output_dir}[/yellow]")


async def main():
    """Main demo function"""

    console.print("[bold cyan]🚀 Multi-Agent Research System - Auto-Save Demo[/bold cyan]\n")

    try:
        # Run the main auto-save demo
        await demo_auto_save_system()

        # Add separator
        console.print("\n" + "=" * 80 + "\n")

        # Run the extraction utility demo
        await demo_extraction_utility()

    except KeyboardInterrupt:
        console.print(f"\n[yellow]⏹️ Demo interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Demo error: {e}[/red]")

    console.print(
        f"\n[bold green]🎯 Demo complete! Check the research_outputs/ directory[/bold green]"
    )


if __name__ == "__main__":
    asyncio.run(main())
