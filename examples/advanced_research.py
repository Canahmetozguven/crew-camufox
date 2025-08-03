#!/usr/bin/env python3
"""
Advanced Research Example
Demonstrates advanced features of the Deep Web Research Tool
"""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import DeepWebResearcher


async def advanced_research_example():
    """Demonstrate advanced research features"""
    
    # Initialize researcher with custom configuration
    researcher = DeepWebResearcher(
        model_name="magistral:24b",
        headless=True,
        proxy=None  # Set proxy if needed
    )
    
    # Example 1: Deep research with fact checking
    print("🔍 Advanced Example 1: Deep Research with Fact Checking")
    print("-" * 60)
    
    deep_report = await researcher.research(
        query="quantum computing breakthroughs 2024",
        focus_areas=[
            "quantum supremacy achievements",
            "commercial applications", 
            "major companies involved",
            "technical challenges"
        ],
        max_sources=20,
        depth="deep",
        fact_check=True,
        exclude_domains=["wikipedia.org", "reddit.com"]  # Exclude certain domains
    )
    
    researcher.display_results(deep_report)
    
    # Example 2: Targeted research with specific focus
    print("\n\n🔍 Advanced Example 2: Targeted Market Research")
    print("-" * 60)
    
    market_report = await researcher.research(
        query="sustainable energy investments",
        focus_areas=[
            "venture capital trends",
            "government policies",
            "technology innovations",
            "market forecasts"
        ],
        max_sources=15,
        depth="medium",
        fact_check=True
    )
    
    researcher.display_results(market_report)
    
    # Example 3: Academic research focus
    print("\n\n🔍 Advanced Example 3: Academic Research")
    print("-" * 60)
    
    academic_report = await researcher.research(
        query="machine learning interpretability research",
        focus_areas=[
            "explainable AI methods",
            "recent publications", 
            "ethical considerations",
            "industry applications"
        ],
        max_sources=12,
        depth="deep",
        fact_check=True
    )
    
    researcher.display_results(academic_report)


async def custom_research_workflow():
    """Demonstrate custom research workflow"""
    
    print("\n\n🔧 Custom Research Workflow")
    print("-" * 60)
    
    researcher = DeepWebResearcher()
    
    # Step 1: Create custom research plan
    print("Step 1: Creating custom research plan...")
    plan = researcher.coordinator.create_research_plan(
        query="blockchain technology adoption in healthcare",
        focus_areas=["privacy", "interoperability", "regulatory compliance"],
        max_sources=15,
        depth="deep"
    )
    
    print(f"Research plan created with {len(plan['search_terms'])} search terms")
    print("Search terms:", plan['search_terms'][:5])
    
    # Step 2: Execute research
    print("\nStep 2: Executing research...")
    report = await researcher.research(
        query="blockchain technology adoption in healthcare",
        focus_areas=["privacy", "interoperability", "regulatory compliance"],
        max_sources=15,
        depth="deep"
    )
    
    # Step 3: Custom analysis
    print("\nStep 3: Custom analysis...")
    sources = report.get("sources", [])
    
    # Analyze by source type
    source_types = {}
    for source in sources:
        source_type = source.get("source_type", "unknown")
        if source_type not in source_types:
            source_types[source_type] = []
        source_types[source_type].append(source)
    
    print("Sources by type:")
    for source_type, type_sources in source_types.items():
        print(f"  {source_type}: {len(type_sources)} sources")
    
    # Find highest credibility sources
    high_cred_sources = [s for s in sources if s.get("credibility_score", 0) > 0.8]
    print(f"\nHigh credibility sources (>0.8): {len(high_cred_sources)}")
    
    researcher.display_results(report)


if __name__ == "__main__":
    print("🚀 Advanced Deep Web Research Examples")
    print("=" * 80)
    
    # Run advanced examples
    asyncio.run(advanced_research_example())
    
    # Run custom workflow
    asyncio.run(custom_research_workflow())
