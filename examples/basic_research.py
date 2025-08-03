#!/usr/bin/env python3
"""
Basic Research Example
Simple usage of the Deep Web Research Tool
"""

import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import DeepWebResearcher


async def basic_research_example():
    """Demonstrate basic research functionality"""
    
    # Initialize the researcher
    researcher = DeepWebResearcher()
    
    # Example 1: Technology research
    print("🔍 Example 1: Technology Research")
    print("-" * 50)
    
    tech_report = await researcher.research(
        query="artificial intelligence latest developments 2024",
        max_sources=10,
        depth="medium",
        fact_check=True
    )
    
    researcher.display_results(tech_report)
    
    # Example 2: Business research
    print("\n\n🔍 Example 2: Business Research")  
    print("-" * 50)
    
    business_report = await researcher.research(
        query="electric vehicle market trends",
        focus_areas=["market share", "key players", "growth projections"],
        max_sources=8,
        depth="surface"
    )
    
    researcher.display_results(business_report)


if __name__ == "__main__":
    asyncio.run(basic_research_example())
