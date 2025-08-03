#!/usr/bin/env python3
"""
Enhanced Search Example
Demonstrates the new Google, Google Scholar, and Bing Academic search capabilities
"""

import asyncio
import json
from datetime import datetime

from src.tools.research_tools_enhanced import ResearchToolkit


async def demo_enhanced_search():
    """Demonstrate enhanced search capabilities"""
    
    # Initialize the research toolkit
    toolkit = ResearchToolkit(
        headless=True,  # Run in headless mode for better performance
        request_delay=2.0  # 2 second delay between requests to be respectful
    )
    
    research_query = "artificial intelligence machine learning 2024"
    print(f"🔍 Demonstrating enhanced search for: '{research_query}'")
    print("="*70)
    
    # 1. Enhanced Google Search (includes Google + Scholar + Bing Academic)
    print("\n📊 1. ENHANCED GOOGLE SEARCH (Google + Scholar + Bing Academic)")
    print("-" * 50)
    
    enhanced_results = await toolkit.search_enhanced_google(
        query=research_query,
        max_results=10,
        include_scholar=True,
        include_bing_academic=True
    )
    
    print(f"Found {len(enhanced_results)} enhanced results:")
    for i, result in enumerate(enhanced_results[:3], 1):
        print(f"\n{i}. {result['title'][:80]}...")
        print(f"   URL: {result['url']}")
        print(f"   Source: {result['search_engine']} | Type: {result['source_type']}")
        print(f"   Credibility: {result['credibility_score']:.2f}")
        if result['citations'] > 0:
            print(f"   Citations: {result['citations']}")
        if result['authors']:
            print(f"   Authors: {result['authors'][:100]}...")
        print(f"   Summary: {result['summary'][:200]}...")
    
    # 2. Academic-Only Search
    print("\n\n🎓 2. ACADEMIC-ONLY SEARCH (Scholar + Academic sources)")
    print("-" * 50)
    
    academic_results = await toolkit.search_academic_only(
        query=research_query,
        max_results=8
    )
    
    print(f"Found {len(academic_results)} academic results:")
    for i, result in enumerate(academic_results[:3], 1):
        print(f"\n{i}. {result['title'][:80]}...")
        print(f"   URL: {result['url']}")
        print(f"   Source: {result['search_engine']} | Credibility: {result['credibility_score']:.2f}")
        if result['citations'] > 0:
            print(f"   Citations: {result['citations']}")
        if result['authors']:
            print(f"   Authors: {result['authors'][:100]}...")
        if result['abstract']:
            print(f"   Abstract: {result['abstract'][:150]}...")
    
    # 3. Comprehensive Premium Search
    print("\n\n💎 3. COMPREHENSIVE PREMIUM SEARCH (All sources)")
    print("-" * 50)
    
    comprehensive_results = await toolkit.search_comprehensive_premium(
        query=research_query,
        max_results=15,
        prioritize_academic=True
    )
    
    print(f"Found {len(comprehensive_results)} comprehensive results:")
    
    # Show quality analysis
    academic_count = sum(1 for r in comprehensive_results if r['quality_indicators']['is_academic'])
    cited_count = sum(1 for r in comprehensive_results if r['quality_indicators']['has_citations'])
    credible_count = sum(1 for r in comprehensive_results if r['quality_indicators']['credible_domain'])
    recent_count = sum(1 for r in comprehensive_results if r['quality_indicators']['recent_content'])
    
    print(f"\nQuality Analysis:")
    print(f"   📚 Academic sources: {academic_count}/{len(comprehensive_results)}")
    print(f"   📖 With citations: {cited_count}/{len(comprehensive_results)}")
    print(f"   🏅 Credible domains: {credible_count}/{len(comprehensive_results)}")
    print(f"   📅 Recent content: {recent_count}/{len(comprehensive_results)}")
    
    # Show top 3 results
    print(f"\nTop 3 Results:")
    for i, result in enumerate(comprehensive_results[:3], 1):
        print(f"\n{i}. {result['title'][:80]}...")
        print(f"   URL: {result['url']}")
        print(f"   Source: {result['search_engine']} | Type: {result['source_type']}")
        print(f"   Credibility: {result['credibility_score']:.2f}")
        
        quality = result['quality_indicators']
        indicators = []
        if quality['is_academic']: indicators.append("🎓Academic")
        if quality['has_citations']: indicators.append("📖Cited")
        if quality['credible_domain']: indicators.append("🏅Credible")
        if quality['recent_content']: indicators.append("📅Recent")
        
        if indicators:
            print(f"   Quality: {' '.join(indicators)}")
        
        print(f"   Summary: {result['summary'][:200]}...")
    
    # 4. Search Engine Comparison
    print("\n\n⚖️  4. SEARCH ENGINE COMPARISON")
    print("-" * 50)
    
    engine_stats = {}
    for result in comprehensive_results:
        engine = result['search_engine']
        if engine not in engine_stats:
            engine_stats[engine] = {
                'count': 0,
                'avg_credibility': 0,
                'total_credibility': 0,
                'academic_count': 0,
                'citation_count': 0
            }
        
        stats = engine_stats[engine]
        stats['count'] += 1
        stats['total_credibility'] += result['credibility_score']
        stats['avg_credibility'] = stats['total_credibility'] / stats['count']
        
        if result['source_type'] == 'academic':
            stats['academic_count'] += 1
        
        if result['citations'] > 0:
            stats['citation_count'] += 1
    
    for engine, stats in engine_stats.items():
        print(f"\n{engine.upper()}:")
        print(f"   Results: {stats['count']}")
        print(f"   Avg Credibility: {stats['avg_credibility']:.2f}")
        print(f"   Academic Sources: {stats['academic_count']}")
        print(f"   With Citations: {stats['citation_count']}")
    
    # 5. Save results to file
    print("\n\n💾 5. SAVING RESULTS")
    print("-" * 50)
    
    # Prepare data for saving
    report_data = {
        "query": research_query,
        "timestamp": datetime.now().isoformat(),
        "search_results": {
            "enhanced_google": enhanced_results,
            "academic_only": academic_results,
            "comprehensive": comprehensive_results
        },
        "statistics": {
            "total_sources": len(comprehensive_results),
            "academic_sources": academic_count,
            "cited_sources": cited_count,
            "credible_sources": credible_count,
            "recent_sources": recent_count,
            "engine_breakdown": engine_stats
        }
    }
    
    # Save to JSON
    filename = f"enhanced_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"✅ Results saved to: {filename}")
    print(f"📊 Total sources found: {len(comprehensive_results)}")
    print(f"🎯 Average credibility: {sum(r['credibility_score'] for r in comprehensive_results) / len(comprehensive_results):.2f}")
    
    print("\n" + "="*70)
    print("🎉 Enhanced search demonstration complete!")
    print("🚀 Your research system now has premium search capabilities!")


async def demo_specific_searches():
    """Demonstrate specific search scenarios"""
    toolkit = ResearchToolkit(headless=True, request_delay=1.5)
    
    print("\n🎯 SPECIFIC SEARCH SCENARIOS")
    print("="*50)
    
    # Scientific research
    print("\n1. Scientific Research Query:")
    scientific_results = await toolkit.search_academic_only(
        query="climate change machine learning modeling 2024",
        max_results=5
    )
    print(f"   Found {len(scientific_results)} scientific papers")
    
    # Technology trends
    print("\n2. Technology Trends Query:")
    tech_results = await toolkit.search_enhanced_google(
        query="artificial intelligence trends 2024 industry applications",
        max_results=8,
        include_scholar=True
    )
    print(f"   Found {len(tech_results)} tech trend sources")
    
    # Market research
    print("\n3. Market Research Query:")
    market_results = await toolkit.search_comprehensive_premium(
        query="AI market size growth predictions 2024 2025",
        max_results=10,
        prioritize_academic=False  # Include more diverse sources for market data
    )
    print(f"   Found {len(market_results)} market research sources")
    
    print("\n✅ Specific scenario demonstrations complete!")


if __name__ == "__main__":
    print("🔍 ENHANCED SEARCH CAPABILITIES DEMO")
    print("Using Google, Google Scholar, and Bing Academic")
    print("="*70)
    
    try:
        # Run main demonstration
        asyncio.run(demo_enhanced_search())
        
        # Run specific scenarios
        asyncio.run(demo_specific_searches())
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        print("Make sure all dependencies are installed and the system is properly configured.")
