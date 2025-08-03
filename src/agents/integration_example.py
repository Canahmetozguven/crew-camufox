#!/usr/bin/env python3
"""
Integration Example: Enhanced Deep Researcher Agent with Tool Composition
Demonstrates how to use the enhanced agent with tool composition capabilities
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

# Example usage of the Enhanced Deep Researcher Agent
async def example_enhanced_research():
    """Example of using the Enhanced Deep Researcher Agent"""
    
    try:
        # Import the enhanced agent
        from src.agents.enhanced_deep_researcher import EnhancedDeepResearcherAgent
        
        print("🚀 Enhanced Deep Researcher Agent Integration Example")
        print("=" * 60)
        
        # Initialize the enhanced agent with tool composition enabled
        researcher = EnhancedDeepResearcherAgent(
            model_name="magistral:latest",
            browser_model_name="granite3.3:8b",
            headless=True,
            use_composition=True  # Enable tool composition
        )
        
        # Display capabilities
        capabilities = researcher.get_capabilities()
        print(f"✅ Agent initialized with capabilities:")
        print(f"   - Agent Type: {capabilities['agent_type']}")
        print(f"   - Version: {capabilities['version']}")
        print(f"   - Main Model: {capabilities['models']['main']}")
        print(f"   - Browser Model: {capabilities['models']['browser']}")
        print(f"   - Tool Composition: {'Enabled' if 'tool_composition' in capabilities else 'Disabled'}")
        print(f"   - Features: {len(capabilities['features'])} available")
        print()
        
        # Example research plan
        research_plan = {
            "id": "example_001",
            "query": "artificial intelligence safety research 2024",
            "research_depth": "comprehensive",
            "max_sources": 15,
            "execution_phases": [
                {
                    "phase": 1,
                    "name": "Initial Discovery",
                    "focus": "primary_sources"
                },
                {
                    "phase": 2,
                    "name": "Deep Analysis",
                    "focus": "detailed_content"
                }
            ],
            "search_strategies": {
                "primary_terms": [
                    "artificial intelligence safety research 2024",
                    "AI alignment research papers",
                    "machine learning safety frameworks"
                ],
                "secondary_terms": [
                    "AI safety governance",
                    "responsible AI development",
                    "AI risk assessment methodologies"
                ]
            },
            "quality_criteria": {
                "content_quality": {
                    "min_word_count": 200
                },
                "exclusion_criteria": [
                    "paywall_protected",
                    "broken_links"
                ]
            },
            "fact_check_strategy": {
                "enabled": True
            },
            "config": {
                "sources_per_round": 10
            }
        }
        
        print("📋 Research Plan:")
        print(f"   - Query: {research_plan['query']}")
        print(f"   - Depth: {research_plan['research_depth']}")
        print(f"   - Max Sources: {research_plan['max_sources']}")
        print(f"   - Phases: {len(research_plan['execution_phases'])}")
        print()
        
        # Execute research
        print("🔍 Starting Enhanced Research...")
        start_time = datetime.now()
        
        results = await researcher.execute_research_plan(research_plan)
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        # Display results
        print("\n📊 Research Results Summary:")
        print("=" * 40)
        print(f"✅ Status: {results['completion_status']}")
        print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
        print(f"📄 Sources Found: {len(results.get('sources', []))}")
        print(f"🔧 Enhancement: {results.get('enhancement_used', 'unknown')}")
        
        # Quality metrics
        quality_metrics = results.get("quality_metrics", {})
        if quality_metrics:
            print(f"\n📈 Quality Metrics:")
            print(f"   - Overall Score: {quality_metrics.get('overall_score', 0):.3f}")
            print(f"   - Avg Credibility: {quality_metrics.get('avg_credibility', 0):.3f}")
            print(f"   - Avg Relevance: {quality_metrics.get('avg_relevance', 0):.3f}")
            print(f"   - Source Diversity: {quality_metrics.get('source_diversity', 0):.3f}")
            print(f"   - Content Completeness: {quality_metrics.get('completeness', 0):.3f}")
        
        # Performance stats (if tool composition was used)
        performance_stats = results.get("performance_stats", {})
        if performance_stats:
            search_performance = performance_stats.get("search_pipeline_performance", {})
            if search_performance:
                print(f"\n⚡ Performance Statistics:")
                print(f"   - Total Executions: {search_performance.get('total_executions', 0)}")
                print(f"   - Success Rate: {search_performance.get('success_rate', 0)}%")
                print(f"   - Avg Execution Time: {search_performance.get('average_execution_time', 0):.2f}s")
        
        # Show sample sources
        sources = results.get("sources", [])
        if sources:
            print(f"\n📚 Sample Sources ({min(3, len(sources))} of {len(sources)}):")
            for i, source in enumerate(sources[:3], 1):
                print(f"   {i}. {source.get('title', 'No title')[:60]}...")
                print(f"      URL: {source.get('url', 'No URL')}")
                print(f"      Credibility: {source.get('credibility_score', 0):.2f}")
                print(f"      Relevance: {source.get('relevance_score', 0):.2f}")
                print()
        
        # Health check
        print("🏥 System Health Check:")
        health_status = await researcher.health_check()
        print(f"   - Overall Status: {health_status.get('overall_status', 'unknown')}")
        if 'search_pipeline' in health_status:
            search_health = health_status['search_pipeline']
            print(f"   - Search Pipeline: {search_health.get('status', 'unknown')}")
        
        print("\n✅ Enhanced Research Integration Example Completed!")
        return results
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please ensure all dependencies are installed:")
        print("   - pip install crewai camoufox langchain-ollama pydantic rich beautifulsoup4")
        return None
        
    except Exception as e:
        print(f"❌ Error during research: {e}")
        return None


async def example_comparison():
    """Example comparing enhanced vs legacy mode"""
    
    try:
        from src.agents.enhanced_deep_researcher import EnhancedDeepResearcherAgent
        
        print("\n🔬 Enhanced vs Legacy Mode Comparison")
        print("=" * 50)
        
        # Simple research plan for comparison
        simple_plan = {
            "id": "comparison_001",
            "query": "renewable energy trends 2024",
            "research_depth": "moderate",
            "max_sources": 8,
            "execution_phases": [
                {
                    "phase": 1,
                    "name": "Primary Search",
                    "focus": "main_sources"
                }
            ],
            "search_strategies": {
                "primary_terms": ["renewable energy trends 2024"],
                "secondary_terms": []
            },
            "quality_criteria": {
                "content_quality": {"min_word_count": 150}
            },
            "fact_check_strategy": {"enabled": False},
            "config": {"sources_per_round": 8}
        }
        
        # Test Enhanced Mode
        print("🚀 Testing Enhanced Mode (Tool Composition)...")
        enhanced_researcher = EnhancedDeepResearcherAgent(
            model_name="magistral:latest",
            browser_model_name="granite3.3:8b",
            headless=True,
            use_composition=True
        )
        
        start_time = datetime.now()
        enhanced_results = await enhanced_researcher.execute_research_plan(simple_plan)
        enhanced_time = (datetime.now() - start_time).total_seconds()
        
        # Test Legacy Mode
        print("\n🔄 Testing Legacy Mode...")
        legacy_researcher = EnhancedDeepResearcherAgent(
            model_name="magistral:latest",
            browser_model_name="granite3.3:8b",
            headless=True,
            use_composition=False
        )
        
        start_time = datetime.now()
        legacy_results = await legacy_researcher.execute_research_plan(simple_plan)
        legacy_time = (datetime.now() - start_time).total_seconds()
        
        # Compare results
        print("\n📊 Comparison Results:")
        print("=" * 30)
        
        enhanced_sources = len(enhanced_results.get("sources", []))
        legacy_sources = len(legacy_results.get("sources", []))
        
        print(f"Enhanced Mode:")
        print(f"   - Sources: {enhanced_sources}")
        print(f"   - Time: {enhanced_time:.2f}s")
        print(f"   - Status: {enhanced_results.get('completion_status', 'unknown')}")
        
        print(f"\nLegacy Mode:")
        print(f"   - Sources: {legacy_sources}")
        print(f"   - Time: {legacy_time:.2f}s")
        print(f"   - Status: {legacy_results.get('completion_status', 'unknown')}")
        
        # Performance improvement
        if enhanced_time > 0 and legacy_time > 0:
            time_improvement = ((legacy_time - enhanced_time) / legacy_time) * 100
            print(f"\n⚡ Performance:")
            print(f"   - Time improvement: {time_improvement:.1f}%")
            print(f"   - Source efficiency: {enhanced_sources/enhanced_time:.2f} sources/sec vs {legacy_sources/legacy_time:.2f} sources/sec")
        
        return {
            "enhanced": enhanced_results,
            "legacy": legacy_results,
            "comparison": {
                "enhanced_time": enhanced_time,
                "legacy_time": legacy_time,
                "enhanced_sources": enhanced_sources,
                "legacy_sources": legacy_sources
            }
        }
        
    except Exception as e:
        print(f"❌ Comparison error: {e}")
        return None


if __name__ == "__main__":
    async def main():
        """Run the integration examples"""
        
        # Run basic example
        await example_enhanced_research()
        
        # Run comparison example
        await example_comparison()
        
        print("\n🎉 All integration examples completed!")
    
    # Run the examples
    asyncio.run(main())