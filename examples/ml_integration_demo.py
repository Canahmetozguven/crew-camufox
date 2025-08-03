#!/usr/bin/env python3
"""
ML Integration Demo

Demonstrates the comprehensive ML-enhanced research capabilities
provided by the crew-camufox ML integration system.
"""

import asyncio
import json
import os
from pathlib import Path
from datetime import datetime

# Add the src directory to the path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml import (
    MLCoordinator, QueryOptimizer, SourceQualityPredictor, 
    PatternRecognizer, RecommendationEngine
)
from ml.ml_coordinator import MLConfiguration, MLCapability


async def demo_ml_integration():
    """Demonstrate comprehensive ML integration capabilities"""
    print("🤖 Crew-Camufox ML Integration Demo")
    print("=" * 50)
    
    # Create configuration for all ML capabilities
    config = MLConfiguration(
        enabled_capabilities=[
            MLCapability.QUERY_OPTIMIZATION,
            MLCapability.SOURCE_QUALITY_PREDICTION,
            MLCapability.PATTERN_RECOGNITION,
            MLCapability.RESEARCH_RECOMMENDATIONS
        ],
        model_path="./ml_models",
        learning_rate=0.01,
        feedback_threshold=10,
        auto_save_interval=50,
        quality_threshold=0.7
    )
    
    # Initialize ML coordinator
    coordinator = MLCoordinator(config)
    
    print(f"✅ ML Coordinator initialized with {len(config.enabled_capabilities)} capabilities")
    
    # Check capability status
    status = coordinator.get_capability_status()
    print("\n📊 ML Capabilities Status:")
    for capability, details in status["capabilities"].items():
        status_icon = "✅" if details["enabled"] else "❌"
        print(f"  {status_icon} {capability}: {details['component_type'] or 'Not available'}")
    
    # Demo 1: Query Optimization
    print("\n🔍 Demo 1: Query Optimization")
    print("-" * 30)
    
    test_queries = [
        "AI research papers",
        "climate change impact",
        "machine learning applications",
        "renewable energy technologies"
    ]
    
    for query in test_queries:
        result = await coordinator.optimize_query(query)
        if result["status"] == "success":
            print(f"Original: '{query}'")
            print(f"Optimized: '{result['optimized_query']}'")
            if result["expansion"]:
                print(f"Expanded terms: {result['expansion']['expanded_terms'][:3]}")
                print(f"Confidence: {result['expansion']['confidence_score']:.2f}")
            print()
    
    # Demo 2: Source Quality Prediction
    print("🏆 Demo 2: Source Quality Prediction")
    print("-" * 30)
    
    test_sources = [
        {
            "url": "https://nature.com/articles/science-breakthrough",
            "content": "Peer-reviewed research article published in Nature journal discussing breakthrough in quantum computing. The study presents rigorous methodology, statistical analysis, and peer review validation.",
            "type": "academic"
        },
        {
            "url": "https://myblog.com/my-opinion-on-ai",
            "content": "Personal blog post sharing opinions about AI. No citations or references provided. Based on personal experience only.",
            "type": "blog"
        },
        {
            "url": "https://reuters.com/technology/ai-breakthrough",
            "content": "Breaking news report from Reuters about AI breakthrough. Multiple sources cited, expert quotes included, fact-checked content.",
            "type": "news"
        }
    ]
    
    for source in test_sources:
        result = await coordinator.predict_source_quality(
            source["url"], 
            source["content"], 
            source["type"]
        )
        
        if result["status"] == "success":
            print(f"Source: {source['url']}")
            print(f"Type: {source['type']}")
            print(f"Overall Quality: {result['overall_quality']:.2f}")
            print(f"Credibility: {result['credibility_level']}")
            print(f"Authority: {result['quality_scores']['authority']:.2f}")
            print(f"Accuracy: {result['quality_scores']['accuracy']:.2f}")
            print()
    
    # Demo 3: Research Session Analysis
    print("📊 Demo 3: Research Session Analysis")
    print("-" * 30)
    
    # Simulate research session data
    session_data = {
        "user_id": "demo_user",
        "query": "artificial intelligence applications in healthcare",
        "search_engines": ["google", "bing", "academic"],
        "sources_found": 15,
        "sources_used": 8,
        "time_spent": 45,
        "domain": "healthcare",
        "quality_threshold": 0.7,
        "collaboration": False,
        "depth": "detailed"
    }
    
    result = await coordinator.analyze_research_session(session_data)
    if result["status"] == "success":
        print(f"Session Analysis ID: {result['observation_id']}")
        print(f"Pattern Type: {result['pattern_type']}")
        print(f"Success Metrics: {result['success_metrics']}")
        
        if result["recommendations"]:
            print("\n💡 AI Recommendations:")
            for i, rec in enumerate(result["recommendations"][:3], 1):
                print(f"  {i}. {rec['title']}")
                print(f"     {rec['description']}")
                print(f"     Confidence: {rec['confidence']:.2f} | Priority: {rec['priority']}")
                print()
    
    # Demo 4: Personalized Recommendations
    print("🎯 Demo 4: Personalized Recommendations")
    print("-" * 30)
    
    research_context = {
        "query": "sustainable energy solutions",
        "domain": "environment",
        "urgency": "high",
        "depth": "comprehensive",
        "collaboration": True,
        "quality_threshold": 0.8,
        "available_time": 120,
        "previous_queries": [
            "renewable energy", 
            "solar panel efficiency", 
            "wind energy storage"
        ]
    }
    
    recommendations = await coordinator.get_research_recommendations("demo_user", research_context)
    
    if recommendations:
        print("🌟 Personalized Research Recommendations:")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"\n{i}. {rec['title']} [{rec['priority']}]")
            print(f"   Type: {rec['type']}")
            print(f"   Expected Benefit: {rec['expected_benefit']}")
            print(f"   Confidence: {rec['confidence']:.2f}")
            print(f"   Implementation: {rec['implementation_effort']}")
            
            if rec['implementation_steps']:
                print("   Steps:")
                for step in rec['implementation_steps'][:2]:
                    print(f"     • {step}")
    
    # Demo 5: Learning from Feedback
    print("\n🧠 Demo 5: Learning from Feedback")
    print("-" * 30)
    
    # Simulate research outcomes
    outcomes = {
        "total_results": 25,
        "relevant_results": 18,
        "quality_score": 0.85,
        "execution_time": 42.5,
        "success_rate": 0.72,
        "user_satisfaction": 4.2
    }
    
    # Simulate user feedback
    feedback = {
        "overall_satisfaction": 4,
        "query_optimization_helpful": True,
        "source_quality_accurate": True,
        "recommendation_feedback": [
            {
                "recommendation_id": "rec_001",
                "was_helpful": True,
                "improvement": 0.3,
                "difficulty": "easy",
                "comments": "Great suggestion, saved time"
            }
        ]
    }
    
    coordinator.learn_from_feedback(session_data, outcomes, feedback)
    print("✅ Feedback processed and models updated")
    
    # Demo 6: ML Insights
    print("\n📈 Demo 6: ML Insights Dashboard")
    print("-" * 30)
    
    insights = coordinator.get_ml_insights()
    
    print("🔍 Query Optimization Insights:")
    if insights.query_insights:
        for key, value in insights.query_insights.items():
            print(f"  {key}: {value}")
    else:
        print("  No optimization data available yet")
    
    print("\n🏆 Quality Prediction Insights:")
    if insights.quality_insights:
        for key, value in insights.quality_insights.items():
            print(f"  {key}: {value}")
    else:
        print("  No quality prediction data available yet")
    
    print("\n📊 Overall Performance:")
    for key, value in insights.overall_performance.items():
        print(f"  {key}: {value}")
    
    # Demo 7: Export ML Data
    print("\n💾 Demo 7: Export ML Data")
    print("-" * 30)
    
    export_path = "ml_export_demo.json"
    coordinator.export_ml_data(export_path)
    print(f"✅ ML data exported to: {export_path}")
    
    # Show export sample
    if os.path.exists(export_path):
        with open(export_path, "r") as f:
            export_data = json.load(f)
        
        print("📋 Export Sample:")
        print(f"  Session Counter: {export_data['coordinator_state']['session_counter']}")
        print(f"  Learning Enabled: {export_data['coordinator_state']['learning_enabled']}")
        print(f"  Capabilities: {len(export_data['coordinator_state']['configuration']['enabled_capabilities'])}")
        print(f"  Export Time: {export_data['export_timestamp']}")
    
    print("\n🎉 ML Integration Demo Complete!")
    print("=" * 50)
    print("Key Features Demonstrated:")
    print("✅ Query optimization with ML expansion")
    print("✅ Intelligent source quality prediction")
    print("✅ Research pattern recognition")
    print("✅ Personalized recommendations")
    print("✅ Continuous learning from feedback")
    print("✅ Comprehensive insights and analytics")
    print("✅ Data export for analysis")


async def demo_individual_components():
    """Demonstrate individual ML components"""
    print("\n🔧 Individual Components Demo")
    print("=" * 50)
    
    # Query Optimizer Demo
    print("1. Query Optimizer")
    print("-" * 20)
    optimizer = QueryOptimizer("./models/query_optimizer")
    
    query = "machine learning in healthcare"
    optimized, expansion = optimizer.optimize_query(query)
    print(f"Original: {query}")
    print(f"Optimized: {optimized}")
    print(f"Expansion terms: {expansion.expanded_terms[:3]}")
    
    # Source Quality Predictor Demo
    print("\n2. Source Quality Predictor")
    print("-" * 20)
    predictor = SourceQualityPredictor("./models/quality_predictor")
    
    url = "https://example-journal.com/research-paper"
    content = "Peer-reviewed research with methodology, references, and statistical analysis."
    
    from ml.source_predictor import ContentType
    features = predictor.extract_features(url, content)
    quality = predictor.predict_quality(features, ContentType.ACADEMIC_PAPER)
    
    print(f"URL: {url}")
    print(f"Overall Quality: {quality.overall_quality:.2f}")
    print(f"Authority Score: {quality.authority_score:.2f}")
    
    # Pattern Recognizer Demo
    print("\n3. Pattern Recognizer")
    print("-" * 20)
    recognizer = PatternRecognizer("./models/pattern_recognizer")
    
    session = {
        "query": "AI research",
        "time_spent": 30,
        "sources_quality": [0.8, 0.9, 0.7],
        "success_rate": 0.85
    }
    
    observation = recognizer.observe_research_session(session)
    print(f"Pattern ID: {observation.pattern_id}")
    print(f"Pattern Type: {observation.pattern_type.value}")
    
    # Recommendation Engine Demo
    print("\n4. Recommendation Engine")
    print("-" * 20)
    rec_engine = RecommendationEngine("./models/recommendation_engine")
    
    from ml.recommendation_engine import ResearchContext
    context = ResearchContext(
        query="sustainable technology",
        domain="environment",
        urgency="medium",
        depth="detailed",
        collaboration=False,
        quality_threshold=0.7,
        available_time=60
    )
    
    recommendations = rec_engine.generate_recommendations("demo_user", context)
    if recommendations:
        print(f"Generated {len(recommendations)} recommendations")
        print(f"Top recommendation: {recommendations[0].title}")
    
    print("\n✅ Individual components demo complete!")


if __name__ == "__main__":
    print("🚀 Starting Crew-Camufox ML Integration Demo")
    print("This demo showcases the comprehensive ML capabilities")
    print("for intelligent research enhancement.\n")
    
    # Run main demo
    asyncio.run(demo_ml_integration())
    
    # Run individual components demo
    asyncio.run(demo_individual_components())
    
    print("\n🏆 Demo completed successfully!")
    print("The ML integration system is ready for production use.")
