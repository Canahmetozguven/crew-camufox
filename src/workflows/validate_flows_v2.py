"""
Simple validation script for CrewAI Flows 2.0 implementation
Tests core functionality without external dependencies
"""

import sys
import traceback
from datetime import datetime

def test_flows_v2_imports():
    """Test if Flows 2.0 components can be imported"""
    try:
        from .flows_v2 import (
            EnhancedResearchFlowV2,
            FlowOrchestrator,
            AdvancedFlowState,
            FlowEvent,
            FlowEventType,
            FlowExecutionMode,
            FlowPriority,
            CREWAI_FLOWS_V2_AVAILABLE
        )
        print("✅ Flows 2.0 imports successful")
        return True
    except Exception as e:
        print(f"❌ Flows 2.0 imports failed: {e}")
        return False

def test_integration_imports():
    """Test if integration components can be imported"""
    try:
        from .flow_integration import (
            FlowWorkflowAdapter,
            EnhancedWorkflowManager,
            FlowMigrationConfig
        )
        print("✅ Integration imports successful")
        return True
    except Exception as e:
        print(f"❌ Integration imports failed: {e}")
        return False

def test_basic_flow_creation():
    """Test basic flow creation and initialization"""
    try:
        from .flows_v2 import EnhancedResearchFlowV2, FlowEventType
        
        # Create flow
        flow = EnhancedResearchFlowV2()
        print("✅ Flow creation successful")
        
        # Test initialization
        context = {
            "query": "Test research query",
            "execution_mode": "sequential",
            "priority": 2
        }
        
        result = flow.initialize_enhanced_flow(context)
        print(f"✅ Flow initialization successful: {result.get('status')}")
        
        # Test state access
        analytics = flow.get_execution_analytics()
        print(f"✅ Analytics access successful: {analytics.get('flow_id')}")
        
        return True
    except Exception as e:
        print(f"❌ Basic flow creation failed: {e}")
        traceback.print_exc()
        return False

def test_workflow_manager():
    """Test enhanced workflow manager"""
    try:
        from .flow_integration import EnhancedWorkflowManager
        
        manager = EnhancedWorkflowManager()
        print("✅ Workflow manager creation successful")
        
        # Test analytics
        analytics = manager.get_execution_analytics()
        print(f"✅ Manager analytics successful: {analytics.get('message', 'No executions yet')}")
        
        return True
    except Exception as e:
        print(f"❌ Workflow manager test failed: {e}")
        traceback.print_exc()
        return False

def test_event_system():
    """Test event system functionality"""
    try:
        from .flows_v2 import EnhancedResearchFlowV2, FlowEvent, FlowEventType
        
        flow = EnhancedResearchFlowV2()
        
        # Test event listener
        events_received = []
        
        def test_handler(event):
            events_received.append(event)
        
        flow.add_event_listener(FlowEventType.FLOW_STARTED, test_handler)
        
        # Emit test event
        test_event = FlowEvent(
            event_type=FlowEventType.FLOW_STARTED,
            flow_id="test_flow"
        )
        flow.emit_event(test_event)
        
        if len(events_received) > 0:
            print("✅ Event system working")
            return True
        else:
            print("❌ Event system not working")
            return False
            
    except Exception as e:
        print(f"❌ Event system test failed: {e}")
        traceback.print_exc()
        return False

def test_orchestrator():
    """Test flow orchestrator"""
    try:
        from .flows_v2 import FlowOrchestrator, EnhancedResearchFlowV2
        
        orchestrator = FlowOrchestrator()
        flow = EnhancedResearchFlowV2()
        
        # Register flow
        flow_id = orchestrator.register_flow(flow)
        print(f"✅ Flow registration successful: {flow_id}")
        
        if flow_id in orchestrator.flows:
            print("✅ Orchestrator working correctly")
            return True
        else:
            print("❌ Orchestrator registration failed")
            return False
            
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all validation tests"""
    print("🚀 CrewAI Flows 2.0 Validation")
    print("=" * 50)
    
    tests = [
        ("Flows 2.0 Imports", test_flows_v2_imports),
        ("Integration Imports", test_integration_imports),
        ("Basic Flow Creation", test_basic_flow_creation),
        ("Workflow Manager", test_workflow_manager),
        ("Event System", test_event_system),
        ("Flow Orchestrator", test_orchestrator)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}...")
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 VALIDATION RESULTS")
    print("=" * 50)
    print(f"✅ Tests Passed: {passed}/{total}")
    print(f"❌ Tests Failed: {total - passed}/{total}")
    print(f"📈 Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All tests passed! Flows 2.0 implementation is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Review implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)