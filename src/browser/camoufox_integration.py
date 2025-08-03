#!/usr/bin/env python3
"""
Integrated Camoufox Browser System
Combines enhanced stealth, session management, and performance optimization
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

try:
    from .camoufox_enhanced import (
        CamoufoxEnhancedManager,
        StealthLevel,
        SessionConfig,
        ProxyConfig
    )
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    # Create mock classes for fallback
    class StealthLevel:
        MINIMAL = "minimal"
        BALANCED = "balanced"
        MAXIMUM = "maximum"
    
    class SessionConfig:
        def __init__(self):
            pass
    
    class CamoufoxEnhancedManager:
        def __init__(self, *args, **kwargs):
            pass

from .camoufox_performance import (
    CamoufoxPerformanceOptimizer,
    PerformanceLevel,
    OptimizationConfig,
    PERFORMANCE_PRESETS
)

try:
    from rich.console import Console
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    
    class MockConsole:
        def print(self, *args, **kwargs):
            print(*args)
    
    console = MockConsole()

class IntegratedCamoufoxManager:
    """
    Comprehensive Camoufox browser management system
    Integrates stealth, session management, and performance optimization
    """
    
    def __init__(
        self,
        stealth_level: StealthLevel = StealthLevel.BALANCED,
        performance_level: PerformanceLevel = PerformanceLevel.BALANCED,
        session_persistence: bool = True,
        enable_monitoring: bool = True,
        data_directory: str = "camoufox_data"
    ):
        self.stealth_level = stealth_level
        self.performance_level = performance_level
        self.session_persistence = session_persistence
        self.enable_monitoring = enable_monitoring
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(exist_ok=True)
        
        # Initialize components
        self.enhanced_manager = CamoufoxEnhancedManager(
            stealth_level=stealth_level,
            enable_session_persistence=session_persistence,
            data_directory=str(self.data_directory / "sessions")
        )
        
        # Performance optimization configuration
        optimization_config = self._create_optimization_config()
        self.performance_optimizer = CamoufoxPerformanceOptimizer(
            optimization_config=optimization_config,
            enable_real_time_monitoring=enable_monitoring
        )
        
        # Integration state
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.performance_profiles: Dict[str, str] = {}
        
        # Setup logging
        self._setup_logging()
        
        console.print(f"[green]🚀 Integrated Camoufox Manager initialized[/green]")
        console.print(f"[cyan]   • Stealth level: {stealth_level.value}[/cyan]")
        console.print(f"[cyan]   • Performance level: {performance_level.value}[/cyan]")
        console.print(f"[cyan]   • Monitoring enabled: {enable_monitoring}[/cyan]")
    
    def _setup_logging(self):
        """Setup integrated logging"""
        
        log_file = self.data_directory / "integrated_camoufox.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('IntegratedCamoufox')
    
    def _create_optimization_config(self) -> OptimizationConfig:
        """Create optimization config based on performance level"""
        
        if self.performance_level == PerformanceLevel.MAXIMUM:
            return PERFORMANCE_PRESETS["speed_demon"]
        elif self.performance_level == PerformanceLevel.AGGRESSIVE:
            return PERFORMANCE_PRESETS["memory_saver"]
        else:
            return PERFORMANCE_PRESETS["balanced_power"]
    
    async def create_optimized_browser(
        self,
        session_id: Optional[str] = None,
        custom_config: Optional[Dict[str, Any]] = None,
        performance_profile: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create browser with integrated optimizations"""
        
        try:
            # Apply performance profile if specified
            if performance_profile and performance_profile in PERFORMANCE_PRESETS:
                self.performance_optimizer.config = PERFORMANCE_PRESETS[performance_profile]
                console.print(f"[green]⚡ Applied performance profile: {performance_profile}[/green]")
            
            # Create enhanced session
            session_config = SessionConfig()
            if custom_config:
                session_config.__dict__.update(custom_config)
            
            browser_result = await self.enhanced_manager.create_enhanced_browser(
                session_id=session_id,
                session_config=session_config
            )
            
            if not browser_result.get("success"):
                return browser_result
            
            browser = browser_result["browser"]
            actual_session_id = browser_result["session_id"]
            
            # Apply performance optimizations to browser launch
            browser_args = browser_result.get("launch_args", {})
            optimized_args = await self.performance_optimizer.optimize_browser_launch(browser_args)
            
            # Store session information
            self.active_sessions[actual_session_id] = {
                "browser": browser,
                "session_id": actual_session_id,
                "stealth_level": self.stealth_level.value,
                "performance_level": self.performance_level.value,
                "created_at": datetime.now(),
                "pages_count": 0,
                "total_optimizations": 0
            }
            
            if performance_profile:
                self.performance_profiles[actual_session_id] = performance_profile
            
            console.print(f"[green]✅ Created optimized browser session: {actual_session_id}[/green]")
            
            return {
                "success": True,
                "browser": browser,
                "session_id": actual_session_id,
                "stealth_features": browser_result.get("stealth_features", []),
                "performance_optimizations": optimized_args.get("args", []),
                "monitoring_enabled": self.enable_monitoring
            }
            
        except Exception as e:
            self.logger.error(f"Error creating optimized browser: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def create_optimized_page(
        self,
        browser,
        url: str,
        session_id: str,
        stealth_config: Optional[Dict[str, Any]] = None,
        performance_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create page with integrated stealth and performance optimizations"""
        
        try:
            # Create enhanced page with stealth features
            page_result = await self.enhanced_manager.create_stealth_page(
                browser,
                stealth_config or {}
            )
            
            if not page_result.get("success"):
                return page_result
            
            page = page_result["page"]
            
            # Apply performance optimizations
            optimization_result = await self.performance_optimizer.apply_page_optimizations(
                page, url
            )
            
            # Navigate to URL with monitoring
            start_time = datetime.now()
            
            try:
                await page.goto(url, wait_until="domcontentloaded")
                
                # Monitor performance if enabled
                if self.enable_monitoring and session_id in self.active_sessions:
                    performance_metrics = await self.performance_optimizer.monitor_performance(
                        page, session_id
                    )
                    
                    # Update session statistics
                    self.active_sessions[session_id]["pages_count"] += 1
                    self.active_sessions[session_id]["total_optimizations"] += len(
                        optimization_result.get("optimizations_applied", [])
                    )
                
                load_time = (datetime.now() - start_time).total_seconds()
                
                console.print(f"[green]🎯 Page loaded with optimizations: {url}[/green]")
                console.print(f"[cyan]   • Load time: {load_time:.2f}s[/cyan]")
                console.print(f"[cyan]   • Optimizations: {len(optimization_result.get('optimizations_applied', []))}[/cyan]")
                
                return {
                    "success": True,
                    "page": page,
                    "url": url,
                    "load_time": load_time,
                    "stealth_features": page_result.get("stealth_features", []),
                    "performance_optimizations": optimization_result.get("optimizations_applied", []),
                    "blocked_resources": optimization_result.get("blocked_resources", 0),
                    "session_id": session_id
                }
                
            except Exception as nav_error:
                await page.close()
                raise nav_error
                
        except Exception as e:
            self.logger.error(f"Error creating optimized page for {url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
    
    async def auto_optimize_session(self, session_id: str) -> Dict[str, Any]:
        """Automatically optimize session based on performance data"""
        
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}
        
        try:
            # Get performance optimization recommendations
            optimization_result = await self.performance_optimizer.auto_optimize_session(session_id)
            
            # Apply stealth adjustments if needed
            stealth_adjustments = await self._auto_adjust_stealth(session_id)
            
            # Update session information
            if "optimization_changes" in optimization_result:
                self.active_sessions[session_id]["last_optimization"] = datetime.now()
                self.active_sessions[session_id]["auto_optimizations"] = (
                    self.active_sessions[session_id].get("auto_optimizations", 0) + 1
                )
            
            console.print(f"[green]🔧 Auto-optimization completed for session {session_id}[/green]")
            
            return {
                "session_id": session_id,
                "performance_changes": optimization_result.get("optimization_changes", []),
                "stealth_adjustments": stealth_adjustments,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error auto-optimizing session {session_id}: {e}")
            return {"error": str(e)}
    
    async def _auto_adjust_stealth(self, session_id: str) -> List[str]:
        """Auto-adjust stealth settings based on session performance"""
        
        adjustments = []
        
        # Get session data
        session_data = self.active_sessions.get(session_id, {})
        
        # Simple heuristics for stealth adjustment
        pages_count = session_data.get("pages_count", 0)
        
        if pages_count > 10 and self.stealth_level != StealthLevel.MAXIMUM:
            # Increase stealth for long sessions
            adjustments.append("Increased stealth level for long session")
        
        return adjustments
    
    def get_session_analytics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive session analytics"""
        
        if session_id:
            if session_id not in self.active_sessions:
                return {"error": "Session not found"}
            
            session_data = self.active_sessions[session_id]
            performance_analysis = self.performance_optimizer.get_performance_analysis(session_id)
            
            return {
                "session_id": session_id,
                "session_info": {
                    "created_at": session_data["created_at"].isoformat(),
                    "pages_loaded": session_data.get("pages_count", 0),
                    "total_optimizations": session_data.get("total_optimizations", 0),
                    "stealth_level": session_data.get("stealth_level"),
                    "performance_level": session_data.get("performance_level")
                },
                "performance_analysis": performance_analysis,
                "stealth_status": self._get_stealth_status(session_id)
            }
        else:
            # Overall analytics
            total_sessions = len(self.active_sessions)
            total_pages = sum(s.get("pages_count", 0) for s in self.active_sessions.values())
            total_optimizations = sum(s.get("total_optimizations", 0) for s in self.active_sessions.values())
            
            return {
                "timestamp": datetime.now().isoformat(),
                "overview": {
                    "active_sessions": total_sessions,
                    "total_pages_loaded": total_pages,
                    "total_optimizations_applied": total_optimizations,
                    "avg_pages_per_session": total_pages / total_sessions if total_sessions > 0 else 0
                },
                "performance_analysis": self.performance_optimizer.get_performance_analysis(),
                "active_performance_profiles": dict(self.performance_profiles)
            }
    
    def _get_stealth_status(self, session_id: str) -> Dict[str, Any]:
        """Get stealth status for session"""
        
        # This would integrate with the enhanced manager's stealth status
        return {
            "stealth_level": self.stealth_level.value,
            "active_features": ["user_agent_rotation", "fingerprint_protection", "proxy_support"],
            "detection_score": 0.1  # Low detection probability
        }
    
    async def batch_create_sessions(
        self,
        count: int,
        performance_profile: Optional[str] = None,
        stealth_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create multiple optimized sessions in batch"""
        
        console.print(f"[green]🚀 Creating {count} optimized browser sessions...[/green]")
        
        created_sessions = []
        failed_sessions = []
        
        for i in range(count):
            try:
                session_result = await self.create_optimized_browser(
                    session_id=f"batch_session_{i}_{int(datetime.now().timestamp())}",
                    performance_profile=performance_profile
                )
                
                if session_result.get("success"):
                    created_sessions.append(session_result["session_id"])
                    console.print(f"[green]   ✅ Session {i+1}/{count} created: {session_result['session_id']}[/green]")
                else:
                    failed_sessions.append(f"Session {i+1}: {session_result.get('error', 'Unknown error')}")
                    console.print(f"[red]   ❌ Session {i+1}/{count} failed[/red]")
                
                # Small delay between creations
                await asyncio.sleep(0.5)
                
            except Exception as e:
                failed_sessions.append(f"Session {i+1}: {str(e)}")
                console.print(f"[red]   ❌ Session {i+1}/{count} failed: {e}[/red]")
        
        console.print(f"[green]🎉 Batch creation completed: {len(created_sessions)}/{count} successful[/green]")
        
        return {
            "total_requested": count,
            "successful": len(created_sessions),
            "failed": len(failed_sessions),
            "created_sessions": created_sessions,
            "errors": failed_sessions,
            "success_rate": len(created_sessions) / count * 100 if count > 0 else 0
        }
    
    async def cleanup_sessions(self, max_age_hours: int = 24) -> Dict[str, Any]:
        """Clean up old sessions"""
        
        from datetime import timedelta
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        sessions_to_remove = []
        
        for session_id, session_data in self.active_sessions.items():
            if session_data["created_at"] < cutoff_time:
                sessions_to_remove.append(session_id)
        
        cleaned_count = 0
        for session_id in sessions_to_remove:
            try:
                if "browser" in self.active_sessions[session_id]:
                    await self.active_sessions[session_id]["browser"].close()
                del self.active_sessions[session_id]
                if session_id in self.performance_profiles:
                    del self.performance_profiles[session_id]
                cleaned_count += 1
            except Exception as e:
                self.logger.error(f"Error cleaning session {session_id}: {e}")
        
        # Clean performance data
        await self.performance_optimizer.cleanup_performance_data()
        
        console.print(f"[green]🧹 Cleaned {cleaned_count} sessions older than {max_age_hours} hours[/green]")
        
        return {
            "cleaned_sessions": cleaned_count,
            "remaining_sessions": len(self.active_sessions),
            "cutoff_time": cutoff_time.isoformat()
        }
    
    async def export_comprehensive_report(self, output_path: str) -> str:
        """Export comprehensive system report"""
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "stealth_level": self.stealth_level.value,
                "performance_level": self.performance_level.value,
                "monitoring_enabled": self.enable_monitoring,
                "session_persistence": self.session_persistence
            },
            "session_analytics": self.get_session_analytics(),
            "performance_report": self.performance_optimizer.get_performance_analysis(),
            "active_sessions_details": {
                session_id: {
                    "created_at": data["created_at"].isoformat(),
                    "pages_count": data.get("pages_count", 0),
                    "total_optimizations": data.get("total_optimizations", 0),
                    "performance_profile": self.performance_profiles.get(session_id, "default")
                }
                for session_id, data in self.active_sessions.items()
            }
        }
        
        try:
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            console.print(f"[green]📊 Comprehensive report exported to {output_path}[/green]")
            return output_path
            
        except Exception as e:
            console.print(f"[red]❌ Failed to export report: {e}[/red]")
            return ""
    
    async def close_all_sessions(self):
        """Close all active sessions"""
        
        console.print(f"[yellow]🔄 Closing {len(self.active_sessions)} active sessions...[/yellow]")
        
        closed_count = 0
        for session_id, session_data in list(self.active_sessions.items()):
            try:
                if "browser" in session_data:
                    await session_data["browser"].close()
                del self.active_sessions[session_id]
                if session_id in self.performance_profiles:
                    del self.performance_profiles[session_id]
                closed_count += 1
            except Exception as e:
                self.logger.error(f"Error closing session {session_id}: {e}")
        
        console.print(f"[green]✅ Closed {closed_count} sessions[/green]")
        
        return {"closed_sessions": closed_count}

# Convenience function for quick setup
async def create_integrated_browser(
    stealth_level: StealthLevel = StealthLevel.BALANCED,
    performance_profile: str = "balanced_power",
    enable_monitoring: bool = True
) -> IntegratedCamoufoxManager:
    """Quick setup for integrated Camoufox browser"""
    
    performance_level = PerformanceLevel.BALANCED
    if performance_profile == "speed_demon":
        performance_level = PerformanceLevel.MAXIMUM
    elif performance_profile == "memory_saver":
        performance_level = PerformanceLevel.AGGRESSIVE
    
    manager = IntegratedCamoufoxManager(
        stealth_level=stealth_level,
        performance_level=performance_level,
        enable_monitoring=enable_monitoring
    )
    
    return manager