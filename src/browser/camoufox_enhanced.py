#!/usr/bin/env python3
"""
Advanced Camoufox Browser Management for Enhanced Stealth and Performance
Provides sophisticated browser automation with advanced stealth capabilities
"""

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from rich.console import Console
    console = Console()
except ImportError:
    class MockConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = MockConsole()

@dataclass
class StealthProfile:
    """Configuration for stealth browsing behavior"""
    user_agent: str = ""
    viewport: Tuple[int, int] = (1920, 1080)
    timezone: str = "UTC"
    language: str = "en-US"
    webgl_vendor: str = "Google Inc."
    webgl_renderer: str = "ANGLE (Intel, Intel(R) HD Graphics 620 Direct3D11 vs_5_0 ps_5_0, D3D11-27.20.100.9466)"
    platform: str = "Win32"
    memory: int = 8  # GB
    cpu_cores: int = 4
    screen_resolution: Tuple[int, int] = (1920, 1080)
    color_depth: int = 24
    touch_support: bool = False
    webrtc_leak_protection: bool = True
    canvas_fingerprint_protection: bool = True
    audio_fingerprint_protection: bool = True
    font_fingerprint_protection: bool = True
    geolocation_spoofing: bool = True
    battery_spoofing: bool = True

@dataclass
class SessionConfig:
    """Configuration for browser session management"""
    session_id: str = ""
    profile_path: str = ""
    cookie_storage: str = ""
    local_storage_data: Dict[str, Any] = field(default_factory=dict)
    session_storage_data: Dict[str, Any] = field(default_factory=dict)
    download_directory: str = ""
    extensions: List[str] = field(default_factory=list)
    proxy_config: Optional[Dict[str, str]] = None
    request_interception: bool = True
    javascript_enabled: bool = True
    images_enabled: bool = True
    css_enabled: bool = True
    plugins_enabled: bool = False
    auto_cleanup: bool = True
    session_timeout: int = 3600  # seconds

@dataclass
class PerformanceConfig:
    """Configuration for browser performance optimization"""
    max_concurrent_tabs: int = 5
    memory_limit_mb: int = 2048
    cpu_limit_percent: int = 80
    network_cache_size_mb: int = 100
    disk_cache_size_mb: int = 500
    enable_hardware_acceleration: bool = True
    enable_background_throttling: bool = True
    enable_preload: bool = False
    enable_prefetch: bool = False
    page_load_timeout: int = 30
    script_timeout: int = 15
    enable_compression: bool = True
    enable_http2: bool = True

class EnhancedCamoufoxManager:
    """
    Enhanced Camoufox browser manager with advanced stealth and session management
    """
    
    def __init__(
        self,
        stealth_profile: Optional[StealthProfile] = None,
        session_config: Optional[SessionConfig] = None,
        performance_config: Optional[PerformanceConfig] = None,
        base_data_dir: str = "browser_data"
    ):
        self.stealth_profile = stealth_profile or StealthProfile()
        self.session_config = session_config or SessionConfig()
        self.performance_config = performance_config or PerformanceConfig()
        self.base_data_dir = Path(base_data_dir)
        self.base_data_dir.mkdir(exist_ok=True)
        
        self.active_sessions = {}
        self.session_pools = {}
        self.stealth_configs = {}
        self.performance_monitors = {}
        
        # Initialize stealth detection systems
        self._init_stealth_systems()
        
        console.print(f"[green]🦊 Enhanced Camoufox Manager initialized[/green]")
        console.print(f"[cyan]   • Data directory: {self.base_data_dir}[/cyan]")
        console.print(f"[cyan]   • Stealth features: enabled[/cyan]")
        console.print(f"[cyan]   • Session management: enabled[/cyan]")
    
    def _init_stealth_systems(self):
        """Initialize stealth detection and countermeasure systems"""
        
        # Common user agents for rotation
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ]
        
        # Viewport configurations for natural browsing
        self.viewport_configs = [
            (1920, 1080), (1366, 768), (1536, 864), (1440, 900),
            (1600, 900), (1280, 720), (1920, 1200), (2560, 1440)
        ]
        
        # Timezone configurations
        self.timezones = [
            "America/New_York", "Europe/London", "Asia/Tokyo",
            "America/Los_Angeles", "Europe/Berlin", "Asia/Shanghai",
            "Australia/Sydney", "America/Chicago"
        ]
        
        console.print("[green]🛡️ Stealth systems initialized[/green]")
    
    async def create_stealth_session(
        self,
        session_id: Optional[str] = None,
        custom_stealth: Optional[StealthProfile] = None,
        custom_performance: Optional[PerformanceConfig] = None
    ) -> str:
        """Create a new stealth browser session"""
        
        if not session_id:
            session_id = f"stealth_session_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Use custom configs or generate dynamic stealth profile
        stealth_profile = custom_stealth or self._generate_dynamic_stealth_profile()
        performance_config = custom_performance or self.performance_config
        
        # Create session directory
        session_dir = self.base_data_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Configure session
        session_config = SessionConfig(
            session_id=session_id,
            profile_path=str(session_dir / "profile"),
            cookie_storage=str(session_dir / "cookies.json"),
            download_directory=str(session_dir / "downloads"),
            auto_cleanup=True
        )
        
        # Store session configurations
        self.active_sessions[session_id] = {
            'stealth_profile': stealth_profile,
            'session_config': session_config,
            'performance_config': performance_config,
            'created_at': datetime.now(),
            'last_activity': datetime.now(),
            'page_count': 0,
            'request_count': 0,
            'status': 'created'
        }
        
        console.print(f"[green]✅ Created stealth session: {session_id}[/green]")
        console.print(f"[cyan]   • User Agent: {stealth_profile.user_agent[:50]}...[/cyan]")
        console.print(f"[cyan]   • Viewport: {stealth_profile.viewport}[/cyan]")
        console.print(f"[cyan]   • Timezone: {stealth_profile.timezone}[/cyan]")
        
        return session_id
    
    def _generate_dynamic_stealth_profile(self) -> StealthProfile:
        """Generate a dynamic stealth profile with realistic characteristics"""
        
        return StealthProfile(
            user_agent=random.choice(self.user_agents),
            viewport=random.choice(self.viewport_configs),
            timezone=random.choice(self.timezones),
            language=random.choice(["en-US", "en-GB", "de-DE", "fr-FR", "es-ES"]),
            platform=random.choice(["Win32", "MacIntel", "Linux x86_64"]),
            memory=random.choice([4, 8, 16, 32]),
            cpu_cores=random.choice([2, 4, 6, 8, 12, 16]),
            screen_resolution=random.choice(self.viewport_configs),
            color_depth=random.choice([24, 32]),
            touch_support=random.choice([True, False]),
            webrtc_leak_protection=True,
            canvas_fingerprint_protection=True,
            audio_fingerprint_protection=True,
            font_fingerprint_protection=True,
            geolocation_spoofing=True,
            battery_spoofing=True
        )
    
    async def get_camoufox_launch_options(self, session_id: str) -> Dict[str, Any]:
        """Get optimized Camoufox launch options for a session"""
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session_data = self.active_sessions[session_id]
        stealth_profile = session_data['stealth_profile']
        session_config = session_data['session_config']
        performance_config = session_data['performance_config']
        
        # Base launch options
        launch_options = {
            'headless': False,  # Camoufox works best in non-headless mode
            'firefox_user_prefs': {},
            'user_data_dir': session_config.profile_path,
            'args': []
        }
        
        # Stealth configurations
        launch_options['firefox_user_prefs'].update({
            # User agent and platform spoofing
            'general.useragent.override': stealth_profile.user_agent,
            'general.platform.override': stealth_profile.platform,
            
            # Timezone spoofing
            'intl.regional_prefs.use_os_locales': False,
            'javascript.use_us_english_locale': True,
            
            # WebGL spoofing
            'webgl.renderer-string-override': stealth_profile.webgl_renderer,
            'webgl.vendor-string-override': stealth_profile.webgl_vendor,
            
            # Canvas fingerprint protection
            'privacy.resistFingerprinting.canvas': stealth_profile.canvas_fingerprint_protection,
            'privacy.resistFingerprinting.audio': stealth_profile.audio_fingerprint_protection,
            
            # WebRTC leak protection
            'media.peerconnection.enabled': not stealth_profile.webrtc_leak_protection,
            'media.peerconnection.ice.default_address_only': stealth_profile.webrtc_leak_protection,
            
            # Font fingerprint protection
            'gfx.downloadable_fonts.enabled': not stealth_profile.font_fingerprint_protection,
            
            # Geolocation spoofing
            'geo.enabled': stealth_profile.geolocation_spoofing,
            'geo.provider.use_gpsd': False,
            
            # Battery spoofing
            'dom.battery.enabled': not stealth_profile.battery_spoofing,
            
            # Performance optimizations
            'browser.tabs.remote.autostart': True,
            'layers.acceleration.force-enabled': performance_config.enable_hardware_acceleration,
            'browser.sessionstore.max_tabs_undo': 0,
            'browser.sessionstore.max_windows_undo': 0,
            'browser.cache.memory.capacity': performance_config.network_cache_size_mb * 1024,
            'browser.cache.disk.capacity': performance_config.disk_cache_size_mb * 1024,
            
            # Security and privacy
            'privacy.trackingprotection.enabled': True,
            'privacy.donottrackheader.enabled': True,
            'network.cookie.cookieBehavior': 1,  # Block third-party cookies
            'dom.webnotifications.enabled': False,
            'media.autoplay.default': 2,  # Block autoplay
            
            # Resource loading optimization
            'network.http.max-connections': 30,
            'network.http.max-connections-per-server': 6,
            'network.http.max-persistent-connections-per-server': 4,
            'network.http.pipelining': True,
            'network.http.pipelining.maxrequests': 8,
            'network.http.request.timeout': performance_config.page_load_timeout,
            
            # JavaScript optimization
            'javascript.options.mem.high_water_mark': 128,
            'javascript.options.mem.max': 1024 * 1024 * performance_config.memory_limit_mb // 4,
            
            # Image and media handling
            'permissions.default.image': 1 if performance_config.enable_hardware_acceleration else 2,
            'media.cache_size': performance_config.network_cache_size_mb * 1024 // 2,
        })
        
        # Command line arguments
        launch_options['args'].extend([
            '--no-first-run',
            '--disable-default-apps',
            '--disable-popup-blocking',
            '--disable-translate',
            f'--memory-pressure-off',
            f'--max_old_space_size={performance_config.memory_limit_mb}',
            '--disable-background-timer-throttling',
            '--disable-backgrounding-occluded-windows',
            '--disable-renderer-backgrounding',
        ])
        
        # Proxy configuration
        if session_config.proxy_config:
            proxy_config = session_config.proxy_config
            launch_options['firefox_user_prefs'].update({
                'network.proxy.type': 1,
                'network.proxy.http': proxy_config.get('host', ''),
                'network.proxy.http_port': int(proxy_config.get('port', 8080)),
                'network.proxy.ssl': proxy_config.get('host', ''),
                'network.proxy.ssl_port': int(proxy_config.get('port', 8080)),
            })
            
            if proxy_config.get('username') and proxy_config.get('password'):
                launch_options['firefox_user_prefs'].update({
                    'network.proxy.username': proxy_config['username'],
                    'network.proxy.password': proxy_config['password'],
                })
        
        console.print(f"[green]🚀 Generated launch options for session {session_id}[/green]")
        return launch_options
    
    async def apply_stealth_scripts(self, page, session_id: str) -> None:
        """Apply stealth scripts to a page to avoid detection"""
        
        if session_id not in self.active_sessions:
            return
        
        stealth_profile = self.active_sessions[session_id]['stealth_profile']
        
        # JavaScript to spoof various browser properties
        stealth_script = f"""
        // Override navigator properties
        Object.defineProperty(navigator, 'userAgent', {{
            get: () => '{stealth_profile.user_agent}'
        }});
        
        Object.defineProperty(navigator, 'platform', {{
            get: () => '{stealth_profile.platform}'
        }});
        
        Object.defineProperty(navigator, 'language', {{
            get: () => '{stealth_profile.language}'
        }});
        
        Object.defineProperty(navigator, 'languages', {{
            get: () => ['{stealth_profile.language}']
        }});
        
        Object.defineProperty(navigator, 'hardwareConcurrency', {{
            get: () => {stealth_profile.cpu_cores}
        }});
        
        Object.defineProperty(navigator, 'deviceMemory', {{
            get: () => {stealth_profile.memory}
        }});
        
        Object.defineProperty(navigator, 'maxTouchPoints', {{
            get: () => {1 if stealth_profile.touch_support else 0}
        }});
        
        // Override screen properties
        Object.defineProperty(screen, 'width', {{
            get: () => {stealth_profile.screen_resolution[0]}
        }});
        
        Object.defineProperty(screen, 'height', {{
            get: () => {stealth_profile.screen_resolution[1]}
        }});
        
        Object.defineProperty(screen, 'colorDepth', {{
            get: () => {stealth_profile.color_depth}
        }});
        
        // Override timezone
        Date.prototype.getTimezoneOffset = function() {{
            return -{self._get_timezone_offset(stealth_profile.timezone)};
        }};
        
        // WebGL spoofing
        const originalGetParameter = WebGLRenderingContext.prototype.getParameter;
        WebGLRenderingContext.prototype.getParameter = function(parameter) {{
            if (parameter === 37445) return '{stealth_profile.webgl_vendor}';
            if (parameter === 37446) return '{stealth_profile.webgl_renderer}';
            return originalGetParameter.call(this, parameter);
        }};
        
        // Canvas fingerprint protection
        if ({str(stealth_profile.canvas_fingerprint_protection).lower()}) {{
            const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
            HTMLCanvasElement.prototype.toDataURL = function() {{
                const context = this.getContext('2d');
                if (context) {{
                    const imageData = context.getImageData(0, 0, this.width, this.height);
                    for (let i = 0; i < imageData.data.length; i += 4) {{
                        imageData.data[i] += Math.floor(Math.random() * 3) - 1;
                        imageData.data[i + 1] += Math.floor(Math.random() * 3) - 1;
                        imageData.data[i + 2] += Math.floor(Math.random() * 3) - 1;
                    }}
                    context.putImageData(imageData, 0, 0);
                }}
                return originalToDataURL.apply(this, arguments);
            }};
        }}
        
        // Audio fingerprint protection
        if ({str(stealth_profile.audio_fingerprint_protection).lower()}) {{
            const originalGetChannelData = AudioBuffer.prototype.getChannelData;
            AudioBuffer.prototype.getChannelData = function() {{
                const data = originalGetChannelData.apply(this, arguments);
                for (let i = 0; i < data.length; i++) {{
                    data[i] += (Math.random() - 0.5) * 0.0001;
                }}
                return data;
            }};
        }}
        
        // Battery spoofing
        if ({str(stealth_profile.battery_spoofing).lower()}) {{
            Object.defineProperty(navigator, 'getBattery', {{
                get: () => undefined
            }});
        }}
        
        // Geolocation spoofing
        if ({str(stealth_profile.geolocation_spoofing).lower()}) {{
            Object.defineProperty(navigator, 'geolocation', {{
                get: () => undefined
            }});
        }}
        
        // Remove automation indicators
        Object.defineProperty(navigator, 'webdriver', {{
            get: () => undefined
        }});
        
        delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
        delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
        delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;
        
        // Override plugin detection
        Object.defineProperty(navigator, 'plugins', {{
            get: () => {{
                return [
                    {{
                        name: 'Chrome PDF Plugin',
                        filename: 'internal-pdf-viewer',
                        description: 'Portable Document Format'
                    }},
                    {{
                        name: 'Chrome PDF Viewer',
                        filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai',
                        description: ''
                    }}
                ];
            }}
        }});
        
        console.log('🛡️ Stealth scripts applied successfully');
        """
        
        try:
            await page.evaluate(stealth_script)
            console.print(f"[green]🛡️ Applied stealth scripts for session {session_id}[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Failed to apply some stealth scripts: {e}[/yellow]")
    
    def _get_timezone_offset(self, timezone: str) -> int:
        """Get timezone offset in minutes"""
        timezone_offsets = {
            'UTC': 0,
            'America/New_York': 300,  # EST
            'America/Los_Angeles': 480,  # PST
            'Europe/London': 0,  # GMT
            'Europe/Berlin': -60,  # CET
            'Asia/Tokyo': -540,  # JST
            'Asia/Shanghai': -480,  # CST
            'Australia/Sydney': -660,  # AEDT
            'America/Chicago': 360,  # CST
        }
        return timezone_offsets.get(timezone, 0)
    
    async def manage_session_resources(self, session_id: str) -> Dict[str, Any]:
        """Monitor and manage session resources"""
        
        if session_id not in self.active_sessions:
            return {'error': 'Session not found'}
        
        session_data = self.active_sessions[session_id]
        performance_config = session_data['performance_config']
        
        # Update session activity
        session_data['last_activity'] = datetime.now()
        
        # Resource monitoring data
        resource_data = {
            'session_id': session_id,
            'uptime': (datetime.now() - session_data['created_at']).total_seconds(),
            'page_count': session_data['page_count'],
            'request_count': session_data['request_count'],
            'memory_limit_mb': performance_config.memory_limit_mb,
            'cpu_limit_percent': performance_config.cpu_limit_percent,
            'status': session_data['status'],
            'last_activity': session_data['last_activity'].isoformat()
        }
        
        # Check for session timeout
        if session_data['session_config'].session_timeout > 0:
            timeout_delta = timedelta(seconds=session_data['session_config'].session_timeout)
            if datetime.now() - session_data['last_activity'] > timeout_delta:
                await self.cleanup_session(session_id)
                resource_data['status'] = 'timed_out'
        
        return resource_data
    
    async def rotate_stealth_profile(self, session_id: str) -> bool:
        """Rotate stealth profile for a session to avoid detection"""
        
        if session_id not in self.active_sessions:
            return False
        
        # Generate new stealth profile
        new_profile = self._generate_dynamic_stealth_profile()
        self.active_sessions[session_id]['stealth_profile'] = new_profile
        
        console.print(f"[green]🔄 Rotated stealth profile for session {session_id}[/green]")
        console.print(f"[cyan]   • New User Agent: {new_profile.user_agent[:50]}...[/cyan]")
        console.print(f"[cyan]   • New Viewport: {new_profile.viewport}[/cyan]")
        console.print(f"[cyan]   • New Timezone: {new_profile.timezone}[/cyan]")
        
        return True
    
    async def create_session_pool(
        self,
        pool_name: str,
        pool_size: int = 3,
        rotation_interval: int = 1800  # 30 minutes
    ) -> str:
        """Create a pool of rotating browser sessions for high-volume operations"""
        
        session_ids = []
        for i in range(pool_size):
            session_id = await self.create_stealth_session(
                session_id=f"{pool_name}_session_{i}"
            )
            session_ids.append(session_id)
        
        pool_config = {
            'name': pool_name,
            'session_ids': session_ids,
            'current_index': 0,
            'rotation_interval': rotation_interval,
            'last_rotation': datetime.now(),
            'created_at': datetime.now(),
            'total_requests': 0
        }
        
        self.session_pools[pool_name] = pool_config
        
        console.print(f"[green]🏊 Created session pool '{pool_name}' with {pool_size} sessions[/green]")
        return pool_name
    
    async def get_pool_session(self, pool_name: str) -> Optional[str]:
        """Get the next available session from a pool"""
        
        if pool_name not in self.session_pools:
            return None
        
        pool = self.session_pools[pool_name]
        
        # Check if rotation is needed
        if datetime.now() - pool['last_rotation'] > timedelta(seconds=pool['rotation_interval']):
            await self._rotate_pool_sessions(pool_name)
        
        # Get current session
        current_session = pool['session_ids'][pool['current_index']]
        
        # Move to next session for next request
        pool['current_index'] = (pool['current_index'] + 1) % len(pool['session_ids'])
        pool['total_requests'] += 1
        
        return current_session
    
    async def _rotate_pool_sessions(self, pool_name: str):
        """Rotate stealth profiles for all sessions in a pool"""
        
        if pool_name not in self.session_pools:
            return
        
        pool = self.session_pools[pool_name]
        
        for session_id in pool['session_ids']:
            await self.rotate_stealth_profile(session_id)
        
        pool['last_rotation'] = datetime.now()
        console.print(f"[green]🔄 Rotated all sessions in pool '{pool_name}'[/green]")
    
    async def cleanup_session(self, session_id: str) -> bool:
        """Clean up a browser session and its resources"""
        
        if session_id not in self.active_sessions:
            return False
        
        session_data = self.active_sessions[session_id]
        session_config = session_data['session_config']
        
        try:
            # Clean up session files if auto_cleanup is enabled
            if session_config.auto_cleanup:
                session_dir = Path(session_config.profile_path).parent
                if session_dir.exists():
                    import shutil
                    shutil.rmtree(session_dir, ignore_errors=True)
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            console.print(f"[green]🧹 Cleaned up session {session_id}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to cleanup session {session_id}: {e}[/red]")
            return False
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up all expired sessions"""
        
        cleaned_count = 0
        expired_sessions = []
        
        for session_id, session_data in self.active_sessions.items():
            session_config = session_data['session_config']
            
            if session_config.session_timeout > 0:
                timeout_delta = timedelta(seconds=session_config.session_timeout)
                if datetime.now() - session_data['last_activity'] > timeout_delta:
                    expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            if await self.cleanup_session(session_id):
                cleaned_count += 1
        
        console.print(f"[green]🧹 Cleaned up {cleaned_count} expired sessions[/green]")
        return cleaned_count
    
    async def get_session_statistics(self) -> Dict[str, Any]:
        """Get comprehensive session statistics"""
        
        active_count = len(self.active_sessions)
        pool_count = len(self.session_pools)
        
        # Calculate total requests across all sessions
        total_requests = sum(
            session_data['request_count'] 
            for session_data in self.active_sessions.values()
        )
        
        # Pool statistics
        pool_stats = {}
        for pool_name, pool_data in self.session_pools.items():
            pool_stats[pool_name] = {
                'session_count': len(pool_data['session_ids']),
                'total_requests': pool_data['total_requests'],
                'last_rotation': pool_data['last_rotation'].isoformat(),
                'created_at': pool_data['created_at'].isoformat()
            }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'active_sessions': active_count,
            'session_pools': pool_count,
            'total_requests': total_requests,
            'pool_statistics': pool_stats,
            'session_details': {
                session_id: {
                    'created_at': data['created_at'].isoformat(),
                    'last_activity': data['last_activity'].isoformat(),
                    'page_count': data['page_count'],
                    'request_count': data['request_count'],
                    'status': data['status']
                }
                for session_id, data in self.active_sessions.items()
            }
        }
    
    async def export_session_config(self, session_id: str, export_path: str) -> bool:
        """Export session configuration for backup or replication"""
        
        if session_id not in self.active_sessions:
            return False
        
        session_data = self.active_sessions[session_id]
        
        export_data = {
            'session_id': session_id,
            'stealth_profile': {
                'user_agent': session_data['stealth_profile'].user_agent,
                'viewport': session_data['stealth_profile'].viewport,
                'timezone': session_data['stealth_profile'].timezone,
                'language': session_data['stealth_profile'].language,
                'platform': session_data['stealth_profile'].platform,
                'memory': session_data['stealth_profile'].memory,
                'cpu_cores': session_data['stealth_profile'].cpu_cores,
                'screen_resolution': session_data['stealth_profile'].screen_resolution,
                'color_depth': session_data['stealth_profile'].color_depth,
                'touch_support': session_data['stealth_profile'].touch_support,
            },
            'session_config': {
                'download_directory': session_data['session_config'].download_directory,
                'proxy_config': session_data['session_config'].proxy_config,
                'session_timeout': session_data['session_config'].session_timeout,
            },
            'performance_config': {
                'max_concurrent_tabs': session_data['performance_config'].max_concurrent_tabs,
                'memory_limit_mb': session_data['performance_config'].memory_limit_mb,
                'cpu_limit_percent': session_data['performance_config'].cpu_limit_percent,
                'page_load_timeout': session_data['performance_config'].page_load_timeout,
            },
            'statistics': {
                'created_at': session_data['created_at'].isoformat(),
                'page_count': session_data['page_count'],
                'request_count': session_data['request_count'],
            }
        }
        
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            console.print(f"[green]✅ Exported session config to {export_path}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to export session config: {e}[/red]")
            return False