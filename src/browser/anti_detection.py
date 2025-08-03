#!/usr/bin/env python3
"""
Enhanced Browser Automation with Advanced Anti-Bot Detection Bypass
Integrates Camoufox with sophisticated CAPTCHA solving and human behavior simulation
"""

import asyncio
import json
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import re

try:
    from camoufox import AsyncCamoufox
    CAMOUFOX_AVAILABLE = True
except ImportError:
    print("⚠️ Camoufox not available")
    CAMOUFOX_AVAILABLE = False
    AsyncCamoufox = None

try:
    from rich.console import Console
    console = Console()
except ImportError:
    class MockConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = MockConsole()

from .camoufox_enhanced import EnhancedCamoufoxManager, StealthProfile, SessionConfig, PerformanceConfig
from .captcha_solver import AdvancedCaptchaSolver, CaptchaSolverIntegration, HumanBehaviorProfile

class AntiDetectionBrowserManager:
    """
    Comprehensive browser automation system with advanced anti-detection capabilities
    """
    
    def __init__(self, 
                 base_data_dir: str = "browser_data",
                 llm_endpoint: str = "http://localhost:11434/api/generate",
                 llm_model: str = "granite3.3:8b"):
        
        if not CAMOUFOX_AVAILABLE:
            raise RuntimeError("Camoufox is required for anti-detection browser management")
        
        self.base_data_dir = Path(base_data_dir)
        self.base_data_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.camoufox_manager = EnhancedCamoufoxManager(base_data_dir=str(self.base_data_dir))
        self.captcha_solver = AdvancedCaptchaSolver(
            llm_endpoint=llm_endpoint,
            llm_model=llm_model
        )
        
        # Active browser instances
        self.active_browsers = {}
        self.active_pages = {}
        
        # Detection countermeasures
        self.detection_patterns = {
            'cloudflare_challenge': [
                'Checking your browser before accessing',
                'DDoS protection by Cloudflare',
                'cf-challenge-form',
                'challenge-form'
            ],
            'bot_detection': [
                'bot detected',
                'automated traffic',
                'suspicious activity',
                'verify you are human'
            ],
            'rate_limiting': [
                'too many requests',
                'rate limited',
                'slow down',
                'wait before continuing'
            ],
            'access_denied': [
                'access denied',
                'forbidden',
                '403 Forbidden',
                'blocked'
            ]
        }
        
        console.print("[green]🤖 Anti-Detection Browser Manager initialized[/green]")
        console.print(f"[cyan]   • Data directory: {self.base_data_dir}[/cyan]")
        console.print(f"[cyan]   • LLM Model: {llm_model}[/cyan]")
        console.print(f"[cyan]   • CAPTCHA Solver: enabled[/cyan]")
    
    async def create_stealth_browser(self,
                                   session_id: Optional[str] = None,
                                   headless: bool = False,
                                   proxy_config: Optional[Dict[str, str]] = None,
                                   custom_stealth: Optional[StealthProfile] = None) -> str:
        """Create a new stealth browser instance with advanced anti-detection"""
        
        if not session_id:
            session_id = f"stealth_browser_{int(time.time())}_{random.randint(1000, 9999)}"
        
        try:
            # Create session in Camoufox manager
            await self.camoufox_manager.create_stealth_session(
                session_id=session_id,
                custom_stealth=custom_stealth
            )
            
            # Get optimized launch options
            launch_options = await self.camoufox_manager.get_camoufox_launch_options(session_id)
            
            # Enhanced Camoufox configuration for anti-detection
            camoufox_config = {
                'headless': headless,
                'disable_blink_features': 'AutomationControlled',
                'disable_features': 'VizDisplayCompositor',
                'block_images': True,  # For performance
                'block_media': True,   # For performance
                'geoip': True if proxy_config else False,
                'proxy': proxy_config,
                'fingerprint': {
                    'os': ['windows', 'macos', 'linux'],
                    'screen': {'max_width': 1920, 'max_height': 1080}
                },
                'addons': ['ublock_origin'],  # Built-in ad blocker
                'exclude_addons': [],
                'config': self._get_enhanced_fingerprint_config(session_id)
            }
            
            # Launch Camoufox browser
            if AsyncCamoufox is None:
                raise RuntimeError("AsyncCamoufox is not available")
            browser = AsyncCamoufox(**camoufox_config)
            await browser.start()
            
            # Store browser instance
            self.active_browsers[session_id] = {
                'browser': browser,
                'created_at': datetime.now(),
                'config': camoufox_config,
                'session_data': self.camoufox_manager.active_sessions[session_id],
                'pages': {}
            }
            
            console.print(f"[green]🦊 Created stealth browser: {session_id}[/green]")
            console.print(f"[cyan]   • Headless: {headless}[/cyan]")
            console.print(f"[cyan]   • Proxy: {'enabled' if proxy_config else 'disabled'}[/cyan]")
            console.print(f"[cyan]   • Anti-detection: enabled[/cyan]")
            
            return session_id
            
        except Exception as e:
            console.print(f"[red]❌ Failed to create stealth browser: {e}[/red]")
            raise
    
    def _get_enhanced_fingerprint_config(self, session_id: str) -> Dict[str, Any]:
        """Generate enhanced fingerprint configuration for maximum stealth"""
        
        session_data = self.camoufox_manager.active_sessions[session_id]
        stealth_profile = session_data['stealth_profile']
        
        config = {
            # Navigator properties
            'navigator.userAgent': stealth_profile.user_agent,
            'navigator.platform': stealth_profile.platform,
            'navigator.language': stealth_profile.language,
            'navigator.languages': [stealth_profile.language],
            'navigator.hardwareConcurrency': stealth_profile.cpu_cores,
            'navigator.deviceMemory': stealth_profile.memory,
            'navigator.maxTouchPoints': 1 if stealth_profile.touch_support else 0,
            
            # Screen properties
            'screen.width': stealth_profile.screen_resolution[0],
            'screen.height': stealth_profile.screen_resolution[1],
            'screen.colorDepth': stealth_profile.color_depth,
            
            # Window properties
            'window.innerWidth': stealth_profile.viewport[0],
            'window.innerHeight': stealth_profile.viewport[1],
            'window.outerWidth': stealth_profile.viewport[0],
            'window.outerHeight': stealth_profile.viewport[1] + random.randint(50, 100),
            
            # WebGL properties
            'webgl.vendor': stealth_profile.webgl_vendor,
            'webgl.renderer': stealth_profile.webgl_renderer,
            
            # Timezone
            'timezone': stealth_profile.timezone,
            
            # Audio context spoofing
            'AudioContext.sampleRate': random.choice([44100, 48000]),
            'AudioContext.outputLatency': random.uniform(0.01, 0.05),
            'AudioContext.maxChannelCount': random.choice([2, 6, 8]),
            
            # Canvas anti-fingerprinting
            'canvas.aaOffset': random.randint(1, 3),
            'canvas.aaCapOffset': random.choice([True, False]),
            
            # Font configuration
            'fonts': self._get_realistic_font_list(stealth_profile.platform),
            
            # Media devices
            'mediaDevices.cameras': random.randint(0, 2),
            'mediaDevices.microphones': random.randint(0, 2),
            'mediaDevices.speakers': random.randint(1, 3),
            
            # WebRTC configuration
            'webrtc.ipv4': self._generate_realistic_ip(),
            'webrtc.ipv6': self._generate_realistic_ipv6(),
            
            # Battery API spoofing
            'battery.charging': random.choice([True, False]),
            'battery.level': random.uniform(0.2, 0.95),
            'battery.chargingTime': random.randint(3600, 14400) if random.choice([True, False]) else float('inf'),
            'battery.dischargingTime': random.randint(7200, 28800),
        }
        
        return config
    
    def _get_realistic_font_list(self, platform: str) -> List[str]:
        """Get realistic font list based on platform"""
        
        base_fonts = ["Arial", "Times New Roman", "Helvetica", "Courier New"]
        
        if platform.lower() in ['win32', 'windows']:
            return base_fonts + [
                "Calibri", "Cambria", "Segoe UI", "Tahoma", "Verdana",
                "Comic Sans MS", "Impact", "Lucida Console", "Georgia"
            ]
        elif platform.lower() in ['macintel', 'macos']:
            return base_fonts + [
                "San Francisco", "Helvetica Neue", "Lucida Grande",
                "Monaco", "Menlo", "Avenir", "Optima"
            ]
        else:  # Linux
            return base_fonts + [
                "Ubuntu", "Liberation Sans", "DejaVu Sans", "Droid Sans",
                "Noto Sans", "Source Sans Pro"
            ]
    
    def _generate_realistic_ip(self) -> str:
        """Generate a realistic IPv4 address"""
        # Generate realistic private/public IP ranges
        if random.choice([True, False]):
            # Private IP ranges
            ranges = [
                ((10, 0, 0, 0), (10, 255, 255, 255)),      # 10.0.0.0/8
                ((172, 16, 0, 0), (172, 31, 255, 255)),    # 172.16.0.0/12
                ((192, 168, 0, 0), (192, 168, 255, 255)),  # 192.168.0.0/16
            ]
            start, end = random.choice(ranges)
            return f"{random.randint(start[0], end[0])}.{random.randint(start[1], end[1])}.{random.randint(start[2], end[2])}.{random.randint(start[3], end[3])}"
        else:
            # Public IP (avoiding reserved ranges)
            return f"{random.randint(1, 223)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
    
    def _generate_realistic_ipv6(self) -> str:
        """Generate a realistic IPv6 address"""
        return ":".join([f"{random.randint(0, 65535):04x}" for _ in range(8)])
    
    async def create_stealth_page(self, session_id: str, page_id: Optional[str] = None) -> str:
        """Create a new page with stealth configuration and CAPTCHA handling"""
        
        if session_id not in self.active_browsers:
            raise ValueError(f"Browser session {session_id} not found")
        
        if not page_id:
            page_id = f"page_{int(time.time())}_{random.randint(100, 999)}"
        
        try:
            browser_data = self.active_browsers[session_id]
            browser = browser_data['browser']
            
            # Create new page
            page = await browser.new_page()
            
            # Apply stealth scripts
            await self.camoufox_manager.apply_stealth_scripts(page, session_id)
            
            # Configure page for anti-detection
            await self._configure_page_anti_detection(page, session_id)
            
            # Store page reference
            browser_data['pages'][page_id] = {
                'page': page,
                'created_at': datetime.now(),
                'captcha_solver': CaptchaSolverIntegration(self.camoufox_manager, {}),
                'navigation_count': 0,
                'last_activity': datetime.now()
            }
            
            self.active_pages[page_id] = {
                'session_id': session_id,
                'page': page,
                'captcha_solver': CaptchaSolverIntegration(self.camoufox_manager, {})
            }
            
            console.print(f"[green]📄 Created stealth page: {page_id}[/green]")
            return page_id
            
        except Exception as e:
            console.print(f"[red]❌ Failed to create stealth page: {e}[/red]")
            raise
    
    async def _configure_page_anti_detection(self, page, session_id: str):
        """Configure page-level anti-detection measures"""
        
        try:
            # Enhanced stealth script injection
            await page.add_init_script("""
                // Enhanced webdriver detection removal
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                    configurable: true
                });
                
                // Remove automation indicators
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_JSON;
                delete window.cdc_adoQpoasnfa76pfcZLmcfl_Object;
                
                // Override permission queries
                const originalQuery = navigator.permissions.query;
                navigator.permissions.query = function(parameters) {
                    return parameters.name === 'notifications' ?
                        Promise.resolve({state: Notification.permission}) :
                        originalQuery.call(navigator.permissions, parameters);
                };
                
                // Spoof plugin information
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [
                        {
                            name: 'Chrome PDF Plugin',
                            filename: 'internal-pdf-viewer',
                            description: 'Portable Document Format',
                            length: 1
                        },
                        {
                            name: 'Chrome PDF Viewer',
                            filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai',
                            description: '',
                            length: 1
                        }
                    ]
                });
                
                // Enhanced iframe detection blocking
                const originalCreateElement = document.createElement;
                document.createElement = function(tagName) {
                    const element = originalCreateElement.call(document, tagName);
                    if (tagName.toLowerCase() === 'iframe') {
                        element.addEventListener('load', function() {
                            try {
                                const iframeDoc = this.contentDocument || this.contentWindow.document;
                                if (iframeDoc) {
                                    // Apply same protections to iframe
                                    Object.defineProperty(iframeDoc.defaultView.navigator, 'webdriver', {
                                        get: () => undefined
                                    });
                                }
                            } catch (e) {
                                // Cross-origin iframe, ignore
                            }
                        });
                    }
                    return element;
                };
                
                console.log('🛡️ Enhanced anti-detection measures applied');
            """)
            
            # Set realistic request headers
            await page.set_extra_http_headers({
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept-Language': 'en-US,en;q=0.9',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Upgrade-Insecure-Requests': '1'
            })
            
            # Configure viewport with slight randomization
            session_data = self.camoufox_manager.active_sessions[session_id]
            stealth_profile = session_data['stealth_profile']
            
            viewport = {
                'width': stealth_profile.viewport[0] + random.randint(-10, 10),
                'height': stealth_profile.viewport[1] + random.randint(-10, 10),
                'deviceScaleFactor': random.choice([1.0, 1.25, 1.5, 2.0])
            }
            await page.set_viewport_size(viewport)
            
            console.print(f"[green]🛡️ Applied anti-detection configuration[/green]")
            
        except Exception as e:
            console.print(f"[yellow]⚠️ Some anti-detection measures failed: {e}[/yellow]")
    
    async def navigate_with_protection(self, 
                                     page_id: str, 
                                     url: str, 
                                     wait_until: str = 'domcontentloaded',
                                     max_retries: int = 3,
                                     handle_captcha: bool = True) -> bool:
        """Navigate to URL with comprehensive protection against detection and CAPTCHAs"""
        
        if page_id not in self.active_pages:
            raise ValueError(f"Page {page_id} not found")
        
        page_data = self.active_pages[page_id]
        page = page_data['page']
        session_id = page_data['session_id']
        
        for attempt in range(max_retries):
            try:
                console.print(f"[blue]🌐 Navigating to {url} (attempt {attempt + 1})[/blue]")
                
                # Pre-navigation behavior simulation
                await self._simulate_pre_navigation_behavior(page)
                
                # Navigate with timeout
                try:
                    await page.goto(url, wait_until=wait_until, timeout=30000)
                except Exception as nav_error:
                    console.print(f"[yellow]⚠️ Navigation timeout or error: {nav_error}[/yellow]")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(random.uniform(2, 5))
                        continue
                    else:
                        return False
                
                # Wait for page to stabilize
                await asyncio.sleep(random.uniform(1, 3))
                
                # Check for detection and handle accordingly
                detection_result = await self._check_for_detection(page)
                
                if detection_result['detected']:
                    console.print(f"[yellow]🚨 Detection found: {detection_result['type']}[/yellow]")
                    
                    if detection_result['type'] == 'captcha' and handle_captcha:
                        # Handle CAPTCHA
                        success = await self._handle_captcha_detection(page, page_id)
                        if success:
                            console.print("[green]✅ CAPTCHA handled successfully[/green]")
                            return True
                        else:
                            console.print("[red]❌ CAPTCHA handling failed[/red]")
                    elif detection_result['type'] == 'cloudflare':
                        # Handle Cloudflare challenge
                        success = await self._handle_cloudflare_detection(page, page_id)
                        if success:
                            console.print("[green]✅ Cloudflare challenge resolved[/green]")
                            return True
                        else:
                            console.print("[red]❌ Cloudflare challenge failed[/red]")
                    elif detection_result['type'] == 'rate_limiting':
                        # Handle rate limiting
                        wait_time = random.uniform(30, 120)  # Wait 30-120 seconds
                        console.print(f"[yellow]⏳ Rate limited, waiting {wait_time:.1f} seconds[/yellow]")
                        await asyncio.sleep(wait_time)
                    
                    # Retry if not successful
                    if attempt < max_retries - 1:
                        await asyncio.sleep(random.uniform(3, 8))
                        continue
                    else:
                        return False
                else:
                    # Successful navigation
                    console.print("[green]✅ Navigation successful[/green]")
                    
                    # Update activity tracking
                    browser_data = self.active_browsers[session_id]
                    browser_data['pages'][page_id]['navigation_count'] += 1
                    browser_data['pages'][page_id]['last_activity'] = datetime.now()
                    
                    # Post-navigation behavior simulation
                    await self._simulate_post_navigation_behavior(page)
                    
                    return True
                    
            except Exception as e:
                console.print(f"[red]❌ Navigation attempt {attempt + 1} failed: {e}[/red]")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(random.uniform(5, 10))
                else:
                    return False
        
        return False
    
    async def _simulate_pre_navigation_behavior(self, page):
        """Simulate realistic human behavior before navigation"""
        
        try:
            # Random delay to simulate thinking/preparation
            await asyncio.sleep(random.uniform(0.5, 2.0))
            
            # Simulate mouse movement (if not headless)
            viewport = await page.viewport_size()
            if viewport:
                # Move mouse to random position
                await page.mouse.move(
                    random.uniform(50, viewport['width'] - 50),
                    random.uniform(50, viewport['height'] - 50)
                )
                
                # Brief pause
                await asyncio.sleep(random.uniform(0.2, 0.8))
                
        except Exception as e:
            console.print(f"[yellow]⚠️ Pre-navigation behavior simulation failed: {e}[/yellow]")
    
    async def _simulate_post_navigation_behavior(self, page):
        """Simulate realistic human behavior after navigation"""
        
        try:
            # Wait a bit for page to load completely
            await asyncio.sleep(random.uniform(1, 3))
            
            # Simulate reading behavior with scrolling
            viewport = await page.viewport_size()
            if viewport:
                # Random scroll pattern
                scroll_count = random.randint(1, 3)
                for _ in range(scroll_count):
                    scroll_distance = random.randint(100, 400)
                    await page.mouse.wheel(0, scroll_distance)
                    await asyncio.sleep(random.uniform(0.5, 2.0))
                
                # Move mouse to simulate reading
                for _ in range(random.randint(2, 5)):
                    await page.mouse.move(
                        random.uniform(100, viewport['width'] - 100),
                        random.uniform(100, viewport['height'] - 100)
                    )
                    await asyncio.sleep(random.uniform(0.3, 1.5))
                    
        except Exception as e:
            console.print(f"[yellow]⚠️ Post-navigation behavior simulation failed: {e}[/yellow]")
    
    async def _check_for_detection(self, page) -> Dict[str, Any]:
        """Check if the page contains any anti-bot detection systems"""
        
        try:
            page_content = await page.content()
            page_title = await page.title()
            page_url = page.url
            
            # Check for various detection patterns
            for detection_type, patterns in self.detection_patterns.items():
                for pattern in patterns:
                    if (pattern.lower() in page_content.lower() or 
                        pattern.lower() in page_title.lower() or
                        pattern.lower() in page_url.lower()):
                        
                        return {
                            'detected': True,
                            'type': 'cloudflare' if 'cloudflare' in detection_type else 'captcha' if 'captcha' in pattern.lower() else detection_type,
                            'pattern': pattern,
                            'page_title': page_title,
                            'page_url': page_url
                        }
            
            # Check for CAPTCHA elements specifically
            captcha_challenges = await self.captcha_solver.detect_captcha_challenges(page)
            if captcha_challenges:
                return {
                    'detected': True,
                    'type': 'captcha',
                    'challenges': captcha_challenges,
                    'page_title': page_title,
                    'page_url': page_url
                }
            
            return {
                'detected': False,
                'type': None,
                'page_title': page_title,
                'page_url': page_url
            }
            
        except Exception as e:
            console.print(f"[yellow]⚠️ Detection check failed: {e}[/yellow]")
            return {'detected': False, 'type': None, 'error': str(e)}
    
    async def _handle_captcha_detection(self, page, page_id: str) -> bool:
        """Handle CAPTCHA detection with advanced solving"""
        
        try:
            page_data = self.active_pages[page_id]
            captcha_integration = page_data['captcha_solver']
            
            # Detect all CAPTCHAs on the page
            challenges = await self.captcha_solver.detect_captcha_challenges(page)
            
            if not challenges:
                console.print("[yellow]⚠️ No CAPTCHAs found during handling[/yellow]")
                return False
            
            # Solve each challenge
            all_solved = True
            for challenge in challenges:
                console.print(f"[cyan]🧩 Solving {challenge.challenge_type} challenge...[/cyan]")
                
                success = await self.captcha_solver.solve_captcha_challenge(page, challenge)
                if not success:
                    all_solved = False
                    console.print(f"[red]❌ Failed to solve {challenge.challenge_type}[/red]")
                else:
                    console.print(f"[green]✅ Solved {challenge.challenge_type}[/green]")
            
            # Wait for page to process the solutions
            if all_solved:
                await asyncio.sleep(random.uniform(2, 5))
                
                # Check if challenges are actually resolved
                remaining_challenges = await self.captcha_solver.detect_captcha_challenges(page)
                if not remaining_challenges:
                    console.print("[green]✅ All CAPTCHAs successfully resolved[/green]")
                    return True
                else:
                    console.print(f"[yellow]⚠️ {len(remaining_challenges)} CAPTCHAs still present[/yellow]")
                    return False
            
            return all_solved
            
        except Exception as e:
            console.print(f"[red]❌ CAPTCHA handling failed: {e}[/red]")
            return False
    
    async def _handle_cloudflare_detection(self, page, page_id: str) -> bool:
        """Handle Cloudflare detection specifically"""
        
        try:
            # Create a cloudflare-specific challenge
            from .captcha_solver import CaptchaChallenge
            
            cloudflare_challenge = CaptchaChallenge(
                challenge_type='cloudflare',
                challenge_url=page.url,
                difficulty_level='extreme'
            )
            
            # Use the advanced Cloudflare solver
            success = await self.captcha_solver._solve_cloudflare_challenge(page, cloudflare_challenge)
            
            if success:
                console.print("[green]✅ Cloudflare challenge resolved[/green]")
                await asyncio.sleep(random.uniform(2, 4))
                return True
            else:
                console.print("[red]❌ Cloudflare challenge failed[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]❌ Cloudflare handling failed: {e}[/red]")
            return False
    
    async def close_page(self, page_id: str):
        """Close a specific page"""
        
        if page_id in self.active_pages:
            try:
                page_data = self.active_pages[page_id]
                await page_data['page'].close()
                
                # Remove from tracking
                session_id = page_data['session_id']
                if session_id in self.active_browsers:
                    browser_data = self.active_browsers[session_id]
                    if page_id in browser_data['pages']:
                        del browser_data['pages'][page_id]
                
                del self.active_pages[page_id]
                console.print(f"[green]🗑️ Closed page: {page_id}[/green]")
                
            except Exception as e:
                console.print(f"[yellow]⚠️ Error closing page {page_id}: {e}[/yellow]")
    
    async def close_browser(self, session_id: str):
        """Close a browser session and all its pages"""
        
        if session_id in self.active_browsers:
            try:
                browser_data = self.active_browsers[session_id]
                browser = browser_data['browser']
                
                # Close all pages first
                page_ids = list(browser_data['pages'].keys())
                for page_id in page_ids:
                    await self.close_page(page_id)
                
                # Close browser
                await browser.close()
                
                # Cleanup session
                await self.camoufox_manager.cleanup_session(session_id)
                
                del self.active_browsers[session_id]
                console.print(f"[green]🗑️ Closed browser session: {session_id}[/green]")
                
            except Exception as e:
                console.print(f"[yellow]⚠️ Error closing browser {session_id}: {e}[/yellow]")
    
    async def cleanup_all(self):
        """Cleanup all active browsers and sessions"""
        
        console.print("[blue]🧹 Cleaning up all browser sessions...[/blue]")
        
        # Close all browsers
        session_ids = list(self.active_browsers.keys())
        for session_id in session_ids:
            await self.close_browser(session_id)
        
        # Cleanup expired sessions in Camoufox manager
        await self.camoufox_manager.cleanup_expired_sessions()
        
        console.print("[green]✅ Cleanup completed[/green]")
    
    async def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report of all active sessions"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'active_browsers': len(self.active_browsers),
            'active_pages': len(self.active_pages),
            'browser_details': {},
            'camoufox_stats': await self.camoufox_manager.get_session_statistics()
        }
        
        for session_id, browser_data in self.active_browsers.items():
            report['browser_details'][session_id] = {
                'created_at': browser_data['created_at'].isoformat(),
                'page_count': len(browser_data['pages']),
                'config': {
                    'headless': browser_data['config'].get('headless', False),
                    'proxy_enabled': browser_data['config'].get('proxy') is not None,
                    'anti_detection': True
                },
                'pages': {
                    page_id: {
                        'created_at': page_data['created_at'].isoformat(),
                        'navigation_count': page_data['navigation_count'],
                        'last_activity': page_data['last_activity'].isoformat()
                    }
                    for page_id, page_data in browser_data['pages'].items()
                }
            }
        
        return report

# Usage example function
async def demo_anti_detection_browser():
    """Demonstration of the anti-detection browser capabilities"""
    
    console.print("[blue]🚀 Starting Anti-Detection Browser Demo[/blue]")
    
    # Initialize the manager
    browser_manager = AntiDetectionBrowserManager()
    
    try:
        # Create a stealth browser
        session_id = await browser_manager.create_stealth_browser(
            headless=False,  # Set to True for headless operation
            proxy_config=None  # Add proxy config if needed
        )
        
        # Create a stealth page
        page_id = await browser_manager.create_stealth_page(session_id)
        
        # Navigate to a challenging site
        test_urls = [
            "https://bot.sannysoft.com/",  # Bot detection test
            "https://abrahamjuliot.github.io/creepjs/",  # Fingerprinting test
            "https://nopecha.com/demo/cloudflare",  # Cloudflare test
            "https://www.google.com"  # Basic functionality test
        ]
        
        for url in test_urls:
            console.print(f"\n[cyan]🌐 Testing: {url}[/cyan]")
            success = await browser_manager.navigate_with_protection(
                page_id=page_id,
                url=url,
                handle_captcha=True
            )
            
            if success:
                console.print(f"[green]✅ Successfully accessed {url}[/green]")
                # Wait a bit to observe results
                await asyncio.sleep(5)
            else:
                console.print(f"[red]❌ Failed to access {url}[/red]")
        
        # Get status report
        status = await browser_manager.get_status_report()
        console.print(f"\n[blue]📊 Status Report:[/blue]")
        console.print(f"   • Active browsers: {status['active_browsers']}")
        console.print(f"   • Active pages: {status['active_pages']}")
        
    finally:
        # Cleanup
        await browser_manager.cleanup_all()
        console.print("[green]🎯 Demo completed successfully[/green]")

if __name__ == "__main__":
    asyncio.run(demo_anti_detection_browser())
