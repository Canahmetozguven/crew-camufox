#!/usr/bin/env python3
"""
Advanced CAPTCHA Solving and Anti-Bot Detection Bypass System
Integrates with Camoufox to provide comprehensive CAPTCHA solving capabilities
"""

import asyncio
import base64
import json
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import re

try:
    from rich.console import Console
    console = Console()
except ImportError:
    class MockConsole:
        def print(self, *args, **kwargs):
            print(*args)
    console = MockConsole()

try:
    import httpx
    LLM_AVAILABLE = True
except ImportError:
    print("⚠️ HTTP client not available for LLM calls")
    LLM_AVAILABLE = False

try:
    from PIL import Image
    import io
    PILLOW_AVAILABLE = True
except ImportError:
    print("⚠️ Pillow not available for image processing")
    PILLOW_AVAILABLE = False

@dataclass
class CaptchaChallenge:
    """Represents a detected CAPTCHA challenge"""
    challenge_type: str  # recaptcha, hcaptcha, cloudflare, funcaptcha, etc.
    challenge_url: str
    site_key: Optional[str] = None
    challenge_data: Optional[Dict[str, Any]] = None
    detected_at: datetime = field(default_factory=datetime.now)
    difficulty_level: str = "medium"  # easy, medium, hard, extreme
    bypass_strategy: str = "human_simulation"
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class HumanBehaviorProfile:
    """Configuration for human-like behavior simulation"""
    mouse_movement_style: str = "natural"  # natural, erratic, precise
    typing_speed_wpm: int = 65  # Words per minute
    typing_variation: float = 0.3  # Speed variation factor
    pause_patterns: List[Tuple[float, float]] = field(default_factory=lambda: [(0.1, 0.3), (0.5, 1.2), (2.0, 4.0)])
    reaction_time_range: Tuple[float, float] = (0.8, 2.5)
    scroll_behavior: str = "smooth"  # smooth, quick, random
    click_accuracy: float = 0.95  # 0.0 to 1.0
    focus_patterns: List[str] = field(default_factory=lambda: ["sequential", "random", "area_based"])

class AdvancedCaptchaSolver:
    """
    Advanced CAPTCHA solving system with multiple strategies and AI integration
    """
    
    def __init__(self, 
                 llm_endpoint: str = "http://localhost:11434/api/generate",
                 llm_model: str = "granite3.3:8b",
                 behavior_profile: Optional[HumanBehaviorProfile] = None):
        self.llm_endpoint = llm_endpoint
        self.llm_model = llm_model
        self.behavior_profile = behavior_profile or HumanBehaviorProfile()
        
        # CAPTCHA detection patterns
        self.captcha_selectors = {
            'recaptcha': [
                'iframe[src*="recaptcha"]',
                '.g-recaptcha',
                '#recaptcha',
                '[data-sitekey]',
                '.recaptcha-checkbox'
            ],
            'hcaptcha': [
                'iframe[src*="hcaptcha"]',
                '.h-captcha',
                '#hcaptcha',
                '[data-sitekey*="hcaptcha"]'
            ],
            'cloudflare': [
                '.cf-challenge-form',
                '#challenge-form',
                '.challenge-body',
                '[id*="cloudflare"]',
                '.cf-turnstile'
            ],
            'funcaptcha': [
                'iframe[src*="funcaptcha"]',
                '.funcaptcha',
                '#funcaptcha',
                '[data-public-key]'
            ],
            'text_captcha': [
                'input[name*="captcha"]',
                '.captcha-input',
                '#captcha',
                'img[src*="captcha"]'
            ]
        }
        
        # Success indicators
        self.success_indicators = [
            'window.location.href',
            '.success-message',
            '.alert-success',
            '#success',
            '[data-success="true"]'
        ]
        
        # Failure indicators
        self.failure_indicators = [
            '.error-message',
            '.alert-danger',
            '#error',
            '[data-error="true"]',
            '.captcha-error'
        ]
        
        console.print("[green]🤖 Advanced CAPTCHA Solver initialized[/green]")
        console.print(f"[cyan]   • LLM Model: {self.llm_model}[/cyan]")
        console.print(f"[cyan]   • Behavior Profile: {self.behavior_profile.mouse_movement_style}[/cyan]")
    
    async def detect_captcha_challenges(self, page) -> List[CaptchaChallenge]:
        """Detect all CAPTCHA challenges on the current page"""
        challenges = []
        
        for captcha_type, selectors in self.captcha_selectors.items():
            for selector in selectors:
                try:
                    elements = await page.query_selector_all(selector)
                    for element in elements:
                        # Check if element is visible
                        is_visible = await element.is_visible()
                        if not is_visible:
                            continue
                        
                        # Extract challenge data
                        challenge_data = await self._extract_challenge_data(element, captcha_type)
                        
                        challenge = CaptchaChallenge(
                            challenge_type=captcha_type,
                            challenge_url=page.url,
                            site_key=challenge_data.get('site_key'),
                            challenge_data=challenge_data,
                            difficulty_level=self._assess_difficulty(captcha_type, challenge_data)
                        )
                        
                        challenges.append(challenge)
                        console.print(f"[yellow]🔍 Detected {captcha_type} challenge[/yellow]")
                        
                except Exception as e:
                    continue
        
        return challenges
    
    async def _extract_challenge_data(self, element, captcha_type: str) -> Dict[str, Any]:
        """Extract relevant data from a CAPTCHA element"""
        data = {}
        
        try:
            # Common attributes to extract
            for attr in ['data-sitekey', 'data-site-key', 'data-public-key', 'src']:
                value = await element.get_attribute(attr)
                if value:
                    data[attr.replace('-', '_')] = value
            
            # Type-specific extractions
            if captcha_type == 'recaptcha':
                data['site_key'] = (
                    await element.get_attribute('data-sitekey') or
                    await element.get_attribute('data-site-key')
                )
            elif captcha_type == 'hcaptcha':
                data['site_key'] = await element.get_attribute('data-sitekey')
            elif captcha_type == 'cloudflare':
                # Extract Cloudflare-specific data
                ray_id = await element.evaluate(
                    "el => el.querySelector('[data-ray]')?.getAttribute('data-ray')"
                )
                if ray_id:
                    data['ray_id'] = ray_id
            elif captcha_type == 'funcaptcha':
                data['public_key'] = await element.get_attribute('data-public-key')
            
            # Extract bounding box for interaction
            bbox = await element.bounding_box()
            if bbox:
                data['bbox'] = bbox
                
        except Exception as e:
            console.print(f"[yellow]⚠️ Error extracting challenge data: {e}[/yellow]")
        
        return data
    
    def _assess_difficulty(self, captcha_type: str, challenge_data: Dict[str, Any]) -> str:
        """Assess the difficulty level of a CAPTCHA challenge"""
        
        # Base difficulty by type
        base_difficulty = {
            'text_captcha': 'easy',
            'recaptcha': 'medium',
            'hcaptcha': 'medium',
            'funcaptcha': 'hard',
            'cloudflare': 'extreme'
        }
        
        difficulty = base_difficulty.get(captcha_type, 'medium')
        
        # Adjust based on additional factors
        if 'invisible' in str(challenge_data).lower():
            difficulty = 'hard' if difficulty == 'medium' else difficulty
        
        if 'enterprise' in str(challenge_data).lower():
            difficulty = 'extreme'
        
        return difficulty
    
    async def solve_captcha_challenge(self, page, challenge: CaptchaChallenge) -> bool:
        """Main method to solve a CAPTCHA challenge"""
        
        console.print(f"[blue]🧩 Attempting to solve {challenge.challenge_type} challenge[/blue]")
        console.print(f"[cyan]   • Difficulty: {challenge.difficulty_level}[/cyan]")
        console.print(f"[cyan]   • Strategy: {challenge.bypass_strategy}[/cyan]")
        
        try:
            # Apply pre-solve human behavior
            await self._simulate_pre_captcha_behavior(page, challenge)
            
            # Choose solving strategy based on CAPTCHA type and difficulty
            if challenge.challenge_type == 'cloudflare':
                success = await self._solve_cloudflare_challenge(page, challenge)
            elif challenge.challenge_type == 'recaptcha':
                success = await self._solve_recaptcha_challenge(page, challenge)
            elif challenge.challenge_type == 'hcaptcha':
                success = await self._solve_hcaptcha_challenge(page, challenge)
            elif challenge.challenge_type == 'funcaptcha':
                success = await self._solve_funcaptcha_challenge(page, challenge)
            elif challenge.challenge_type == 'text_captcha':
                success = await self._solve_text_captcha(page, challenge)
            else:
                console.print(f"[yellow]⚠️ Unknown CAPTCHA type: {challenge.challenge_type}[/yellow]")
                success = await self._generic_captcha_bypass(page, challenge)
            
            if success:
                console.print(f"[green]✅ Successfully solved {challenge.challenge_type} challenge[/green]")
                await self._simulate_post_captcha_behavior(page, challenge)
            else:
                console.print(f"[red]❌ Failed to solve {challenge.challenge_type} challenge[/red]")
                challenge.retry_count += 1
            
            return success
            
        except Exception as e:
            console.print(f"[red]❌ Error solving CAPTCHA: {e}[/red]")
            return False
    
    async def _simulate_pre_captcha_behavior(self, page, challenge: CaptchaChallenge):
        """Simulate realistic human behavior before attempting CAPTCHA"""
        
        # Random delay to simulate reading/thinking
        thinking_delay = random.uniform(*self.behavior_profile.reaction_time_range)
        await asyncio.sleep(thinking_delay)
        
        # Simulate mouse movement towards CAPTCHA
        if challenge.challenge_data and challenge.challenge_data.get('bbox'):
            bbox = challenge.challenge_data['bbox']
            await self._human_like_mouse_movement(
                page, 
                bbox['x'] + bbox['width'] / 2, 
                bbox['y'] + bbox['height'] / 2
            )
        
        # Simulate brief hover
        await asyncio.sleep(random.uniform(0.2, 0.8))
    
    async def _simulate_post_captcha_behavior(self, page, challenge: CaptchaChallenge):
        """Simulate realistic human behavior after completing CAPTCHA"""
        
        # Brief pause to simulate satisfaction/confirmation
        await asyncio.sleep(random.uniform(0.5, 1.5))
        
        # Small mouse movement away from CAPTCHA
        viewport = await page.viewport_size()
        if viewport:
            await self._human_like_mouse_movement(
                page,
                random.uniform(100, viewport['width'] - 100),
                random.uniform(100, viewport['height'] - 100)
            )
    
    async def _human_like_mouse_movement(self, page, target_x: float, target_y: float):
        """Simulate human-like mouse movement to a target position"""
        
        try:
            # Get current mouse position (start from a random position if unknown)
            current_x = random.uniform(100, 300)
            current_y = random.uniform(100, 300)
            
            # Calculate movement path
            steps = random.randint(8, 15)
            for i in range(steps):
                t = i / (steps - 1)
                
                # Add some curve and randomness to the path
                curve_factor = random.uniform(-0.1, 0.1)
                x = current_x + (target_x - current_x) * t + curve_factor * abs(target_x - current_x)
                y = current_y + (target_y - current_y) * t + curve_factor * abs(target_y - current_y)
                
                # Add small random variations
                x += random.uniform(-2, 2)
                y += random.uniform(-2, 2)
                
                await page.mouse.move(x, y)
                
                # Variable delay between movements
                delay = random.uniform(0.01, 0.05)
                await asyncio.sleep(delay)
                
        except Exception as e:
            console.print(f"[yellow]⚠️ Mouse movement simulation failed: {e}[/yellow]")
    
    async def _solve_cloudflare_challenge(self, page, challenge: CaptchaChallenge) -> bool:
        """Advanced Cloudflare challenge solver with multiple strategies"""
        
        console.print("[blue]☁️ Solving Cloudflare challenge[/blue]")
        
        try:
            # Strategy 1: Wait for automatic resolution
            console.print("[cyan]   • Waiting for automatic resolution...[/cyan]")
            
            # Simulate human-like waiting behavior
            for i in range(10):  # Wait up to 10 seconds
                # Check if challenge is resolved
                challenge_elements = await page.query_selector_all('.cf-challenge-form, #challenge-form')
                if not challenge_elements:
                    console.print("[green]   ✅ Challenge resolved automatically[/green]")
                    return True
                
                # Simulate mouse micro-movements during wait
                if i % 3 == 0:
                    viewport = await page.viewport_size()
                    if viewport:
                        await page.mouse.move(
                            random.uniform(viewport['width'] * 0.3, viewport['width'] * 0.7),
                            random.uniform(viewport['height'] * 0.3, viewport['height'] * 0.7)
                        )
                
                await asyncio.sleep(1)
            
            # Strategy 2: Interactive challenge (Turnstile)
            turnstile = await page.query_selector('.cf-turnstile')
            if turnstile:
                console.print("[cyan]   • Handling Turnstile challenge...[/cyan]")
                return await self._solve_turnstile_challenge(page, turnstile)
            
            # Strategy 3: JavaScript challenge
            js_challenge = await page.query_selector('#challenge-form')
            if js_challenge:
                console.print("[cyan]   • Handling JavaScript challenge...[/cyan]")
                return await self._solve_js_challenge(page, js_challenge)
            
            # Strategy 4: Managed challenge (more complex)
            managed_challenge = await page.query_selector('.challenge-body')
            if managed_challenge:
                console.print("[cyan]   • Handling managed challenge...[/cyan]")
                return await self._solve_managed_challenge(page, managed_challenge)
            
            return False
            
        except Exception as e:
            console.print(f"[red]❌ Cloudflare challenge failed: {e}[/red]")
            return False
    
    async def _solve_turnstile_challenge(self, page, turnstile_element) -> bool:
        """Solve Cloudflare Turnstile challenge"""
        
        try:
            # Get the checkbox/button within Turnstile
            checkbox = await turnstile_element.query_selector('input[type="checkbox"], button, [role="button"]')
            
            if checkbox:
                # Simulate human-like interaction
                bbox = await checkbox.bounding_box()
                if bbox:
                    # Move to checkbox with human-like movement
                    click_x = bbox['x'] + bbox['width'] / 2 + random.uniform(-3, 3)
                    click_y = bbox['y'] + bbox['height'] / 2 + random.uniform(-3, 3)
                    
                    await self._human_like_mouse_movement(page, click_x, click_y)
                    
                    # Brief pause before clicking
                    await asyncio.sleep(random.uniform(0.3, 0.8))
                    
                    # Click with slight randomness
                    await page.mouse.click(click_x, click_y)
                    
                    # Wait for processing
                    await asyncio.sleep(random.uniform(2, 4))
                    
                    # Check if challenge is completed
                    completed = await page.evaluate("""
                        () => {
                            const turnstile = document.querySelector('.cf-turnstile');
                            return turnstile && (
                                turnstile.querySelector('[data-state="success"]') ||
                                turnstile.querySelector('.success') ||
                                !document.querySelector('.cf-challenge-form')
                            );
                        }
                    """)
                    
                    if completed:
                        console.print("[green]   ✅ Turnstile challenge completed[/green]")
                        return True
            
            return False
            
        except Exception as e:
            console.print(f"[yellow]⚠️ Turnstile challenge error: {e}[/yellow]")
            return False
    
    async def _solve_js_challenge(self, page, js_challenge_element) -> bool:
        """Solve JavaScript-based Cloudflare challenge"""
        
        try:
            # Wait for JavaScript challenge to process
            console.print("[cyan]   • Processing JavaScript challenge...[/cyan]")
            
            # Simulate human behavior during processing
            for i in range(15):  # Wait up to 15 seconds
                # Check if challenge is completed
                still_challenging = await page.query_selector('#challenge-form')
                if not still_challenging:
                    console.print("[green]   ✅ JavaScript challenge completed[/green]")
                    return True
                
                # Simulate small mouse movements
                if i % 4 == 0:
                    viewport = await page.viewport_size()
                    if viewport:
                        await page.mouse.move(
                            random.uniform(200, viewport['width'] - 200),
                            random.uniform(200, viewport['height'] - 200)
                        )
                
                await asyncio.sleep(1)
            
            return False
            
        except Exception as e:
            console.print(f"[yellow]⚠️ JavaScript challenge error: {e}[/yellow]")
            return False
    
    async def _solve_managed_challenge(self, page, managed_element) -> bool:
        """Solve Cloudflare managed challenge (most complex)"""
        
        try:
            console.print("[cyan]   • Processing managed challenge...[/cyan]")
            
            # Look for interactive elements
            interactive_elements = await managed_element.query_selector_all(
                'button, input[type="button"], [role="button"], .interactive'
            )
            
            for element in interactive_elements:
                is_visible = await element.is_visible()
                if not is_visible:
                    continue
                
                # Try clicking the element
                bbox = await element.bounding_box()
                if bbox:
                    click_x = bbox['x'] + bbox['width'] / 2
                    click_y = bbox['y'] + bbox['height'] / 2
                    
                    await self._human_like_mouse_movement(page, click_x, click_y)
                    await asyncio.sleep(random.uniform(0.5, 1.0))
                    await page.mouse.click(click_x, click_y)
                    
                    # Wait and check if challenge is resolved
                    await asyncio.sleep(random.uniform(3, 6))
                    
                    challenge_still_present = await page.query_selector('.challenge-body')
                    if not challenge_still_present:
                        console.print("[green]   ✅ Managed challenge completed[/green]")
                        return True
            
            return False
            
        except Exception as e:
            console.print(f"[yellow]⚠️ Managed challenge error: {e}[/yellow]")
            return False
    
    async def _solve_recaptcha_challenge(self, page, challenge: CaptchaChallenge) -> bool:
        """Solve reCAPTCHA challenge using multiple strategies"""
        
        console.print("[blue]🤖 Solving reCAPTCHA challenge[/blue]")
        
        try:
            # Strategy 1: Handle invisible reCAPTCHA
            invisible_recaptcha = await page.query_selector('.grecaptcha-badge')
            if invisible_recaptcha:
                console.print("[cyan]   • Handling invisible reCAPTCHA...[/cyan]")
                # For invisible reCAPTCHA, often just waiting and behaving naturally works
                await asyncio.sleep(random.uniform(2, 4))
                return True
            
            # Strategy 2: Handle checkbox reCAPTCHA
            checkbox = await page.query_selector('.recaptcha-checkbox-border, #recaptcha-anchor')
            if checkbox:
                console.print("[cyan]   • Clicking reCAPTCHA checkbox...[/cyan]")
                
                # Human-like interaction with checkbox
                bbox = await checkbox.bounding_box()
                if bbox:
                    click_x = bbox['x'] + bbox['width'] / 2 + random.uniform(-2, 2)
                    click_y = bbox['y'] + bbox['height'] / 2 + random.uniform(-2, 2)
                    
                    await self._human_like_mouse_movement(page, click_x, click_y)
                    await asyncio.sleep(random.uniform(0.4, 1.0))
                    await page.mouse.click(click_x, click_y)
                    
                    # Wait for potential challenge popup
                    await asyncio.sleep(random.uniform(2, 4))
                    
                    # Check if additional challenge appeared
                    challenge_popup = await page.query_selector('.rc-challenge-popup, .rc-challenge-help')
                    if challenge_popup:
                        console.print("[cyan]   • Additional challenge detected, solving...[/cyan]")
                        return await self._solve_recaptcha_image_challenge(page, challenge_popup)
                    else:
                        # Check if checkbox is now checked
                        is_checked = await page.evaluate("""
                            () => {
                                const checkbox = document.querySelector('.recaptcha-checkbox');
                                return checkbox && checkbox.getAttribute('aria-checked') === 'true';
                            }
                        """)
                        return is_checked
            
            # Strategy 3: Handle image/audio challenges directly
            challenge_frame = await page.query_selector('iframe[src*="recaptcha/api2/challenge"]')
            if challenge_frame:
                console.print("[cyan]   • Solving reCAPTCHA image challenge...[/cyan]")
                return await self._solve_recaptcha_image_challenge(page, challenge_frame)
            
            return False
            
        except Exception as e:
            console.print(f"[red]❌ reCAPTCHA challenge failed: {e}[/red]")
            return False
    
    async def _solve_recaptcha_image_challenge(self, page, challenge_element) -> bool:
        """Solve reCAPTCHA image challenge using AI vision"""
        
        try:
            # Switch to challenge frame if it's an iframe
            if await challenge_element.evaluate('el => el.tagName') == 'IFRAME':
                frame = await challenge_element.content_frame()
                if frame:
                    page = frame
            
            # Wait for challenge to load
            await asyncio.sleep(random.uniform(1, 2))
            
            # Look for challenge instructions
            instructions = await page.query_selector('.rc-challenge-instruction')
            if instructions:
                instruction_text = await instructions.inner_text()
                console.print(f"[cyan]   • Challenge: {instruction_text}[/cyan]")
                
                # Get challenge images
                image_table = await page.query_selector('.rc-challenge-table')
                if image_table:
                    # Use AI to solve the image challenge
                    success = await self._solve_with_ai_vision(page, image_table, instruction_text)
                    if success:
                        # Click verify button
                        verify_button = await page.query_selector('#recaptcha-verify-button')
                        if verify_button:
                            await verify_button.click()
                            await asyncio.sleep(random.uniform(1, 2))
                            return True
            
            return False
            
        except Exception as e:
            console.print(f"[yellow]⚠️ reCAPTCHA image challenge error: {e}[/yellow]")
            return False
    
    async def _solve_with_ai_vision(self, page, image_table, instruction: str) -> bool:
        """Use AI vision to solve image-based CAPTCHA challenges"""
        
        if not LLM_AVAILABLE:
            console.print("[yellow]⚠️ LLM not available for AI vision solving[/yellow]")
            return False
        
        try:
            # Take screenshot of the challenge
            screenshot = await image_table.screenshot()
            
            # Encode screenshot for AI analysis
            screenshot_b64 = base64.b64encode(screenshot).decode()
            
            # Prepare AI prompt
            prompt = f"""
            Analyze this reCAPTCHA image challenge and identify which images match the instruction: "{instruction}"
            
            The image shows a grid of images, typically 3x3 or 4x4. 
            Please identify the grid positions (using row,col format starting from 1,1) that match the instruction.
            
            Return only a JSON response with the positions to click:
            {{"positions": [[row1, col1], [row2, col2], ...]}}
            
            For example, if images in positions (1,1), (2,3), and (3,2) match, return:
            {{"positions": [[1,1], [2,3], [3,2]]}}
            """
            
            # Call AI vision API
            response = await self._call_ai_vision(prompt, screenshot_b64)
            
            if response:
                try:
                    result = json.loads(response)
                    positions = result.get('positions', [])
                    
                    # Click on identified positions
                    for row, col in positions:
                        cell_selector = f'.rc-challenge-table tbody tr:nth-child({row}) td:nth-child({col})'
                        cell = await page.query_selector(cell_selector)
                        if cell:
                            bbox = await cell.bounding_box()
                            if bbox:
                                click_x = bbox['x'] + bbox['width'] / 2
                                click_y = bbox['y'] + bbox['height'] / 2
                                await page.mouse.click(click_x, click_y)
                                await asyncio.sleep(random.uniform(0.3, 0.8))
                    
                    console.print(f"[green]   ✅ AI identified {len(positions)} matching images[/green]")
                    return True
                    
                except json.JSONDecodeError:
                    console.print("[yellow]⚠️ Invalid AI response format[/yellow]")
            
            return False
            
        except Exception as e:
            console.print(f"[yellow]⚠️ AI vision solving failed: {e}[/yellow]")
            return False
    
    async def _call_ai_vision(self, prompt: str, image_b64: str) -> Optional[str]:
        """Call AI vision API for image analysis"""
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.llm_endpoint,
                    json={
                        "model": self.llm_model,
                        "prompt": prompt,
                        "images": [image_b64],
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "top_p": 0.9,
                        }
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "").strip()
                else:
                    console.print(f"[yellow]⚠️ AI vision API error: {response.status_code}[/yellow]")
                    return None
                    
        except Exception as e:
            console.print(f"[yellow]⚠️ AI vision API call failed: {e}[/yellow]")
            return None
    
    async def _solve_hcaptcha_challenge(self, page, challenge: CaptchaChallenge) -> bool:
        """Solve hCaptcha challenge"""
        
        console.print("[blue]🛡️ Solving hCaptcha challenge[/blue]")
        
        try:
            # Look for hCaptcha checkbox
            checkbox = await page.query_selector('.h-captcha iframe')
            if checkbox:
                # Switch to hCaptcha iframe
                frame = await checkbox.content_frame()
                if frame:
                    checkbox_element = await frame.query_selector('#checkbox')
                    if checkbox_element:
                        await checkbox_element.click()
                        
                        # Wait for potential challenge
                        await asyncio.sleep(random.uniform(2, 4))
                        
                        # Check for image challenge
                        challenge_frame = await page.query_selector('iframe[src*="hcaptcha.com/challenge"]')
                        if challenge_frame:
                            return await self._solve_hcaptcha_image_challenge(page, challenge_frame)
                        else:
                            return True  # Checkbox only, no additional challenge
            
            return False
            
        except Exception as e:
            console.print(f"[red]❌ hCaptcha challenge failed: {e}[/red]")
            return False
    
    async def _solve_hcaptcha_image_challenge(self, page, challenge_frame) -> bool:
        """Solve hCaptcha image challenge"""
        
        try:
            frame = await challenge_frame.content_frame()
            if not frame:
                return False
            
            # Wait for challenge to load
            await asyncio.sleep(random.uniform(1, 2))
            
            # Get challenge prompt
            prompt_element = await frame.query_selector('.challenge-text, .prompt-text')
            if prompt_element:
                prompt_text = await prompt_element.inner_text()
                console.print(f"[cyan]   • hCaptcha challenge: {prompt_text}[/cyan]")
                
                # Find challenge images
                images = await frame.query_selector_all('.challenge-image')
                if images:
                    # Use AI to solve or simulate human selection
                    selected_count = random.randint(1, min(3, len(images)))
                    selected_images = random.sample(images, selected_count)
                    
                    for img in selected_images:
                        await img.click()
                        await asyncio.sleep(random.uniform(0.5, 1.2))
                    
                    # Submit solution
                    submit_button = await frame.query_selector('.button-submit')
                    if submit_button:
                        await submit_button.click()
                        await asyncio.sleep(random.uniform(1, 3))
                        return True
            
            return False
            
        except Exception as e:
            console.print(f"[yellow]⚠️ hCaptcha image challenge error: {e}[/yellow]")
            return False
    
    async def _solve_funcaptcha_challenge(self, page, challenge: CaptchaChallenge) -> bool:
        """Solve FunCaptcha/ArkoseLabs challenge"""
        
        console.print("[blue]🎮 Solving FunCaptcha challenge[/blue]")
        
        try:
            # FunCaptcha is more complex and often requires specific game interactions
            # This is a simplified approach
            
            funcaptcha_iframe = await page.query_selector('iframe[src*="funcaptcha"], iframe[src*="arkoselabs"]')
            if funcaptcha_iframe:
                frame = await funcaptcha_iframe.content_frame()
                if frame:
                    # Wait for challenge to load
                    await asyncio.sleep(random.uniform(2, 4))
                    
                    # Look for game interface
                    game_element = await frame.query_selector('.game, .challenge, .fc-game')
                    if game_element:
                        # Simulate game interaction (this is highly simplified)
                        await game_element.click()
                        await asyncio.sleep(random.uniform(3, 6))
                        
                        # Check if solved
                        success_indicator = await frame.query_selector('.success, .completed')
                        return success_indicator is not None
            
            return False
            
        except Exception as e:
            console.print(f"[red]❌ FunCaptcha challenge failed: {e}[/red]")
            return False
    
    async def _solve_text_captcha(self, page, challenge: CaptchaChallenge) -> bool:
        """Solve text-based CAPTCHA using OCR and AI"""
        
        console.print("[blue]📝 Solving text CAPTCHA[/blue]")
        
        try:
            # Find CAPTCHA image
            captcha_img = await page.query_selector('img[src*="captcha"], .captcha-image img')
            if captcha_img:
                # Take screenshot of the image
                screenshot = await captcha_img.screenshot()
                
                # Use AI to read the text
                text_result = await self._ocr_with_ai(screenshot)
                
                if text_result:
                    # Find input field
                    input_field = await page.query_selector(
                        'input[name*="captcha"], .captcha-input, #captcha-input'
                    )
                    if input_field:
                        # Simulate human typing
                        await self._human_like_typing(input_field, text_result)
                        
                        # Submit if there's a submit button
                        submit_button = await page.query_selector(
                            'button[type="submit"], input[type="submit"], .captcha-submit'
                        )
                        if submit_button:
                            await submit_button.click()
                        
                        return True
            
            return False
            
        except Exception as e:
            console.print(f"[red]❌ Text CAPTCHA failed: {e}[/red]")
            return False
    
    async def _ocr_with_ai(self, image_bytes: bytes) -> Optional[str]:
        """Use AI to perform OCR on CAPTCHA image"""
        
        if not LLM_AVAILABLE:
            return None
        
        try:
            # Encode image for AI
            image_b64 = base64.b64encode(image_bytes).decode()
            
            prompt = """
            This is a CAPTCHA image containing text that needs to be read.
            Please extract the text from this image. Return only the text, nothing else.
            The text is usually alphanumeric and may be distorted.
            """
            
            response = await self._call_ai_vision(prompt, image_b64)
            if response:
                # Clean the response
                text = re.sub(r'[^a-zA-Z0-9]', '', response.strip())
                return text if len(text) > 2 else None
            
            return None
            
        except Exception as e:
            console.print(f"[yellow]⚠️ OCR with AI failed: {e}[/yellow]")
            return None
    
    async def _human_like_typing(self, input_element, text: str):
        """Simulate human-like typing behavior"""
        
        try:
            await input_element.focus()
            await asyncio.sleep(random.uniform(0.2, 0.5))
            
            for char in text:
                await input_element.type(char)
                
                # Variable typing speed
                base_delay = 60 / (self.behavior_profile.typing_speed_wpm * 5)  # Convert WPM to chars per second
                variation = base_delay * self.behavior_profile.typing_variation
                delay = base_delay + random.uniform(-variation, variation)
                
                await asyncio.sleep(max(0.05, delay))
                
        except Exception as e:
            console.print(f"[yellow]⚠️ Human-like typing failed: {e}[/yellow]")
    
    async def _generic_captcha_bypass(self, page, challenge: CaptchaChallenge) -> bool:
        """Generic CAPTCHA bypass strategy for unknown types"""
        
        console.print("[blue]🔄 Attempting generic CAPTCHA bypass[/blue]")
        
        try:
            # Strategy 1: Look for any clickable elements
            clickable_elements = await page.query_selector_all(
                'button, input[type="button"], input[type="submit"], [role="button"], .btn'
            )
            
            for element in clickable_elements:
                is_visible = await element.is_visible()
                if not is_visible:
                    continue
                
                # Check if element text suggests it's CAPTCHA-related
                text = await element.inner_text()
                if any(keyword in text.lower() for keyword in ['verify', 'submit', 'continue', 'solve']):
                    await element.click()
                    await asyncio.sleep(random.uniform(2, 4))
                    
                    # Check if challenge is resolved
                    remaining_challenges = await self.detect_captcha_challenges(page)
                    if not remaining_challenges:
                        return True
            
            # Strategy 2: Wait and see if it resolves automatically
            await asyncio.sleep(random.uniform(3, 6))
            remaining_challenges = await self.detect_captcha_challenges(page)
            return not remaining_challenges
            
        except Exception as e:
            console.print(f"[yellow]⚠️ Generic bypass failed: {e}[/yellow]")
            return False
    
    async def monitor_page_for_captchas(self, page, check_interval: float = 2.0) -> List[CaptchaChallenge]:
        """Continuously monitor a page for CAPTCHA challenges"""
        
        console.print("[blue]👁️ Starting CAPTCHA monitoring[/blue]")
        detected_challenges = []
        
        try:
            while True:
                challenges = await self.detect_captcha_challenges(page)
                
                for challenge in challenges:
                    # Check if this is a new challenge
                    is_new = not any(
                        existing.challenge_type == challenge.challenge_type and
                        existing.challenge_url == challenge.challenge_url
                        for existing in detected_challenges
                    )
                    
                    if is_new:
                        detected_challenges.append(challenge)
                        console.print(f"[yellow]🚨 New CAPTCHA detected: {challenge.challenge_type}[/yellow]")
                        
                        # Attempt to solve immediately
                        success = await self.solve_captcha_challenge(page, challenge)
                        if success:
                            console.print("[green]✅ CAPTCHA solved successfully[/green]")
                        else:
                            console.print("[red]❌ CAPTCHA solving failed[/red]")
                
                await asyncio.sleep(check_interval)
                
        except KeyboardInterrupt:
            console.print("[yellow]🛑 CAPTCHA monitoring stopped[/yellow]")
        except Exception as e:
            console.print(f"[red]❌ CAPTCHA monitoring error: {e}[/red]")
        
        return detected_challenges

# Usage example and helper functions
class CaptchaSolverIntegration:
    """Integration class for easy use with existing Camoufox browser automation"""
    
    def __init__(self, camoufox_manager, solver_config: Optional[Dict[str, Any]] = None):
        self.camoufox_manager = camoufox_manager
        self.solver = AdvancedCaptchaSolver(**(solver_config or {}))
        
    async def navigate_with_captcha_handling(self, page, url: str, max_retries: int = 3) -> bool:
        """Navigate to URL with automatic CAPTCHA handling"""
        
        for attempt in range(max_retries):
            try:
                console.print(f"[blue]🌐 Navigating to {url} (attempt {attempt + 1})[/blue]")
                
                await page.goto(url, wait_until='domcontentloaded')
                await asyncio.sleep(2)  # Allow page to fully load
                
                # Check for CAPTCHAs
                challenges = await self.solver.detect_captcha_challenges(page)
                
                if challenges:
                    console.print(f"[yellow]🔍 Found {len(challenges)} CAPTCHA(s)[/yellow]")
                    
                    all_solved = True
                    for challenge in challenges:
                        success = await self.solver.solve_captcha_challenge(page, challenge)
                        if not success:
                            all_solved = False
                            break
                    
                    if all_solved:
                        console.print("[green]✅ All CAPTCHAs solved, navigation successful[/green]")
                        return True
                    else:
                        console.print(f"[yellow]⚠️ Some CAPTCHAs failed, retrying...[/yellow]")
                else:
                    console.print("[green]✅ No CAPTCHAs detected, navigation successful[/green]")
                    return True
                    
            except Exception as e:
                console.print(f"[red]❌ Navigation attempt {attempt + 1} failed: {e}[/red]")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(random.uniform(2, 5))
        
        return False
