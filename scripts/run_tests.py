#!/usr/bin/env python3
"""
Test runner script for crew-camufox project.
Provides easy execution of different test categories and comprehensive test reporting.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


class TestRunner:
    """Test runner for crew-camufox with multiple test categories."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_dir = project_root / "tests"
        
    def run_unit_tests(self, verbose: bool = False) -> int:
        """Run unit tests."""
        print("🧪 Running Unit Tests...")
        cmd = ["uv", "run", "pytest", "tests/unit/", "-m", "unit"]
        if verbose:
            cmd.extend(["-v", "-s"])
        cmd.extend(["--tb=short", "--color=yes"])
        return subprocess.run(cmd, cwd=self.project_root, check=False).returncode
    
    def run_integration_tests(self, verbose: bool = False) -> int:
        """Run integration tests."""
        print("🔗 Running Integration Tests...")
        cmd = ["uv", "run", "pytest", "tests/integration/", "-m", "integration"]
        if verbose:
            cmd.extend(["-v", "-s"])
        cmd.extend(["--tb=short", "--color=yes"])
        return subprocess.run(cmd, cwd=self.project_root, check=False).returncode
    
    def run_performance_tests(self, verbose: bool = False) -> int:
        """Run performance tests."""
        print("⚡ Running Performance Tests...")
        cmd = ["uv", "run", "pytest", "tests/performance/", "-m", "performance"]
        if verbose:
            cmd.extend(["-v", "-s"])
        cmd.extend(["--tb=short", "--color=yes", "--durations=10"])
        return subprocess.run(cmd, cwd=self.project_root, check=False).returncode
    
    def run_concurrency_tests(self, verbose: bool = False) -> int:
        """Run concurrency tests."""
        print("🔄 Running Concurrency Tests...")
        cmd = ["uv", "run", "pytest", "-m", "concurrency"]
        if verbose:
            cmd.extend(["-v", "-s"])
        cmd.extend(["--tb=short", "--color=yes"])
        return subprocess.run(cmd, cwd=self.project_root, check=False).returncode
    
    def run_scalability_tests(self, verbose: bool = False) -> int:
        """Run scalability tests."""
        print("📈 Running Scalability Tests...")
        cmd = ["uv", "run", "pytest", "-m", "scalability"]
        if verbose:
            cmd.extend(["-v", "-s"])
        cmd.extend(["--tb=short", "--color=yes", "--durations=10"])
        return subprocess.run(cmd, cwd=self.project_root, check=False).returncode
    
    def run_stress_tests(self, verbose: bool = False) -> int:
        """Run stress tests."""
        print("💪 Running Stress Tests...")
        cmd = ["uv", "run", "pytest", "-m", "stress"]
        if verbose:
            cmd.extend(["-v", "-s"])
        cmd.extend(["--tb=short", "--color=yes", "--durations=20"])
        return subprocess.run(cmd, cwd=self.project_root, check=False).returncode
    
    def run_reliability_tests(self, verbose: bool = False) -> int:
        """Run reliability tests."""
        print("🛡️ Running Reliability Tests...")
        cmd = ["uv", "run", "pytest", "-m", "reliability"]
        if verbose:
            cmd.extend(["-v", "-s"])
        cmd.extend(["--tb=short", "--color=yes"])
        return subprocess.run(cmd, cwd=self.project_root, check=False).returncode
    
    def run_quick_tests(self, verbose: bool = False) -> int:
        """Run quick test suite (unit tests only)."""
        print("🏃‍♂️ Running Quick Test Suite...")
        cmd = ["uv", "run", "pytest", "tests/unit/", "-x"]  # Stop on first failure
        if verbose:
            cmd.extend(["-v", "-s"])
        cmd.extend(["--tb=short", "--color=yes"])
        return subprocess.run(cmd, cwd=self.project_root, check=False).returncode
    
    def run_full_suite(self, verbose: bool = False) -> int:
        """Run complete test suite with coverage."""
        print("🎯 Running Full Test Suite...")
        
        cmd = [
            "uv", "run", "pytest",
            "tests/",
            "--cov=src",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-report=xml:coverage.xml"
        ]
        if verbose:
            cmd.extend(["-v", "-s"])
        cmd.extend(["--tb=short", "--color=yes", "--durations=20"])
        
        result = subprocess.run(cmd, cwd=self.project_root, check=False).returncode
        
        if result == 0:
            print("✅ Full test suite completed successfully!")
            print("📊 Coverage report available at: htmlcov/index.html")
        else:
            print("❌ Some tests failed. Check the output above.")
        
        return result
    
    def run_phase3_tests(self, verbose: bool = False) -> int:
        """Run Phase 3 Testing & Quality Assurance tests."""
        print("🎭 Running Phase 3 Testing & Quality Assurance...")
        
        test_categories = [
            ("Unit Tests", self.run_unit_tests),
            ("Integration Tests", self.run_integration_tests),
            ("Performance Tests", self.run_performance_tests),
            ("Concurrency Tests", self.run_concurrency_tests),
            ("Reliability Tests", self.run_reliability_tests)
        ]
        
        results = {}
        start_time = time.time()
        
        for category, test_func in test_categories:
            print(f"\n--- {category} ---")
            result = test_func(verbose)
            results[category] = result
            
            if result == 0:
                print(f"✅ {category} passed")
            else:
                print(f"❌ {category} failed (exit code: {result})")
        
        # Summary
        total_time = time.time() - start_time
        passed = sum(1 for r in results.values() if r == 0)
        total = len(results)
        
        print(f"\n{'='*60}")
        print(f"📋 PHASE 3 TEST SUITE SUMMARY")
        print(f"{'='*60}")
        print(f"⏱️  Total Time: {total_time:.2f} seconds")
        print(f"✅ Passed: {passed}/{total} test categories")
        print(f"❌ Failed: {total - passed}/{total} test categories")
        
        for category, result in results.items():
            status = "✅ PASS" if result == 0 else "❌ FAIL"
            print(f"   {status}: {category}")
        
        if all(r == 0 for r in results.values()):
            print("\n🎉 All Phase 3 tests passed! Quality assurance complete.")
            return 0
        else:
            print("\n⚠️  Some test categories failed. Review the output above.")
            return 1
    
    def lint_and_format(self) -> int:
        """Run linting and formatting checks."""
        print("🔍 Running Code Quality Checks...")
        
        # Install tools if not present
        tools = ["black", "isort", "flake8", "mypy"]
        for tool in tools:
            subprocess.run(["uv", "add", "--dev", tool], 
                          capture_output=True, check=False)
        
        results = []
        
        # Format with black
        print("🎨 Formatting code with black...")
        result = subprocess.run([
            "uv", "run", "black", "src/", "tests/", "--check", "--diff"
        ], cwd=self.project_root, check=False).returncode
        results.append(("Black formatting", result))
        
        # Sort imports with isort
        print("📦 Checking import sorting with isort...")
        result = subprocess.run([
            "uv", "run", "isort", "src/", "tests/", "--check-only", "--diff"
        ], cwd=self.project_root, check=False).returncode
        results.append(("Import sorting", result))
        
        # Lint with flake8
        print("🔍 Linting with flake8...")
        result = subprocess.run([
            "uv", "run", "flake8", "src/", "tests/", 
            "--max-line-length=88", "--extend-ignore=E203,W503"
        ], cwd=self.project_root, check=False).returncode
        results.append(("Flake8 linting", result))
        
        # Type checking with mypy (optional, may have issues)
        print("🎯 Type checking with mypy...")
        result = subprocess.run([
            "uv", "run", "mypy", "src/", "--ignore-missing-imports"
        ], cwd=self.project_root, check=False).returncode
        results.append(("Type checking", result))
        
        # Summary
        print("\n📋 CODE QUALITY SUMMARY:")
        for check, result in results:
            status = "✅ PASS" if result == 0 else "❌ FAIL"
            print(f"   {status}: {check}")
        
        if all(r == 0 for _, r in results):
            print("✅ All code quality checks passed!")
            return 0
        else:
            print("⚠️  Some code quality checks failed.")
            return 1


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Test runner for crew-camufox")
    
    parser.add_argument(
        "test_type",
        choices=[
            "unit", "integration", "performance", "concurrency", 
            "scalability", "stress", "reliability", "quick", 
            "full", "phase3", "lint"
        ],
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory"
    )
    
    args = parser.parse_args()
    
    runner = TestRunner(args.project_root)
    
    test_functions = {
        "unit": runner.run_unit_tests,
        "integration": runner.run_integration_tests,
        "performance": runner.run_performance_tests,
        "concurrency": runner.run_concurrency_tests,
        "scalability": runner.run_scalability_tests,
        "stress": runner.run_stress_tests,
        "reliability": runner.run_reliability_tests,
        "quick": runner.run_quick_tests,
        "full": runner.run_full_suite,
        "phase3": runner.run_phase3_tests,
        "lint": runner.lint_and_format
    }
    
    test_func = test_functions[args.test_type]
    
    if args.test_type == "lint":
        exit_code = test_func()
    else:
        exit_code = test_func(args.verbose)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
