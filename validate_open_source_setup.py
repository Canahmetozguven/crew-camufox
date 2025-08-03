#!/usr/bin/env python3
"""
Open Source Release Validation Script for Crew-Camufox
Validates that all essential files and configurations are properly set up.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

class ReleaseValidator:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.errors = []
        self.warnings = []
        self.success_count = 0
        self.total_checks = 0
    
    def check_file_exists(self, file_path: str, required: bool = True) -> bool:
        """Check if a file exists"""
        self.total_checks += 1
        full_path = self.project_root / file_path
        
        if full_path.exists():
            print(f"✅ {file_path} exists")
            self.success_count += 1
            return True
        else:
            message = f"❌ {file_path} missing"
            if required:
                self.errors.append(message)
            else:
                self.warnings.append(message)
            print(message)
            return False
    
    def check_directory_exists(self, dir_path: str, required: bool = True) -> bool:
        """Check if a directory exists"""
        self.total_checks += 1
        full_path = self.project_root / dir_path
        
        if full_path.exists() and full_path.is_dir():
            print(f"✅ {dir_path}/ directory exists")
            self.success_count += 1
            return True
        else:
            message = f"❌ {dir_path}/ directory missing"
            if required:
                self.errors.append(message)
            else:
                self.warnings.append(message)
            print(message)
            return False
    
    def check_file_content(self, file_path: str, required_content: str, description: str) -> bool:
        """Check if file contains required content"""
        self.total_checks += 1
        full_path = self.project_root / file_path
        
        if not full_path.exists():
            message = f"❌ {file_path} missing for {description} check"
            self.errors.append(message)
            print(message)
            return False
        
        try:
            content = full_path.read_text()
            if required_content in content:
                print(f"✅ {file_path} contains {description}")
                self.success_count += 1
                return True
            else:
                message = f"❌ {file_path} missing {description}"
                self.errors.append(message)
                print(message)
                return False
        except Exception as e:
            message = f"❌ Error reading {file_path}: {e}"
            self.errors.append(message)
            print(message)
            return False
    
    def validate_pyproject_toml(self) -> bool:
        """Validate pyproject.toml configuration"""
        print("\n🔍 Validating pyproject.toml...")
        
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                print("❌ Cannot validate pyproject.toml - tomllib/tomli not available")
                return False
        
        try:
            with open(self.project_root / "pyproject.toml", "rb") as f:
                data = tomllib.load(f)
            
            # Check essential fields
            project = data.get("project", {})
            required_fields = ["name", "version", "description", "authors", "license"]
            
            for field in required_fields:
                if field in project:
                    print(f"✅ pyproject.toml has {field}")
                    self.success_count += 1
                else:
                    message = f"❌ pyproject.toml missing {field}"
                    self.errors.append(message)
                    print(message)
                
                self.total_checks += 1
            
            # Check URLs
            urls = project.get("urls", {})
            if urls:
                print(f"✅ pyproject.toml has project URLs")
                self.success_count += 1
            else:
                message = "⚠️ pyproject.toml missing project URLs"
                self.warnings.append(message)
                print(message)
            
            self.total_checks += 1
            
            return True
            
        except Exception as e:
            message = f"❌ Error validating pyproject.toml: {e}"
            self.errors.append(message)
            print(message)
            return False
    
    def validate_github_files(self) -> bool:
        """Validate GitHub-specific files"""
        print("\n🔍 Validating GitHub configuration...")
        
        github_files = [
            ".github/workflows/ci.yml",
            ".github/workflows/publish.yml",
            ".github/workflows/dependency-check.yml",
            ".github/workflows/code-quality.yml",
            ".github/ISSUE_TEMPLATE/bug_report.md",
            ".github/ISSUE_TEMPLATE/feature_request.md",
            ".github/pull_request_template.md"
        ]
        
        for file_path in github_files:
            self.check_file_exists(file_path, required=True)
        
        return len(self.errors) == 0
    
    def validate_documentation(self) -> bool:
        """Validate documentation structure"""
        print("\n🔍 Validating documentation...")
        
        doc_files = [
            "README.md",
            "CONTRIBUTING.md", 
            "CHANGELOG.md",
            "SECURITY.md",
            "docs/user/installation.md",
            "docs/user/configuration.md",
            "docs/user/quick-start.md",
            "docs/user/examples.md"
        ]
        
        for file_path in doc_files:
            self.check_file_exists(file_path, required=True)
        
        return len(self.errors) == 0
    
    def validate_configuration(self) -> bool:
        """Validate configuration files"""
        print("\n🔍 Validating configuration...")
        
        config_files = [
            ".env.example",
            ".gitignore",
            ".pre-commit-config.yaml"
        ]
        
        for file_path in config_files:
            self.check_file_exists(file_path, required=True)
        
        # Check .gitignore content
        gitignore_patterns = [
            "__pycache__/",
            "*.pyc",
            ".env",
            ".venv",
            "dist/",
            "build/",
            "*.log"
        ]
        
        gitignore_path = self.project_root / ".gitignore"
        if gitignore_path.exists():
            content = gitignore_path.read_text()
            for pattern in gitignore_patterns:
                if pattern in content:
                    print(f"✅ .gitignore includes {pattern}")
                    self.success_count += 1
                else:
                    message = f"⚠️ .gitignore missing {pattern}"
                    self.warnings.append(message)
                    print(message)
                self.total_checks += 1
        
        return True
    
    def validate_license(self) -> bool:
        """Validate license file"""
        print("\n🔍 Validating license...")
        
        if self.check_file_exists("LICENSE", required=True):
            return self.check_file_content("LICENSE", "MIT License", "MIT license text")
        
        return False
    
    def check_scripts_executable(self) -> bool:
        """Check if scripts are executable"""
        print("\n🔍 Checking script permissions...")
        
        scripts = [
            "scripts/setup.sh",
            "scripts/dev.sh"
        ]
        
        for script in scripts:
            script_path = self.project_root / script
            if script_path.exists():
                if os.access(script_path, os.X_OK):
                    print(f"✅ {script} is executable")
                    self.success_count += 1
                else:
                    message = f"⚠️ {script} is not executable (run: chmod +x {script})"
                    self.warnings.append(message)
                    print(message)
            else:
                message = f"❌ {script} missing"
                self.errors.append(message)
                print(message)
            
            self.total_checks += 1
        
        return True
    
    def run_validation(self) -> bool:
        """Run complete validation"""
        print("🚀 Validating Crew-Camufox Open Source Release Setup...\n")
        
        # Core validation checks
        self.validate_license()
        self.validate_pyproject_toml()
        self.validate_configuration()
        self.validate_documentation()
        self.validate_github_files()
        self.check_scripts_executable()
        
        # Additional checks
        print("\n🔍 Checking additional directories...")
        essential_dirs = ["src", "tests", "examples", "scripts"]
        for dir_name in essential_dirs:
            self.check_directory_exists(dir_name, required=True)
        
        # Summary
        print(f"\n📊 Validation Summary:")
        print(f"✅ Successful checks: {self.success_count}/{self.total_checks}")
        
        if self.errors:
            print(f"\n❌ Critical Issues ({len(self.errors)}):")
            for error in self.errors:
                print(f"  {error}")
        
        if self.warnings:
            print(f"\n⚠️ Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  {warning}")
        
        if not self.errors:
            print(f"\n🎉 All critical checks passed! Ready for open source release.")
            if self.warnings:
                print(f"💡 Consider addressing the warnings above for best practices.")
            return True
        else:
            print(f"\n❌ Please fix the critical issues before releasing.")
            return False

def main():
    """Main validation function"""
    validator = ReleaseValidator()
    success = validator.run_validation()
    
    print(f"\n{'='*60}")
    if success:
        print("🎯 Next Steps for Open Source Release:")
        print("1. Review and update any placeholder URLs in pyproject.toml")
        print("2. Configure GitHub repository settings")
        print("3. Set up GitHub Actions secrets (PYPI_API_TOKEN)")
        print("4. Test the CI/CD pipeline with a test commit")
        print("5. Create your first release!")
        print("\n🚀 Your project is ready for the open source community!")
    else:
        print("🔧 Fix the issues above and run this script again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
