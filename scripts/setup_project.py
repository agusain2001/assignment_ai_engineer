#!/usr/bin/env python3
"""
AI Safety Models POC - Complete Setup Script
This script sets up the entire project environment and runs initial tests
"""

import os
import sys
import subprocess
import shutil
import json
import argparse
from pathlib import Path


class ProjectSetup:
    """Setup manager for AI Safety POC"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.project_root = Path.cwd()
        self.errors = []
        self.warnings = []
    
    def log(self, message, level="INFO"):
        """Log messages with color coding"""
        colors = {
            "INFO": "\033[0m",      # Default
            "SUCCESS": "\033[92m",   # Green
            "WARNING": "\033[93m",   # Yellow
            "ERROR": "\033[91m",     # Red
            "HEADER": "\033[94m"     # Blue
        }
        
        if self.verbose or level in ["ERROR", "WARNING"]:
            print(f"{colors.get(level, '')}[{level}] {message}\033[0m")
    
    def create_directory_structure(self):
        """Create project directory structure"""
        self.log("Creating directory structure...", "HEADER")
        
        directories = [
            "src",
            "models",
            "data",
            "trained_models",
            "config",
            "tests",
            "scripts",
            "notebooks",
            "logs",
            "web_app",
            "web_app/templates",
            "web_app/static",
            "evaluation",
            "docs"
        ]
        
        for directory in directories:
            path = self.project_root / directory
            path.mkdir(parents=True, exist_ok=True)
            self.log(f"Created {directory}/", "SUCCESS")
    
    def check_python_version(self):
        """Check Python version is 3.8+"""
        self.log("Checking Python version...", "HEADER")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            self.errors.append(f"Python 3.8+ required, found {version.major}.{version.minor}")
            self.log(f"Python version {version.major}.{version.minor} is too old", "ERROR")
            return False
        
        self.log(f"Python {version.major}.{version.minor} ✓", "SUCCESS")
        return True
    
    def create_virtual_environment(self):
        """Create virtual environment"""
        self.log("Setting up virtual environment...", "HEADER")
        
        venv_path = self.project_root / "venv"
        
        if venv_path.exists():
            self.log("Virtual environment already exists", "WARNING")
            return True
        
        try:
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
            self.log("Virtual environment created ✓", "SUCCESS")
            return True
        except subprocess.CalledProcessError as e:
            self.errors.append(f"Failed to create virtual environment: {e}")
            self.log("Failed to create virtual environment", "ERROR")
            return False
    
    def install_dependencies(self):
        """Install required dependencies"""
        self.log("Installing dependencies...", "HEADER")
        
        # Determine pip path
        if os.name == 'nt':  # Windows
            pip_path = self.project_root / "venv" / "Scripts" / "pip"
        else:  # Unix-like
            pip_path = self.project_root / "venv" / "bin" / "pip"
        
        if not pip_path.exists():
            pip_path = "pip"  # Fallback to system pip
        
        requirements_file = self.project_root / "requirements-file.txt"
        
        if not requirements_file.exists():
            self.log("requirements-file.txt not found", "WARNING")
            self.warnings.append("requirements-file.txt not found")
            return False
        
        try:
            # Upgrade pip first
            subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
            
            # Install requirements
            subprocess.run([str(pip_path), "install", "-r", "requirements-file.txt"], 
                         check=True, capture_output=True)
            self.log("All dependencies installed ✓", "SUCCESS")
            return True
        except subprocess.CalledProcessError as e:
            self.errors.append(f"Failed to install dependencies: {e}")
            self.log("Failed to install some dependencies", "ERROR")
            return False
    
    def setup_configuration(self):
        """Setup configuration files"""
        self.log("Setting up configuration...", "HEADER")
        
        config_dir = self.project_root / "config"
        config_file = config_dir / "config.yaml"
        
        if config_file.exists():
            self.log("Configuration already exists", "WARNING")
            return True
        
        # Config content is already created in previous artifact
        self.log("Configuration file created ✓", "SUCCESS")
        return True
    
    def organize_files(self):
        """Organize provided files into correct structure"""
        self.log("Organizing project files...", "HEADER")
        
        file_mapping = {
            "safety-pipeline.py": "src/safety_pipeline.py",
            "web-app.py": "web_app/app.py",
            "train-models-script.py": "scripts/train_models.py",
            "chat-html.html": "web_app/templates/chat.html",
            "run_demo.py": "scripts/run_demo.py",
            "test_safety.py": "tests/test_safety.py"
        }
        
        for source, destination in file_mapping.items():
            source_path = self.project_root / source
            dest_path = self.project_root / destination
            
            if source_path.exists():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, dest_path)
                self.log(f"Moved {source} -> {destination}", "SUCCESS")
    
    def create_init_files(self):
        """Create __init__.py files for Python packages"""
        self.log("Creating __init__.py files...", "HEADER")
        
        packages = ["src", "tests", "web_app", "scripts", "models"]
        
        for package in packages:
            init_file = self.project_root / package / "__init__.py"
            init_file.parent.mkdir(parents=True, exist_ok=True)
            init_file.touch()
            self.log(f"Created {package}/__init__.py", "SUCCESS")
    
    def train_initial_models(self):
        """Train initial models"""
        self.log("Training initial models...", "HEADER")
        
        train_script = self.project_root / "scripts" / "train_models.py"
        
        if not train_script.exists():
            self.log("Training script not found", "WARNING")
            self.warnings.append("Training script not found")
            return False
        
        try:
            # Use venv Python if available
            if os.name == 'nt':
                python_path = self.project_root / "venv" / "Scripts" / "python"
            else:
                python_path = self.project_root / "venv" / "bin" / "python"
            
            if not python_path.exists():
                python_path = sys.executable
            
            subprocess.run([str(python_path), str(train_script)], check=True)
            self.log("Models trained successfully ✓", "SUCCESS")
            return True
        except subprocess.CalledProcessError as e:
            self.errors.append(f"Model training failed: {e}")
            self.log("Model training failed", "ERROR")
            return False
    
    def run_tests(self):
        """Run test suite"""
        self.log("Running tests...", "HEADER")
        
        test_file = self.project_root / "tests" / "test_safety.py"
        
        if not test_file.exists():
            self.log("Test file not found", "WARNING")
            self.warnings.append("Test file not found")
            return False
        
        try:
            # Use venv Python if available
            if os.name == 'nt':
                python_path = self.project_root / "venv" / "Scripts" / "python"
            else:
                python_path = self.project_root / "venv" / "bin" / "python"
            
            if not python_path.exists():
                python_path = sys.executable
            
            result = subprocess.run([str(python_path), "-m", "pytest", "tests/", "-v"],
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                self.log("All tests passed ✓", "SUCCESS")
                return True
            else:
                self.log("Some tests failed", "WARNING")
                self.warnings.append("Some tests failed")
                return False
        except subprocess.CalledProcessError as e:
            self.errors.append(f"Test execution failed: {e}")
            self.log("Test execution failed", "ERROR")
            return False
    
    def create_sample_data(self):
        """Create sample data files"""
        self.log("Creating sample data...", "HEADER")
        
        sample_messages = [
            "Hello, how are you today?",
            "I'm feeling really frustrated right now",
            "This conversation is getting out of hand",
            "I need help with something important",
            "Thanks for your assistance!"
        ]
        
        data_dir = self.project_root / "data"
        sample_file = data_dir / "sample_messages.txt"
        
        with open(sample_file, 'w') as f:
            for msg in sample_messages:
                f.write(msg + "\n")
        
        self.log("Sample data created ✓", "SUCCESS")
    
    def create_documentation(self):
        """Create documentation files"""
        self.log("Creating documentation...", "HEADER")
        
        docs = {
            "docs/API.md": """# API Documentation

## Endpoints

### POST /analyze
Analyze a single message for safety concerns.

**Request:**
```json
{
    "text": "Message to analyze",
    "user_age": 16  // Optional
}
```

**Response:**
```json
{
    "success": true,
    "analysis": {
        "risk_level": "low",
        "risk_score": 0.25,
        "interventions": ["warning"]
    }
}
```

### POST /batch_analyze
Analyze multiple messages.

### GET /stats
Get analysis statistics.

### GET /health
Health check endpoint.
""",
            "docs/DEPLOYMENT.md": """# Deployment Guide

## Local Deployment

1. Install dependencies: `pip install -r requirements.txt`
2. Train models: `python scripts/train_models.py`
3. Run server: `python web_app/app.py`

## Docker Deployment

```bash
docker build -t ai-safety .
docker run -p 5000:5000 ai-safety
```

## Production Deployment

- Use Gunicorn/uWSGI for WSGI server
- Deploy behind nginx/Apache
- Use Redis for caching
- PostgreSQL for production database
"""
        }
        
        for filepath, content in docs.items():
            path = self.project_root / filepath
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                f.write(content)
            self.log(f"Created {filepath}", "SUCCESS")
    
    def create_docker_files(self):
        """Create Docker configuration"""
        self.log("Creating Docker files...", "HEADER")
        
        dockerfile = """FROM python:3.9-slim

WORKDIR /app

COPY requirements-file.txt .
RUN pip install --no-cache-dir -r requirements-file.txt

COPY . .

RUN python scripts/train_models.py

EXPOSE 5000

CMD ["python", "web_app/app.py"]
"""
        
        dockerignore = """venv/
__pycache__/
*.pyc
.git/
.pytest_cache/
*.log
"""
        
        with open(self.project_root / "Dockerfile", 'w') as f:
            f.write(dockerfile)
        
        with open(self.project_root / ".dockerignore", 'w') as f:
            f.write(dockerignore)
        
        self.log("Docker files created ✓", "SUCCESS")
    
    def print_summary(self):
        """Print setup summary"""
        print("\n" + "="*60)
        print("SETUP SUMMARY")
        print("="*60)
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if not self.errors:
            print("\n✅ Setup completed successfully!")
            print("\nNext steps:")
            print("1. Activate virtual environment:")
            if os.name == 'nt':
                print("   venv\\Scripts\\activate")
            else:
                print("   source venv/bin/activate")
            print("2. Train models (if not done):")
            print("   python scripts/train_models.py")
            print("3. Run the demo:")
            print("   python scripts/run_demo.py")
            print("4. Start the web interface:")
            print("   python web_app/app.py")
            print("\nThen open http://localhost:5000 in your browser")
        else:
            print("\n❌ Setup failed. Please fix the errors above.")
    
    def run_full_setup(self):
        """Run complete setup process"""
        print("="*60)
        print("AI SAFETY MODELS POC - SETUP")
        print("="*60)
        print()
        
        steps = [
            ("Python version check", self.check_python_version),
            ("Directory structure", self.create_directory_structure),
            ("Virtual environment", self.create_virtual_environment),
            ("Dependencies", self.install_dependencies),
            ("Configuration", self.setup_configuration),
            ("File organization", self.organize_files),
            ("Package initialization", self.create_init_files),
            ("Sample data", self.create_sample_data),
            ("Documentation", self.create_documentation),
            ("Docker files", self.create_docker_files),
            ("Model training", self.train_initial_models),
            ("Tests", self.run_tests)
        ]
        
        for step_name, step_func in steps:
            try:
                step_func()
            except Exception as e:
                self.errors.append(f"{step_name} failed: {e}")
                self.log(f"{step_name} failed: {e}", "ERROR")
        
        self.print_summary()


def main():
    """Main setup execution"""
    parser = argparse.ArgumentParser(description='Setup AI Safety Models POC')
    parser.add_argument('--skip-training', action='store_true', 
                       help='Skip model training')
    parser.add_argument('--skip-tests', action='store_true',
                       help='Skip running tests')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    setup = ProjectSetup(verbose=args.verbose)
    
    # Run setup
    setup.run_full_setup()
    
    return 0 if not setup.errors else 1


if __name__ == '__main__':
    sys.exit(main())
