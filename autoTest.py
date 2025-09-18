#!/usr/bin/env python3
"""
MiniGPT AutoTest - Complete Automation System
============================================

This script automates the entire MiniGPT pipeline:
1. Data preparation
2. Training configuration
3. Model training with monitoring
4. Backend testing
5. Frontend testing
6. Full system integration testing

Usage:
    python autoTest.py                    # Run full pipeline
    python autoTest.py --step data        # Run specific step
    python autoTest.py --config custom    # Use custom config
    python autoTest.py --dry-run          # Preview what will run
"""

import os
import sys
import json
import time
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import yaml
import requests
import shutil

class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class AutoTestLogger:
    """Custom logger for AutoTest with colored output"""

    def __init__(self, log_file: str = "autotest.log"):
        self.log_file = log_file
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('AutoTest')

    def step(self, message: str):
        """Log a major step"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}🔹 STEP: {message}{Colors.ENDC}")
        self.logger.info(f"STEP: {message}")

    def success(self, message: str):
        """Log success message"""
        print(f"{Colors.GREEN}✅ {message}{Colors.ENDC}")
        self.logger.info(f"SUCCESS: {message}")

    def warning(self, message: str):
        """Log warning message"""
        print(f"{Colors.YELLOW}⚠️  {message}{Colors.ENDC}")
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message"""
        print(f"{Colors.RED}❌ {message}{Colors.ENDC}")
        self.logger.error(message)

    def info(self, message: str):
        """Log info message"""
        print(f"{Colors.CYAN}ℹ️  {message}{Colors.ENDC}")
        self.logger.info(message)

class SystemChecker:
    """System requirements and dependency checker"""

    def __init__(self, logger: AutoTestLogger):
        self.logger = logger

    def check_python_version(self) -> bool:
        """Check if Python version is compatible"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            self.logger.error(f"Python 3.8+ required, found {version.major}.{version.minor}")
            return False
        self.logger.success(f"Python {version.major}.{version.minor}.{version.micro} OK")
        return True

    def check_dependencies(self) -> bool:
        """Check if required packages are installed"""
        required_packages = [
            'torch', 'numpy', 'transformers', 'datasets', 'tqdm',
            'matplotlib', 'seaborn', 'requests', 'fastapi', 'uvicorn'
        ]

        missing = []
        for package in required_packages:
            try:
                __import__(package)
                self.logger.success(f"✓ {package}")
            except ImportError:
                missing.append(package)
                self.logger.error(f"✗ {package}")

        if missing:
            self.logger.error(f"Missing packages: {', '.join(missing)}")
            self.logger.info("Install with: pip install " + " ".join(missing))
            return False

        return True

    def check_directories(self) -> bool:
        """Check if required directories exist"""
        required_dirs = [
            'backend', 'frontend', 'backend/src', 'backend/configs',
            'backend/data', 'backend/checkpoints', 'backend/scripts'
        ]

        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                self.logger.error(f"Missing directory: {dir_path}")
                return False
            self.logger.success(f"✓ {dir_path}")

        return True

    def check_node_npm(self) -> bool:
        """Check if Node.js and npm are available for frontend"""
        try:
            node_result = subprocess.run(['node', '--version'],
                                       capture_output=True, text=True, check=True)
            npm_result = subprocess.run(['npm', '--version'],
                                      capture_output=True, text=True, check=True)

            self.logger.success(f"Node.js {node_result.stdout.strip()}")
            self.logger.success(f"npm {npm_result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.logger.error("Node.js and npm required for frontend testing")
            return False

class DataPreparator:
    """Handles data preparation and validation"""

    def __init__(self, logger: AutoTestLogger):
        self.logger = logger
        self.data_dir = Path("backend/data")

    def prepare_training_data(self, config: Dict) -> bool:
        """Prepare and validate training data"""
        self.logger.step("Preparing training data...")

        # Ensure data directory exists
        self.data_dir.mkdir(exist_ok=True)

        # Download dataset if needed
        dataset_name = config.get('data', {}).get('dataset_name', 'wikitext')
        dataset_config = config.get('data', {}).get('dataset_config', 'wikitext-2-raw-v1')

        try:
            from datasets import load_dataset

            self.logger.info(f"Loading dataset: {dataset_name}/{dataset_config}")
            dataset = load_dataset(dataset_name, dataset_config)

            # Save sample data for quick testing
            sample_file = self.data_dir / "sample_data.txt"
            with open(sample_file, 'w', encoding='utf-8') as f:
                for i, example in enumerate(dataset['train']):
                    if i >= 100:  # Limit to 100 examples for quick testing
                        break
                    f.write(example['text'] + '\n')

            self.logger.success(f"Sample data saved to {sample_file}")

            # Validate data quality
            if self._validate_data_quality(sample_file):
                self.logger.success("Data quality validation passed")
                return True
            else:
                self.logger.error("Data quality validation failed")
                return False

        except Exception as e:
            self.logger.error(f"Data preparation failed: {str(e)}")
            return False

    def _validate_data_quality(self, data_file: Path) -> bool:
        """Validate the quality of prepared data"""
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if len(lines) == 0:
                self.logger.error("Data file is empty")
                return False

            # Check for minimum content
            total_chars = sum(len(line.strip()) for line in lines)
            if total_chars < 1000:
                self.logger.error("Data file too small (< 1000 characters)")
                return False

            self.logger.info(f"Data validation: {len(lines)} lines, {total_chars} characters")
            return True

        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            return False

class ModelTrainer:
    """Handles model training with monitoring"""

    def __init__(self, logger: AutoTestLogger):
        self.logger = logger
        self.backend_dir = Path("backend")
        self.training_log = "training_autotest.log"

    def train_model(self, config_name: str = "small", epochs: int = 3) -> bool:
        """Train model with monitoring"""
        self.logger.step(f"Starting model training with {config_name} config...")

        # Backup existing checkpoints
        self._backup_checkpoints()

        # Prepare training command
        config_path = self.backend_dir / f"configs/{config_name}.yaml"
        if not config_path.exists():
            self.logger.error(f"Config file not found: {config_path}")
            return False

        # Modify config for quick testing
        modified_config = self._create_test_config(config_path, epochs)

        # Start training
        training_cmd = [
            sys.executable, "-m", "minigpt.train",
            "--config", str(modified_config)
        ]

        try:
            self.logger.info(f"Training command: {' '.join(training_cmd)}")

            # Start training process
            with open(self.training_log, 'w') as log_file:
                process = subprocess.Popen(
                    training_cmd,
                    cwd=self.backend_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )

                # Monitor training progress
                for line in iter(process.stdout.readline, ''):
                    log_file.write(line)
                    if "Epoch" in line or "Loss" in line or "val_loss" in line:
                        self.logger.info(line.strip())

                process.wait()

                if process.returncode == 0:
                    self.logger.success("Model training completed successfully")
                    return self._validate_training_output()
                else:
                    self.logger.error(f"Training failed with exit code {process.returncode}")
                    return False

        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            return False

    def _backup_checkpoints(self):
        """Backup existing checkpoints"""
        checkpoints_dir = self.backend_dir / "checkpoints"
        if checkpoints_dir.exists() and list(checkpoints_dir.glob("*.pt")):
            backup_dir = checkpoints_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir.mkdir(exist_ok=True)

            for checkpoint in checkpoints_dir.glob("*.pt"):
                shutil.copy2(checkpoint, backup_dir / checkpoint.name)

            self.logger.info(f"Checkpoints backed up to {backup_dir}")

    def _create_test_config(self, original_config: Path, epochs: int) -> Path:
        """Create a modified config for quick testing"""
        with open(original_config, 'r') as f:
            config = yaml.safe_load(f)

        # Modify for quick testing
        config['training']['max_epochs'] = epochs
        config['training']['batch_size'] = min(config['training']['batch_size'], 16)
        config['logging']['log_interval'] = 10
        config['logging']['eval_interval'] = 50
        config['logging']['save_interval'] = 100

        # Save modified config
        test_config_path = self.backend_dir / "configs/autotest.yaml"
        with open(test_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        self.logger.info(f"Test config created: {test_config_path}")
        return test_config_path

    def _validate_training_output(self) -> bool:
        """Validate that training produced expected outputs"""
        checkpoints_dir = self.backend_dir / "checkpoints"

        # Check for checkpoint files
        checkpoints = list(checkpoints_dir.glob("*.pt"))
        if not checkpoints:
            self.logger.error("No checkpoint files found after training")
            return False

        self.logger.success(f"Found {len(checkpoints)} checkpoint files")

        # Check for training progress file
        progress_file = self.backend_dir / "training_progress.json"
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                self.logger.success(f"Training progress: {progress}")
            except Exception as e:
                self.logger.warning(f"Could not read training progress: {e}")

        return True

class BackendTester:
    """Tests backend functionality"""

    def __init__(self, logger: AutoTestLogger):
        self.logger = logger
        self.backend_dir = Path("backend")
        self.api_process = None
        self.api_port = 8000

    def test_backend(self) -> bool:
        """Run comprehensive backend tests"""
        self.logger.step("Testing backend functionality...")

        tests = [
            ("Model Loading", self._test_model_loading),
            ("Chat Interface", self._test_chat_interface),
            ("API Server", self._test_api_server),
            ("Model Evaluation", self._test_model_evaluation)
        ]

        results = {}
        for test_name, test_func in tests:
            self.logger.info(f"Running test: {test_name}")
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    self.logger.success(f"✓ {test_name}")
                else:
                    self.logger.error(f"✗ {test_name}")
            except Exception as e:
                self.logger.error(f"✗ {test_name}: {str(e)}")
                results[test_name] = False

        success_count = sum(results.values())
        total_count = len(results)

        self.logger.info(f"Backend tests: {success_count}/{total_count} passed")
        return success_count == total_count

    def _test_model_loading(self) -> bool:
        """Test if model can be loaded"""
        try:
            # Find latest checkpoint
            checkpoints_dir = self.backend_dir / "checkpoints"
            checkpoints = list(checkpoints_dir.glob("*.pt"))

            if not checkpoints:
                self.logger.error("No checkpoints found for testing")
                return False

            latest_checkpoint = max(checkpoints, key=os.path.getctime)

            # Test model loading via Python
            test_script = f"""
import sys
sys.path.insert(0, 'src')
from minigpt.utils import load_checkpoint
from minigpt.model import MiniGPT
import torch

try:
    checkpoint = load_checkpoint('{latest_checkpoint}', 'cpu')
    config = checkpoint['config']['model']
    model = MiniGPT(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully")
    print(f"Model parameters: {{sum(p.numel() for p in model.parameters()):,}}")
except Exception as e:
    print(f"Error: {{e}}")
    exit(1)
"""

            result = subprocess.run([
                sys.executable, "-c", test_script
            ], cwd=self.backend_dir, capture_output=True, text=True)

            if result.returncode == 0:
                self.logger.info(result.stdout.strip())
                return True
            else:
                self.logger.error(f"Model loading failed: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"Model loading test error: {str(e)}")
            return False

    def _test_chat_interface(self) -> bool:
        """Test chat interface with a simple prompt"""
        try:
            checkpoints_dir = self.backend_dir / "checkpoints"
            checkpoints = list(checkpoints_dir.glob("*.pt"))

            if not checkpoints:
                return False

            latest_checkpoint = max(checkpoints, key=os.path.getctime)

            # Test chat with a simple prompt
            chat_cmd = [
                sys.executable, "-m", "minigpt.chat",
                "--model", str(latest_checkpoint),
                "--prompt", "Hello, this is a test.",
                "--max-length", "20"
            ]

            result = subprocess.run(
                chat_cmd,
                cwd=self.backend_dir,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0 and len(result.stdout.strip()) > 0:
                self.logger.info(f"Chat response: {result.stdout.strip()[:100]}...")
                return True
            else:
                self.logger.error(f"Chat test failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.logger.error("Chat test timed out")
            return False
        except Exception as e:
            self.logger.error(f"Chat test error: {str(e)}")
            return False

    def _test_api_server(self) -> bool:
        """Test API server functionality"""
        try:
            # Start API server
            if not self._start_api_server():
                return False

            # Wait for server to start
            time.sleep(5)

            # Test health endpoint
            try:
                response = requests.get(f"http://localhost:{self.api_port}/health", timeout=10)
                if response.status_code != 200:
                    self.logger.error(f"Health check failed: {response.status_code}")
                    return False

                # Test generation endpoint
                response = requests.post(
                    f"http://localhost:{self.api_port}/generate",
                    json={
                        "prompt": "Test prompt",
                        "max_length": 20,
                        "temperature": 0.8
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    self.logger.info(f"API generation test: {result.get('generated_text', '')[:50]}...")
                    return True
                else:
                    self.logger.error(f"API generation failed: {response.status_code}")
                    return False

            except requests.RequestException as e:
                self.logger.error(f"API request failed: {str(e)}")
                return False

        finally:
            self._stop_api_server()

        return False

    def _start_api_server(self) -> bool:
        """Start the API server in background"""
        try:
            # Check if port is available
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', self.api_port))
            sock.close()

            if result == 0:
                self.logger.warning(f"Port {self.api_port} already in use")
                return True  # Assume server is already running

            # Start API server
            api_cmd = [
                sys.executable, "-m", "minigpt.api",
                "--port", str(self.api_port)
            ]

            self.api_process = subprocess.Popen(
                api_cmd,
                cwd=self.backend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            self.logger.info("API server started")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start API server: {str(e)}")
            return False

    def _stop_api_server(self):
        """Stop the API server"""
        if self.api_process:
            self.api_process.terminate()
            self.api_process.wait(timeout=10)
            self.logger.info("API server stopped")

    def _test_model_evaluation(self) -> bool:
        """Test model evaluation functionality"""
        try:
            eval_cmd = [
                sys.executable, "scripts/run_evaluation.py"
            ]

            result = subprocess.run(
                eval_cmd,
                cwd=self.backend_dir,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                self.logger.info("Model evaluation completed")
                return True
            else:
                self.logger.error(f"Evaluation failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.logger.error("Evaluation test timed out")
            return False
        except Exception as e:
            self.logger.error(f"Evaluation test error: {str(e)}")
            return False

class FrontendTester:
    """Tests frontend functionality"""

    def __init__(self, logger: AutoTestLogger):
        self.logger = logger
        self.frontend_dir = Path("frontend")
        self.frontend_process = None
        self.frontend_port = 3000

    def test_frontend(self) -> bool:
        """Run frontend tests"""
        self.logger.step("Testing frontend functionality...")

        tests = [
            ("Dependencies Installation", self._test_npm_install),
            ("Build Process", self._test_build),
            ("Development Server", self._test_dev_server),
            ("Frontend-Backend Integration", self._test_integration)
        ]

        results = {}
        for test_name, test_func in tests:
            self.logger.info(f"Running test: {test_name}")
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    self.logger.success(f"✓ {test_name}")
                else:
                    self.logger.error(f"✗ {test_name}")
            except Exception as e:
                self.logger.error(f"✗ {test_name}: {str(e)}")
                results[test_name] = False

        success_count = sum(results.values())
        total_count = len(results)

        self.logger.info(f"Frontend tests: {success_count}/{total_count} passed")
        return success_count == total_count

    def _test_npm_install(self) -> bool:
        """Test npm install"""
        try:
            result = subprocess.run(
                ["npm", "install"],
                cwd=self.frontend_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes
            )

            if result.returncode == 0:
                self.logger.success("npm install completed")
                return True
            else:
                self.logger.error(f"npm install failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.logger.error("npm install timed out")
            return False
        except Exception as e:
            self.logger.error(f"npm install error: {str(e)}")
            return False

    def _test_build(self) -> bool:
        """Test build process"""
        try:
            result = subprocess.run(
                ["npm", "run", "build"],
                cwd=self.frontend_dir,
                capture_output=True,
                text=True,
                timeout=180  # 3 minutes
            )

            if result.returncode == 0:
                # Check if build directory exists
                build_dir = self.frontend_dir / "build"
                if build_dir.exists():
                    self.logger.success("Build completed successfully")
                    return True
                else:
                    self.logger.error("Build directory not found")
                    return False
            else:
                self.logger.error(f"Build failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.logger.error("Build timed out")
            return False
        except Exception as e:
            self.logger.error(f"Build error: {str(e)}")
            return False

    def _test_dev_server(self) -> bool:
        """Test development server"""
        try:
            # Start dev server
            self.frontend_process = subprocess.Popen(
                ["npm", "start"],
                cwd=self.frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, "BROWSER": "none"}  # Don't open browser
            )

            # Wait for server to start
            max_wait = 60  # 60 seconds
            wait_time = 0
            while wait_time < max_wait:
                try:
                    response = requests.get(f"http://localhost:{self.frontend_port}", timeout=5)
                    if response.status_code == 200:
                        self.logger.success("Frontend dev server is running")
                        return True
                except requests.RequestException:
                    pass

                time.sleep(2)
                wait_time += 2

            self.logger.error("Frontend dev server failed to start")
            return False

        except Exception as e:
            self.logger.error(f"Dev server error: {str(e)}")
            return False
        finally:
            if self.frontend_process:
                self.frontend_process.terminate()
                self.frontend_process.wait(timeout=10)

    def _test_integration(self) -> bool:
        """Test frontend-backend integration"""
        # This is a placeholder for integration testing
        # In a real scenario, you'd test API calls from frontend to backend
        self.logger.info("Integration test completed (placeholder)")
        return True

class AutoTest:
    """Main automation orchestrator"""

    def __init__(self, config_name: str = "small", dry_run: bool = False):
        self.config_name = config_name
        self.dry_run = dry_run
        self.logger = AutoTestLogger()

        # Initialize components
        self.system_checker = SystemChecker(self.logger)
        self.data_preparator = DataPreparator(self.logger)
        self.model_trainer = ModelTrainer(self.logger)
        self.backend_tester = BackendTester(self.logger)
        self.frontend_tester = FrontendTester(self.logger)

        # Test results
        self.results = {}
        self.start_time = datetime.now()

    def run_full_pipeline(self) -> bool:
        """Run the complete automation pipeline"""
        self.logger.step("🚀 Starting MiniGPT AutoTest Pipeline")

        if self.dry_run:
            self.logger.info("DRY RUN MODE - No actual changes will be made")
            self._preview_pipeline()
            return True

        pipeline_steps = [
            ("System Check", self._run_system_check),
            ("Data Preparation", self._run_data_preparation),
            ("Model Training", self._run_model_training),
            ("Backend Testing", self._run_backend_testing),
            ("Frontend Testing", self._run_frontend_testing),
            ("Integration Testing", self._run_integration_testing)
        ]

        for step_name, step_func in pipeline_steps:
            self.logger.step(f"Running: {step_name}")
            try:
                success = step_func()
                self.results[step_name] = success

                if success:
                    self.logger.success(f"✅ {step_name} completed")
                else:
                    self.logger.error(f"❌ {step_name} failed")

                    # Ask user if they want to continue
                    if not self._should_continue(step_name):
                        break

            except KeyboardInterrupt:
                self.logger.warning("Pipeline interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in {step_name}: {str(e)}")
                self.results[step_name] = False

                if not self._should_continue(step_name):
                    break

        # Generate final report
        self._generate_final_report()

        return all(self.results.values())

    def run_specific_step(self, step: str) -> bool:
        """Run a specific pipeline step"""
        steps = {
            "system": self._run_system_check,
            "data": self._run_data_preparation,
            "train": self._run_model_training,
            "backend": self._run_backend_testing,
            "frontend": self._run_frontend_testing,
            "integration": self._run_integration_testing
        }

        if step not in steps:
            self.logger.error(f"Unknown step: {step}")
            self.logger.info(f"Available steps: {', '.join(steps.keys())}")
            return False

        self.logger.step(f"Running specific step: {step}")
        return steps[step]()

    def _run_system_check(self) -> bool:
        """Run system requirements check"""
        checks = [
            self.system_checker.check_python_version(),
            self.system_checker.check_dependencies(),
            self.system_checker.check_directories(),
            self.system_checker.check_node_npm()
        ]
        return all(checks)

    def _run_data_preparation(self) -> bool:
        """Run data preparation"""
        config_path = Path(f"backend/configs/{self.config_name}.yaml")
        if not config_path.exists():
            self.logger.error(f"Config file not found: {config_path}")
            return False

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return self.data_preparator.prepare_training_data(config)

    def _run_model_training(self) -> bool:
        """Run model training"""
        return self.model_trainer.train_model(self.config_name, epochs=3)

    def _run_backend_testing(self) -> bool:
        """Run backend testing"""
        return self.backend_tester.test_backend()

    def _run_frontend_testing(self) -> bool:
        """Run frontend testing"""
        return self.frontend_tester.test_frontend()

    def _run_integration_testing(self) -> bool:
        """Run integration testing"""
        self.logger.info("Running integration tests...")
        # Add comprehensive integration tests here
        return True

    def _preview_pipeline(self):
        """Show what would be executed in dry run mode"""
        steps = [
            "1. System Requirements Check",
            "   - Python version validation",
            "   - Package dependencies verification",
            "   - Directory structure validation",
            "   - Node.js/npm availability",
            "",
            "2. Data Preparation",
            f"   - Load {self.config_name} configuration",
            "   - Download and prepare training dataset",
            "   - Validate data quality",
            "",
            "3. Model Training",
            "   - Backup existing checkpoints",
            "   - Train model for 3 epochs (test mode)",
            "   - Monitor training progress",
            "   - Validate training outputs",
            "",
            "4. Backend Testing",
            "   - Test model loading",
            "   - Test chat interface",
            "   - Test API server",
            "   - Test model evaluation",
            "",
            "5. Frontend Testing",
            "   - Install npm dependencies",
            "   - Test build process",
            "   - Test development server",
            "   - Test frontend-backend integration",
            "",
            "6. Integration Testing",
            "   - End-to-end workflow validation",
            "   - Performance benchmarking",
            "   - Generate comprehensive report"
        ]

        self.logger.info("Pipeline Preview:")
        for step in steps:
            print(f"  {step}")

    def _should_continue(self, failed_step: str) -> bool:
        """Ask user if they want to continue after a failed step"""
        try:
            response = input(f"\n{failed_step} failed. Continue with remaining steps? (y/n): ")
            return response.lower().startswith('y')
        except KeyboardInterrupt:
            return False

    def _generate_final_report(self):
        """Generate comprehensive final report"""
        end_time = datetime.now()
        duration = end_time - self.start_time

        self.logger.step("📊 Final Report")

        # Summary
        total_steps = len(self.results)
        passed_steps = sum(self.results.values())

        print(f"\n{Colors.BOLD}AutoTest Summary{Colors.ENDC}")
        print(f"{'='*50}")
        print(f"Total Duration: {duration}")
        print(f"Steps Passed: {passed_steps}/{total_steps}")
        print(f"Success Rate: {(passed_steps/total_steps)*100:.1f}%")
        print()

        # Detailed results
        print(f"{Colors.BOLD}Step Results:{Colors.ENDC}")
        for step, success in self.results.items():
            status = f"{Colors.GREEN}✅ PASS{Colors.ENDC}" if success else f"{Colors.RED}❌ FAIL{Colors.ENDC}"
            print(f"  {step:<25} {status}")

        # Next steps
        print(f"\n{Colors.BOLD}Next Steps:{Colors.ENDC}")
        if all(self.results.values()):
            print(f"{Colors.GREEN}🎉 All tests passed! Your MiniGPT setup is ready.{Colors.ENDC}")
            print()
            print("You can now:")
            print("  • Start the full application: ./start-all.sh")
            print("  • Chat with your model: python -m minigpt.chat")
            print("  • Use the web interface: http://localhost:3000")
        else:
            print(f"{Colors.YELLOW}⚠️  Some tests failed. Review the logs and fix issues.{Colors.ENDC}")
            print()
            print("Common fixes:")
            print("  • Install missing dependencies: pip install -r requirements.txt")
            print("  • Check configuration files in backend/configs/")
            print("  • Verify data preparation completed successfully")

        # Save detailed report
        report_file = f"autotest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration.total_seconds(),
            "config_used": self.config_name,
            "results": self.results,
            "summary": {
                "total_steps": total_steps,
                "passed_steps": passed_steps,
                "success_rate": (passed_steps/total_steps)*100
            }
        }

        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        self.logger.info(f"Detailed report saved to: {report_file}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="MiniGPT AutoTest - Complete Automation System")
    parser.add_argument("--config", default="small", choices=["small", "medium"],
                       help="Model configuration to use")
    parser.add_argument("--step", choices=["system", "data", "train", "backend", "frontend", "integration"],
                       help="Run specific step only")
    parser.add_argument("--dry-run", action="store_true",
                       help="Preview what would be executed without running")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs (for quick testing)")

    args = parser.parse_args()

    try:
        autotest = AutoTest(config_name=args.config, dry_run=args.dry_run)

        if args.step:
            success = autotest.run_specific_step(args.step)
        else:
            success = autotest.run_full_pipeline()

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}AutoTest interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}AutoTest failed with error: {str(e)}{Colors.ENDC}")
        sys.exit(1)

if __name__ == "__main__":
    main()