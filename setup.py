"""
VuenCode Setup Script
Quick setup for both local development and competition deployment.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, check=True):
    """Run shell command with error handling."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return None


def setup_local_development():
    """Set up local development environment."""
    print("=== Setting up VuenCode Local Development ===")
    
    # Create virtual environment
    print("\n1. Creating virtual environment...")
    run_command("python -m venv venv")
    
    # Activate and install dependencies
    print("\n2. Installing dependencies...")
    if sys.platform == "win32":
        pip_cmd = "venv\\Scripts\\pip"
        python_cmd = "venv\\Scripts\\python"
    else:
        pip_cmd = "venv/bin/pip"
        python_cmd = "venv/bin/python"
    
    run_command(f"{pip_cmd} install --upgrade pip")
    run_command(f"{pip_cmd} install -r docker/requirements-local.txt")
    
    # Copy local config
    print("\n3. Setting up configuration...")
    if not os.path.exists(".env"):
        run_command("cp configs/local.env .env")
        print("Local configuration created (.env)")
    
    # Create cache directory
    os.makedirs("cache", exist_ok=True)
    print("Cache directory created")
    
    print("\nLocal development setup complete!")
    print("\nTo run the server:")
    print(f"  {python_cmd} -m uvicorn api.main:app --reload")
    print("\nTo run tests:")
    print(f"  {python_cmd} -m pytest tests/ -v")


def setup_docker_local():
    """Set up Docker local development."""
    print("=== Setting up VuenCode Docker Local ===")
    
    # Build and run local container
    print("\n1. Building Docker image...")
    run_command("docker-compose -f docker/docker-compose.yml --profile local build vuencode-local")
    
    print("\n2. Starting services...")
    run_command("docker-compose -f docker/docker-compose.yml --profile local up -d vuencode-local")
    
    print("\n3. Waiting for service to start...")
    import time
    time.sleep(10)
    
    # Health check
    print("\n4. Running health check...")
    health_check = run_command("curl -f http://localhost:8000/health", check=False)
    
    if health_check and health_check.returncode == 0:
        print("Service is healthy!")
        print("\nService available at: http://localhost:8000")
        print("API docs available at: http://localhost:8000/docs")
    else:
        print("Service health check failed")
        print("Check logs: docker-compose -f docker/docker-compose.yml logs vuencode-local")
    
    print("\nTo stop services:")
    print("  docker-compose -f docker/docker-compose.yml --profile local down")


def run_quick_test():
    """Run quick functionality test."""
    print("=== Running Quick Test ===")
    
    # Test with pytest
    test_result = run_command("python -m pytest tests/test_api.py::TestVuenCodeAPI::test_health_endpoint -v", check=False)
    
    if test_result and test_result.returncode == 0:
        print("Quick test passed!")
    else:
        print("Quick test failed")
        return False
    
    return True


def main():
    """Main setup function."""
    print("VuenCode Competition System Setup")
    print("=====================================")
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("Python 3.9+ required")
        sys.exit(1)
    
    print(f"Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Check if we're in the right directory
    if not Path("api/main.py").exists():
        print("Please run this script from the VuenCode root directory")
        sys.exit(1)
    
    # Setup options
    print("\nSetup Options:")
    print("1. Local development (Python virtual environment)")
    print("2. Docker local development") 
    print("3. Quick test only")
    print("4. All of the above")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        setup_local_development()
    elif choice == "2":
        setup_docker_local()
    elif choice == "3":
        run_quick_test()
    elif choice == "4":
        setup_local_development()
        print("\n" + "="*50)
        setup_docker_local()
        print("\n" + "="*50)
        run_quick_test()
    else:
        print("Invalid choice")
        sys.exit(1)
    
    print("\nSetup complete! Ready for competition development.")
    print("\nNext steps:")
    print("1. Add your Gemini API key to configs/competition.env")
    print("2. Test the system with: python -m pytest tests/")
    print("3. For GPU deployment, use: docker-compose --profile competition up")


if __name__ == "__main__":
    main()