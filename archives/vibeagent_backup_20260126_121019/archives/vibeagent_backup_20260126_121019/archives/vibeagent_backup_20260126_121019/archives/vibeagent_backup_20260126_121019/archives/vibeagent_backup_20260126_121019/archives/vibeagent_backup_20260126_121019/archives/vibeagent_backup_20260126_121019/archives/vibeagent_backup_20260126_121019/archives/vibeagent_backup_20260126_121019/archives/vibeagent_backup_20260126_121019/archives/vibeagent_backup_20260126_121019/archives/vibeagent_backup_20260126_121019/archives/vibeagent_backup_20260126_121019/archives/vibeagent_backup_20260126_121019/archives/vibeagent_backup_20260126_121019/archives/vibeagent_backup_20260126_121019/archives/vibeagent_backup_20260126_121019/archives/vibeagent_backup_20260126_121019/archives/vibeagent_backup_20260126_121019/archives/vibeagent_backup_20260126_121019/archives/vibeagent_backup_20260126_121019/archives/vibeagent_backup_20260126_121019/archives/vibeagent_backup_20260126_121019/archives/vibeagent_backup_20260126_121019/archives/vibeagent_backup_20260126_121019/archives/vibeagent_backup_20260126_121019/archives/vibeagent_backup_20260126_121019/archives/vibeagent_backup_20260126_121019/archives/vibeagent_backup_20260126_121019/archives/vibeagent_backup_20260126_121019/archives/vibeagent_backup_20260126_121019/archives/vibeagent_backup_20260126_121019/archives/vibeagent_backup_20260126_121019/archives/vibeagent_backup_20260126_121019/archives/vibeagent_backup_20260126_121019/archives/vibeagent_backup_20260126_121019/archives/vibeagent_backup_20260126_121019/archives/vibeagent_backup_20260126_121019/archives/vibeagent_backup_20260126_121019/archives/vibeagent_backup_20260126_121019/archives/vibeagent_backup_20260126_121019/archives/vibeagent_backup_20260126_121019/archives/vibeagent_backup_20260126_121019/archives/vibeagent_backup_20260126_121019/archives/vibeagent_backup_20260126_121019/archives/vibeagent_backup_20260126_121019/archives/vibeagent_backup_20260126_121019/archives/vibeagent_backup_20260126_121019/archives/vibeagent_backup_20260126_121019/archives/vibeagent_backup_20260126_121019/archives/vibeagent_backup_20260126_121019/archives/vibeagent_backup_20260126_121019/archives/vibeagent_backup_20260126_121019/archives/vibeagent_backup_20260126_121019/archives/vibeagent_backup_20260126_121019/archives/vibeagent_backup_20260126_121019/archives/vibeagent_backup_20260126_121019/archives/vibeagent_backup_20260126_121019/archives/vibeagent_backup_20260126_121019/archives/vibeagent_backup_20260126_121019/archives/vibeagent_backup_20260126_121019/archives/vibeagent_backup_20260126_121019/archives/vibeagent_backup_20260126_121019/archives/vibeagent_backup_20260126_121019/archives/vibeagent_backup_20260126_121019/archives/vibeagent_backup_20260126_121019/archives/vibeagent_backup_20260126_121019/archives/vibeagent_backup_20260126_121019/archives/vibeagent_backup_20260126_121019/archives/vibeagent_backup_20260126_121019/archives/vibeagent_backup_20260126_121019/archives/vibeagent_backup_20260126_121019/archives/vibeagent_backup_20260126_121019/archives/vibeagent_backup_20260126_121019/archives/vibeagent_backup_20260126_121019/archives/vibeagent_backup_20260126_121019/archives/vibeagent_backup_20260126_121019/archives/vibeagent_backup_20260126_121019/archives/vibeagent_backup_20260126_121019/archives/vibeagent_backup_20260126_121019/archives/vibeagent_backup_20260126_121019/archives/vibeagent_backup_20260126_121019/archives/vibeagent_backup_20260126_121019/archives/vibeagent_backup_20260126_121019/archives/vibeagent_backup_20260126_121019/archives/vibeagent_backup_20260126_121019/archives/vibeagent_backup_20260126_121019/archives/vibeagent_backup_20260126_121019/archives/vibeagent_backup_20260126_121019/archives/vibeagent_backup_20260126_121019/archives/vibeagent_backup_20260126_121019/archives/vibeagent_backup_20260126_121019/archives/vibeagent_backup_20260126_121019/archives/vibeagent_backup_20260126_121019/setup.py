#!/usr/bin/env python3
"""
Quick setup script for VibeAgent dependencies
"""

import subprocess
import sys


def install_package(package):
    """Install a Python package."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    print("Installing VibeAgent dependencies...")

    packages = [
        "pyyaml",
        "flask",  # For dashboard web interface
        "pytest",  # For testing
    ]

    for package in packages:
        print(f"\nInstalling {package}...")
        if install_package(package):
            print(f"✓ {package} installed successfully")
        else:
            print(f"✗ Failed to install {package}")

    print("\n✓ Setup complete!")
    print("\nNext steps:")
    print("  1. Initialize database: python3 scripts/init_db.py")
    print("  2. Run verification: python3 verify_implementation.py")
    print("  3. Run tests: pytest tests/")


if __name__ == "__main__":
    main()
