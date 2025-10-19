#!/usr/bin/env python3
"""
Environment setup script for Thoth backend
"""
import os
import shutil
from pathlib import Path

def setup_environment():
    """Setup environment configuration files"""
    print("🔧 Setting up Thoth Environment Configuration...")
    print("=" * 50)
    
    # Check if .env already exists
    env_file = Path(".env")
    example_file = Path("env.example")
    
    if env_file.exists():
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ").lower()
        if response != 'y':
            print("❌ Setup cancelled")
            return
    
    if not example_file.exists():
        print("❌ env.example file not found!")
        print("Please make sure you're in the backend directory")
        return
    
    # Copy example to .env
    try:
        shutil.copy(example_file, env_file)
        print("✅ Created .env file from env.example")
        
        # Show current configuration
        print("\n📋 Current Configuration:")
        print("-" * 30)
        
        with open(env_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    print(f"  {key}: {value}")
        
        print("\n🎉 Environment setup complete!")
        print("\n📝 Next steps:")
        print("1. Edit .env file to customize your configuration")
        print("2. Run: uvicorn main:app --reload")
        print("3. Check ENVIRONMENT_CONFIG.md for detailed documentation")
        
    except Exception as e:
        print(f"❌ Error setting up environment: {e}")

if __name__ == "__main__":
    setup_environment()
