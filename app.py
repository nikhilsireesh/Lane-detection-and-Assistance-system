#!/usr/bin/env python3
"""
Entry point for Render deployment
This file imports and runs the main Flask application
"""

import os
import sys
import importlib.util

# Add the project root and subdirectories to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'phase7_real_time_processing'))
sys.path.insert(0, os.path.join(project_root, 'phase7_real_time_processing', 'scripts'))

print(f"Python path: {sys.path[:3]}")
print(f"Current working directory: {os.getcwd()}")
print(f"Project root: {project_root}")

# Try multiple import methods
app = None

# Method 1: Direct import
try:
    from phase7_real_time_processing.scripts.web_demo import app
    print("✅ Successfully imported app using method 1")
except ImportError as e:
    print(f"❌ Method 1 failed: {e}")
    
    # Method 2: Import from scripts directly
    try:
        from scripts.web_demo import app
        print("✅ Successfully imported app using method 2")
    except ImportError as e:
        print(f"❌ Method 2 failed: {e}")
        
        # Method 3: Direct file import
        try:
            web_demo_path = os.path.join(project_root, 'phase7_real_time_processing', 'scripts', 'web_demo.py')
            spec = importlib.util.spec_from_file_location("web_demo", web_demo_path)
            web_demo_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(web_demo_module)
            app = web_demo_module.app
            print("✅ Successfully imported app using method 3")
        except Exception as e:
            print(f"❌ Method 3 failed: {e}")
            print("All import methods failed!")
            sys.exit(1)

if app is None:
    print("❌ Failed to import Flask app")
    sys.exit(1)

print("🚀 Flask app imported successfully!")

# Configure for production
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print(f"🌐 Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)