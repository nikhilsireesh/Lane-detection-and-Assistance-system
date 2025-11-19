#!/usr/bin/env python3
"""
Entry point for Render deployment
This file imports and runs the main Flask application
"""

import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the Flask app from the actual location
try:
    from phase7_real_time_processing.scripts.web_demo import app
    
    # Configure for production
    if __name__ == "__main__":
        port = int(os.environ.get("PORT", 5001))
        app.run(host="0.0.0.0", port=port, debug=False)
    
except ImportError as e:
    print(f"Error importing Flask app: {e}")
    print("Make sure all dependencies are installed")
    sys.exit(1)