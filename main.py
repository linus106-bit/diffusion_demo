#!/usr/bin/env python3
"""
DiffuChatGPT - Main Entry Point
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from diffusion_demo.app import main

if __name__ == "__main__":
    main()
