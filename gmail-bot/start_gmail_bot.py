#!/usr/bin/env python3
"""
Gmail Bot Launcher
This script changes to the correct directory and starts the Gmail bot.
Use this file to bypass CMD restrictions.
"""

import os
import sys
import subprocess

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the bot directory
    os.chdir(script_dir)
    
    # Path to virtual environment Python
    venv_python = os.path.join(script_dir, '.venv', 'Scripts', 'python.exe')
    bot_script = os.path.join(script_dir, 'gm.py')
    
    print("=" * 50)
    print("Gmail Auto-Reply Bot Launcher")
    print("=" * 50)
    print(f"Working Directory: {script_dir}")
    print(f"Python Path: {venv_python}")
    print(f"Bot Script: {bot_script}")
    print("=" * 50)
    
    # Check if virtual environment exists
    if not os.path.exists(venv_python):
        print("❌ Virtual environment not found!")
        print(f"Expected: {venv_python}")
        print("Please set up the virtual environment first.")
        input("Press Enter to exit...")
        return
    
    # Check if bot script exists
    if not os.path.exists(bot_script):
        print("❌ Bot script not found!")
        print(f"Expected: {bot_script}")
        input("Press Enter to exit...")
        return
    
    print("✅ Starting Gmail Bot...")
    print("Press Ctrl+C to stop the bot")
    print("-" * 50)
    
    try:
        # Run the bot using the virtual environment Python
        subprocess.run([venv_python, bot_script], cwd=script_dir)
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
