#!/usr/bin/env python3
"""
Example usage script for fma_filter.py
This demonstrates various ways to use the FMA filter script.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and display the result."""
    print("\n" + "="*70)
    print(f"EXAMPLE: {description}")
    print("="*70)
    print(f"Command: {cmd}\n")
    
    response = input("Run this command? (y/n): ")
    if response.lower() == 'y':
        subprocess.run(cmd, shell=True)
    else:
        print("Skipped.\n")

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              FMA Weak Beat Music Filter - Examples                   ║
╚══════════════════════════════════════════════════════════════════════╝

This script will walk you through example usages of fma_filter.py
    """)
    
    # Example 1: Show download info
    run_command(
        "python fma_filter.py --download-info --dataset-size medium",
        "Show download commands for FMA medium dataset"
    )
    
    # Example 2: Filter FMA dataset
    fma_data_exists = Path("./fma_data/fma_medium").exists()
    
    if fma_data_exists:
        print("\n✓ FMA medium dataset detected!")
        run_command(
            "python fma_filter.py --filter --audio-dir ./fma_data/fma_medium --output-dir ./weak_beat_music",
            "Filter FMA dataset for weak beat genres"
        )
        
        run_command(
            "python fma_filter.py --filter --audio-dir ./fma_data/fma_medium --genres Ambient Drone --output-dir ./ambient_only",
            "Filter for Ambient and Drone only"
        )
    else:
        print("\n⚠ FMA medium dataset not found at ./fma_data/fma_medium")
        print("Download it first using the commands from Example 1")
    
    # Example 3: Scan local folder
    print("\n" + "="*70)
    print("EXAMPLE: Scan local music folder")
    print("="*70)
    local_path = input("Enter path to your local music folder (or press Enter to skip): ")
    
    if local_path.strip():
        run_command(
            f'python fma_filter.py --scan-local "{local_path}"',
            "Scan local music folder and create inventory"
        )
        
        keywords = input("\nEnter keywords to filter by (space-separated, e.g., 'ambient drone chill'): ")
        if keywords.strip():
            run_command(
                f'python fma_filter.py --filter-local "{local_path}" --keywords {keywords} --output-dir ./local_filtered',
                f"Filter local folder by keywords: {keywords}"
            )
    
    # Example 4: Advanced usage
    print("\n" + "="*70)
    print("ADVANCED EXAMPLES")
    print("="*70)
    print("""
Other useful commands:

1. Save statistics to JSON:
   python fma_filter.py --filter --audio-dir ./fma_data/fma_medium --save-stats stats.json

2. Preserve directory structure:
   python fma_filter.py --filter --audio-dir ./fma_data/fma_medium --preserve-structure

3. Custom output directory:
   python fma_filter.py --filter --audio-dir ./fma_data/fma_medium --output-dir ./my_music

4. Multiple genres:
   python fma_filter.py --filter --audio-dir ./fma_data/fma_medium --genres Ambient Drone Experimental Noise "New Age"

5. Local folder with custom output:
   python fma_filter.py --filter-local ~/Music --keywords ambient meditation --output-dir ./relaxing_music
    """)
    
    print("\n" + "="*70)
    print("For more information, see README.md or run:")
    print("  python fma_filter.py --help")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExamples cancelled by user.")
        sys.exit(0)
