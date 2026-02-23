#!/usr/bin/env python3
"""
Test script to verify fma_filter.py functionality
This creates a mock dataset structure for testing without downloading the full FMA dataset.
"""

import os
import sys
import shutil
from pathlib import Path
import pandas as pd
import tempfile

def create_mock_fma_structure():
    """Create a mock FMA dataset structure for testing."""
    print("Creating mock FMA dataset structure for testing...")
    
    # Create directories
    base_dir = Path("./test_fma_data")
    metadata_dir = base_dir / "fma_metadata"
    audio_dir = base_dir / "fma_test"
    
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock tracks.csv with multi-level headers
    # Create sample data
    track_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    genres = ['Ambient', 'Rock', 'Drone', 'Pop', 'Experimental', 
              'Hip-Hop', 'Noise', 'Electronic', 'Ambient', 'Jazz']
    
    # Create multi-level column structure similar to FMA
    data = {
        ('track', 'genre_top'): genres,
        ('track', 'title'): [f'Track {i}' for i in track_ids],
        ('album', 'title'): [f'Album {i}' for i in track_ids],
        ('artist', 'name'): [f'Artist {i}' for i in track_ids],
    }
    
    df = pd.DataFrame(data, index=track_ids)
    
    # Save to CSV
    tracks_file = metadata_dir / "tracks.csv"
    df.to_csv(tracks_file)
    
    print(f"✓ Created mock metadata: {tracks_file}")
    print(f"  Tracks: {len(track_ids)}")
    print(f"  Genres: {set(genres)}")
    
    # Create mock audio files
    for track_id in track_ids:
        track_str = str(track_id).zfill(6)
        subdir = track_str[:3]
        audio_subdir = audio_dir / subdir
        audio_subdir.mkdir(parents=True, exist_ok=True)
        
        # Create empty mp3 file (placeholder)
        audio_file = audio_subdir / f"{track_str}.mp3"
        audio_file.write_text(f"Mock audio data for track {track_id}")
    
    print(f"✓ Created {len(track_ids)} mock audio files in: {audio_dir}")
    
    return base_dir, audio_dir, metadata_dir

def test_filter_script():
    """Test the FMA filter script with mock data."""
    print("\n" + "="*70)
    print("TESTING FMA FILTER SCRIPT")
    print("="*70 + "\n")
    
    try:
        # Create mock data
        base_dir, audio_dir, metadata_dir = create_mock_fma_structure()
        
        # Import the filter class
        sys.path.insert(0, str(Path.cwd()))
        from fma_filter import FMAFilter
        
        # Initialize filter
        fma_filter = FMAFilter(data_dir=str(base_dir), output_dir="./test_output")
        
        # Test 1: Load metadata
        print("\nTest 1: Loading metadata...")
        tracks = fma_filter.load_metadata()
        print(f"✓ Successfully loaded {len(tracks)} tracks")
        
        # Test 2: Filter by genre
        print("\nTest 2: Filtering by weak beat genres...")
        weak_genres = ['Ambient', 'Drone', 'Experimental', 'Noise']
        filtered = fma_filter.filter_by_genre(tracks, weak_genres)
        print(f"✓ Filtered to {len(filtered)} tracks")
        
        # Test 3: Copy files
        print("\nTest 3: Copying filtered files...")
        copy_stats = fma_filter.copy_filtered_tracks(filtered, str(audio_dir))
        print(f"✓ Copy stats: {copy_stats}")
        
        # Test 4: Generate statistics
        print("\nTest 4: Generating statistics...")
        stats = fma_filter.generate_statistics(tracks, filtered, copy_stats)
        
        # Verify output
        output_files = list(Path("./test_output").glob("*.mp3"))
        print(f"\n✓ Output files created: {len(output_files)}")
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        
        # Cleanup
        print("\nCleaning up test files...")
        shutil.rmtree(base_dir)
        shutil.rmtree("./test_output")
        print("✓ Cleanup complete")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on failure
        try:
            if Path(base_dir).exists():
                shutil.rmtree(base_dir)
            if Path("./test_output").exists():
                shutil.rmtree("./test_output")
        except:
            pass
        
        return False

def test_local_folder():
    """Test local folder scanning functionality."""
    print("\n" + "="*70)
    print("TESTING LOCAL FOLDER SCANNING")
    print("="*70 + "\n")
    
    try:
        # Create mock local music folder
        test_folder = Path("./test_local_music")
        test_folder.mkdir(exist_ok=True)
        
        # Create mock music files with different names
        test_files = [
            "ambient_soundscape_01.mp3",
            "rock_song_01.mp3",
            "drone_meditation.mp3",
            "pop_hit.wav",
            "experimental_noise.flac",
            "jazz_standard.mp3",
            "ambient_chill.mp3"
        ]
        
        for filename in test_files:
            (test_folder / filename).write_text(f"Mock audio: {filename}")
        
        print(f"✓ Created {len(test_files)} mock music files")
        
        # Import the filter class
        sys.path.insert(0, str(Path.cwd()))
        from fma_filter import FMAFilter
        
        fma_filter = FMAFilter(output_dir="./test_local_output")
        
        # Test scanning
        print("\nTest 1: Scanning local folder...")
        inventory = fma_filter.scan_local_folder(str(test_folder), "test_inventory.json")
        print(f"✓ Found {inventory['total_files']} audio files")
        
        # Test filtering
        print("\nTest 2: Filtering by keywords...")
        keywords = ['ambient', 'drone', 'experimental']
        stats = fma_filter.filter_local_folder(str(test_folder), keywords, "./test_local_output")
        
        print(f"\n✓ Filtering complete:")
        print(f"  Total scanned: {stats['total_scanned']}")
        print(f"  Matched: {stats['matched']}")
        print(f"  Copied: {stats['copied']}")
        
        print("\n" + "="*70)
        print("LOCAL FOLDER TESTS PASSED! ✓")
        print("="*70)
        
        # Cleanup
        print("\nCleaning up test files...")
        shutil.rmtree(test_folder)
        shutil.rmtree("./test_local_output")
        if Path("test_inventory.json").exists():
            Path("test_inventory.json").unlink()
        print("✓ Cleanup complete")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on failure
        try:
            if Path(test_folder).exists():
                shutil.rmtree(test_folder)
            if Path("./test_local_output").exists():
                shutil.rmtree("./test_local_output")
            if Path("test_inventory.json").exists():
                Path("test_inventory.json").unlink()
        except:
            pass
        
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              FMA Filter Script - Test Suite                          ║
╚══════════════════════════════════════════════════════════════════════╝

This script will test the fma_filter.py functionality without requiring
the full FMA dataset download.
    """)
    
    # Check if pandas is installed
    try:
        import pandas
        print("✓ pandas is installed")
    except ImportError:
        print("✗ pandas is not installed")
        print("\nPlease install dependencies:")
        print("  pip install pandas")
        sys.exit(1)
    
    # Run tests
    test1_passed = test_filter_script()
    test2_passed = test_local_folder()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"FMA Filter Test: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Local Folder Test: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print("="*70 + "\n")
    
    if test1_passed and test2_passed:
        print("All tests passed! The script is ready to use.")
        print("\nNext steps:")
        print("1. Download the real FMA dataset (use: python fma_filter.py --download-info)")
        print("2. Run the filter on real data (use: python fma_filter.py --filter --audio-dir ...)")
        return 0
    else:
        print("Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTests cancelled by user.")
        sys.exit(1)
