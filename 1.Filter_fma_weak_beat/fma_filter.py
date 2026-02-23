#!/usr/bin/env python3
"""
FMA Dataset Download and Filtering Script
==========================================
This script downloads the FMA (Free Music Archive) dataset and filters music
with weak beats (Ambient, Drone, Experimental, Noise, etc.).

Features:
1. Download FMA metadata and audio files
2. Filter music by genre (weak beat genres)
3. Copy filtered files to output directory
4. Generate statistics and genre distribution
5. Support for scanning local music folders
"""

import os
import sys
import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Set, Tuple
import pandas as pd
from collections import Counter
import json

# Default weak beat genres
DEFAULT_WEAK_BEAT_GENRES = [
    'Ambient',
    'Drone',
    'Experimental',
    'Noise',
    'Easy Listening',
    'Sound Effects',
    'Field Recordings',
    'Spoken Word',
    'New Age'
]

# FMA Dataset URLs
FMA_METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
FMA_SMALL_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
FMA_MEDIUM_URL = "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
FMA_LARGE_URL = "https://os.unil.cloud.switch.ch/fma/fma_large.zip"
FMA_FULL_URL = "https://os.unil.cloud.switch.ch/fma/fma_full.zip"


class FMAFilter:
    """Main class for FMA dataset filtering operations."""
    
    def __init__(self, data_dir: str = "./fma_data", output_dir: str = "./weak_beat_music"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.metadata_dir = self.data_dir / "fma_metadata"
        self.tracks_file = self.metadata_dir / "tracks.csv"
        
    def print_download_commands(self, dataset_size: str = "medium"):
        """Print download commands for FMA dataset."""
        print("\n" + "="*70)
        print("FMA DATASET DOWNLOAD COMMANDS")
        print("="*70)
        
        # Metadata download
        print("\n1. Download FMA Metadata (342 MB):")
        print(f"   wget {FMA_METADATA_URL}")
        print("   OR")
        print(f"   curl -O {FMA_METADATA_URL}")
        
        # Audio download based on size
        audio_urls = {
            'small': (FMA_SMALL_URL, "7.2 GB, 8,000 tracks of 30s, 8 balanced genres"),
            'medium': (FMA_MEDIUM_URL, "25 GB, 25,000 tracks of 30s, 16 unbalanced genres"),
            'large': (FMA_LARGE_URL, "93 GB, 106,574 tracks of 30s, 161 unbalanced genres"),
            'full': (FMA_FULL_URL, "879 GB, 106,574 untrimmed tracks, 161 unbalanced genres")
        }
        
        url, description = audio_urls.get(dataset_size, audio_urls['medium'])
        
        print(f"\n2. Download FMA {dataset_size.upper()} Audio ({description}):")
        print(f"   wget {url}")
        print("   OR")
        print(f"   curl -O {url}")
        
        print("\n3. Extract the downloaded files:")
        print(f"   unzip fma_metadata.zip -d {self.data_dir}")
        print(f"   unzip fma_{dataset_size}.zip -d {self.data_dir}")
        
        print("\n" + "="*70)
        print("\nAfter downloading, run this script again with --filter option.")
        print("="*70 + "\n")
        
    def check_dataset_exists(self) -> bool:
        """Check if FMA dataset files exist."""
        return self.tracks_file.exists()
    
    def load_metadata(self) -> pd.DataFrame:
        """Load and parse FMA tracks metadata."""
        print(f"\nLoading metadata from: {self.tracks_file}")
        
        if not self.tracks_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.tracks_file}\n"
                f"Please download the FMA metadata first."
            )
        
        # FMA tracks.csv has a multi-level header
        tracks = pd.read_csv(self.tracks_file, index_col=0, header=[0, 1])
        
        print(f"Loaded {len(tracks)} tracks from metadata")
        return tracks
    
    def filter_by_genre(self, tracks: pd.DataFrame, genres: List[str]) -> pd.DataFrame:
        """Filter tracks by specified genres."""
        print(f"\nFiltering tracks by genres: {', '.join(genres)}")
        
        # FMA metadata has multi-level columns, genre_top is under 'track' section
        try:
            # Try to access genre_top column
            if ('track', 'genre_top') in tracks.columns:
                genre_col = ('track', 'genre_top')
            elif 'genre_top' in tracks.columns:
                genre_col = 'genre_top'
            else:
                # Find the correct column name
                genre_cols = [col for col in tracks.columns if 'genre' in str(col).lower()]
                print(f"Available genre columns: {genre_cols}")
                raise KeyError("Could not find genre_top column in metadata")
            
            # Filter tracks
            filtered = tracks[tracks[genre_col].isin(genres)]
            print(f"Found {len(filtered)} tracks matching the specified genres")
            
            return filtered
        except Exception as e:
            print(f"Error filtering by genre: {e}")
            print(f"Available columns: {tracks.columns.tolist()[:10]}...")
            raise
    
    def get_audio_path(self, track_id: int, audio_dir: Path) -> Path:
        """Get the path to an audio file given its track ID."""
        # FMA uses a hierarchical directory structure: fma_medium/123/012345.mp3
        track_str = str(track_id).zfill(6)
        subdir = track_str[:3]
        filename = f"{track_str}.mp3"
        return audio_dir / subdir / filename
    
    def copy_filtered_tracks(self, filtered_tracks: pd.DataFrame, 
                            audio_dir: str, 
                            preserve_structure: bool = False) -> Dict:
        """Copy filtered audio files to output directory."""
        audio_path = Path(audio_dir)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio directory not found: {audio_path}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCopying filtered tracks to: {self.output_dir}")
        
        stats = {
            'total': len(filtered_tracks),
            'copied': 0,
            'missing': 0,
            'failed': 0,
            'missing_files': []
        }
        
        for track_id in filtered_tracks.index:
            src_path = self.get_audio_path(track_id, audio_path)
            
            if preserve_structure:
                # Preserve directory structure
                track_str = str(track_id).zfill(6)
                subdir = track_str[:3]
                dest_dir = self.output_dir / subdir
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / f"{track_str}.mp3"
            else:
                # Flat structure
                dest_path = self.output_dir / f"{track_id}.mp3"
            
            if src_path.exists():
                try:
                    shutil.copy2(src_path, dest_path)
                    stats['copied'] += 1
                    if stats['copied'] % 100 == 0:
                        print(f"  Copied {stats['copied']}/{stats['total']} files...")
                except Exception as e:
                    print(f"  Error copying {src_path}: {e}")
                    stats['failed'] += 1
            else:
                stats['missing'] += 1
                stats['missing_files'].append(str(src_path))
        
        print(f"\nCopy complete!")
        print(f"  Successfully copied: {stats['copied']}")
        print(f"  Missing files: {stats['missing']}")
        print(f"  Failed to copy: {stats['failed']}")
        
        return stats
    
    def generate_statistics(self, tracks: pd.DataFrame, 
                          filtered_tracks: pd.DataFrame,
                          copy_stats: Dict = None) -> Dict:
        """Generate and display filtering statistics."""
        print("\n" + "="*70)
        print("FILTERING STATISTICS")
        print("="*70)
        
        # Find genre column
        if ('track', 'genre_top') in tracks.columns:
            genre_col = ('track', 'genre_top')
        elif 'genre_top' in tracks.columns:
            genre_col = 'genre_top'
        else:
            genre_col = None
        
        stats = {
            'total_tracks': len(tracks),
            'filtered_tracks': len(filtered_tracks),
            'filter_percentage': (len(filtered_tracks) / len(tracks) * 100) if len(tracks) > 0 else 0,
        }
        
        print(f"\nTotal tracks in dataset: {stats['total_tracks']:,}")
        print(f"Filtered tracks (weak beat): {stats['filtered_tracks']:,}")
        print(f"Percentage: {stats['filter_percentage']:.2f}%")
        
        if genre_col and len(filtered_tracks) > 0:
            print("\nGenre Distribution in Filtered Tracks:")
            genre_counts = filtered_tracks[genre_col].value_counts()
            stats['genre_distribution'] = genre_counts.to_dict()
            
            for genre, count in genre_counts.items():
                percentage = (count / len(filtered_tracks) * 100)
                print(f"  {genre}: {count:,} tracks ({percentage:.2f}%)")
        
        if copy_stats:
            print("\nCopy Statistics:")
            print(f"  Successfully copied: {copy_stats['copied']:,}")
            print(f"  Missing files: {copy_stats['missing']:,}")
            print(f"  Failed to copy: {copy_stats['failed']:,}")
            stats['copy_stats'] = copy_stats
        
        print("="*70 + "\n")
        
        return stats
    
    def scan_local_folder(self, folder_path: str, 
                         output_file: str = "local_music_inventory.json") -> Dict:
        """Scan a local music folder and create an inventory."""
        folder = Path(folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        
        print(f"\nScanning local folder: {folder}")
        
        # Supported audio extensions
        audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma'}
        
        music_files = []
        for ext in audio_extensions:
            music_files.extend(folder.rglob(f"*{ext}"))
        
        print(f"Found {len(music_files)} audio files")
        
        inventory = {
            'folder': str(folder),
            'total_files': len(music_files),
            'files': []
        }
        
        for audio_file in music_files:
            file_info = {
                'path': str(audio_file),
                'filename': audio_file.name,
                'extension': audio_file.suffix,
                'size_mb': audio_file.stat().st_size / (1024 * 1024)
            }
            inventory['files'].append(file_info)
        
        # Save inventory
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(inventory, f, indent=2)
        
        print(f"Inventory saved to: {output_path}")
        
        return inventory
    
    def filter_local_folder(self, folder_path: str, 
                          genre_keywords: List[str],
                          output_dir: str = None) -> Dict:
        """Filter local music folder by filename keywords."""
        folder = Path(folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        
        if output_dir is None:
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nScanning local folder: {folder}")
        print(f"Filtering by keywords: {', '.join(genre_keywords)}")
        
        # Supported audio extensions
        audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma'}
        
        # Find all audio files
        music_files = []
        for ext in audio_extensions:
            music_files.extend(folder.rglob(f"*{ext}"))
        
        print(f"Total audio files found: {len(music_files)}")
        
        # Filter by keywords in filename (case-insensitive)
        filtered_files = []
        for audio_file in music_files:
            filename_lower = audio_file.name.lower()
            if any(keyword.lower() in filename_lower for keyword in genre_keywords):
                filtered_files.append(audio_file)
        
        print(f"Files matching keywords: {len(filtered_files)}")
        
        # Copy filtered files
        stats = {
            'total_scanned': len(music_files),
            'matched': len(filtered_files),
            'copied': 0,
            'failed': 0
        }
        
        print(f"\nCopying matched files to: {output_dir}")
        
        for src_file in filtered_files:
            dest_file = output_dir / src_file.name
            
            # Handle duplicate filenames
            counter = 1
            while dest_file.exists():
                stem = src_file.stem
                dest_file = output_dir / f"{stem}_{counter}{src_file.suffix}"
                counter += 1
            
            try:
                shutil.copy2(src_file, dest_file)
                stats['copied'] += 1
                if stats['copied'] % 10 == 0:
                    print(f"  Copied {stats['copied']}/{len(filtered_files)} files...")
            except Exception as e:
                print(f"  Error copying {src_file}: {e}")
                stats['failed'] += 1
        
        print("\n" + "="*70)
        print("LOCAL FOLDER FILTERING STATISTICS")
        print("="*70)
        print(f"Total files scanned: {stats['total_scanned']:,}")
        print(f"Files matched: {stats['matched']:,}")
        print(f"Successfully copied: {stats['copied']:,}")
        print(f"Failed to copy: {stats['failed']:,}")
        print("="*70 + "\n")
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download and filter FMA dataset for weak beat music",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show download commands
  python fma_filter.py --download-info
  
  # Filter FMA dataset
  python fma_filter.py --filter --audio-dir ./fma_data/fma_medium
  
  # Filter with custom genres
  python fma_filter.py --filter --audio-dir ./fma_data/fma_medium --genres Ambient Drone Noise
  
  # Scan local folder
  python fma_filter.py --scan-local /path/to/music
  
  # Filter local folder by keywords
  python fma_filter.py --filter-local /path/to/music --keywords ambient drone chill
        """
    )
    
    # Main action arguments
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument('--download-info', action='store_true',
                            help='Show download commands for FMA dataset')
    action_group.add_argument('--filter', action='store_true',
                            help='Filter FMA dataset by weak beat genres')
    action_group.add_argument('--scan-local', type=str, metavar='FOLDER',
                            help='Scan local music folder and create inventory')
    action_group.add_argument('--filter-local', type=str, metavar='FOLDER',
                            help='Filter local music folder by filename keywords')
    
    # Configuration arguments
    parser.add_argument('--data-dir', default='./fma_data',
                       help='Directory for FMA data (default: ./fma_data)')
    parser.add_argument('--output-dir', default='./weak_beat_music',
                       help='Output directory for filtered music (default: ./weak_beat_music)')
    parser.add_argument('--audio-dir', type=str,
                       help='Directory containing FMA audio files (e.g., ./fma_data/fma_medium)')
    parser.add_argument('--genres', nargs='+', default=DEFAULT_WEAK_BEAT_GENRES,
                       help='List of weak beat genres to filter')
    parser.add_argument('--keywords', nargs='+',
                       help='Keywords for filtering local folder (case-insensitive)')
    parser.add_argument('--dataset-size', choices=['small', 'medium', 'large', 'full'],
                       default='medium',
                       help='FMA dataset size for download info (default: medium)')
    parser.add_argument('--preserve-structure', action='store_true',
                       help='Preserve directory structure when copying files')
    parser.add_argument('--save-stats', type=str,
                       help='Save statistics to JSON file')
    
    args = parser.parse_args()
    
    # Initialize filter
    fma_filter = FMAFilter(data_dir=args.data_dir, output_dir=args.output_dir)
    
    try:
        if args.download_info:
            # Show download commands
            fma_filter.print_download_commands(args.dataset_size)
            
        elif args.filter:
            # Filter FMA dataset
            if not args.audio_dir:
                print("Error: --audio-dir is required for filtering")
                print("Example: --audio-dir ./fma_data/fma_medium")
                sys.exit(1)
            
            # Check if dataset exists
            if not fma_filter.check_dataset_exists():
                print("\nError: FMA metadata not found!")
                print("Please download the dataset first:\n")
                fma_filter.print_download_commands(args.dataset_size)
                sys.exit(1)
            
            # Load metadata
            tracks = fma_filter.load_metadata()
            
            # Filter by genre
            filtered_tracks = fma_filter.filter_by_genre(tracks, args.genres)
            
            if len(filtered_tracks) == 0:
                print("\nWarning: No tracks found matching the specified genres.")
                print("Please check your genre list or metadata file.")
                sys.exit(0)
            
            # Copy files
            copy_stats = fma_filter.copy_filtered_tracks(
                filtered_tracks, 
                args.audio_dir,
                args.preserve_structure
            )
            
            # Generate statistics
            stats = fma_filter.generate_statistics(tracks, filtered_tracks, copy_stats)
            
            # Save statistics if requested
            if args.save_stats:
                with open(args.save_stats, 'w') as f:
                    json.dump(stats, f, indent=2)
                print(f"Statistics saved to: {args.save_stats}")
        
        elif args.scan_local:
            # Scan local folder
            inventory = fma_filter.scan_local_folder(args.scan_local)
            print(f"\nFound {inventory['total_files']} audio files")
            print("Inventory saved to: local_music_inventory.json")
        
        elif args.filter_local:
            # Filter local folder
            if not args.keywords:
                print("Error: --keywords is required for filtering local folder")
                print("Example: --keywords ambient drone experimental")
                sys.exit(1)
            
            stats = fma_filter.filter_local_folder(
                args.filter_local,
                args.keywords,
                args.output_dir
            )
            
            # Save statistics if requested
            if args.save_stats:
                with open(args.save_stats, 'w') as f:
                    json.dump(stats, f, indent=2)
                print(f"Statistics saved to: {args.save_stats}")
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
