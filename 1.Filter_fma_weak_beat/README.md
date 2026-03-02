# FMA Dataset Weak Beat Music Filter

A Python script to download and filter the Free Music Archive (FMA) dataset, specifically targeting music with weak beats such as Ambient, Drone, Experimental, Noise, and similar genres.

## Features

- 🎵 **Download FMA Dataset**: Provides commands to download FMA metadata and audio files
- 🔍 **Genre-Based Filtering**: Filter tracks by specified genres (default: weak beat genres)
- 📊 **Statistics & Reports**: Generate detailed statistics and genre distribution
- 📁 **Local Folder Support**: Scan and filter existing music folders without FMA metadata
- 🎯 **Flexible Configuration**: Customizable genres, output directories, and filtering options

## Installation

1. **Clone or download this script**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Make the script executable** (optional):
```bash
chmod +x fma_filter.py
```

## Usage

### 1. Download FMA Dataset

First, get the download commands:

```bash
python fma_filter.py --download-info
```

This will display wget/curl commands for downloading:
- **FMA Metadata** (342 MB) - Required for filtering
- **FMA Medium** (25 GB, 25,000 tracks, 30s clips) - Recommended for most users
- Other sizes: small (7.2 GB), large (93 GB), full (879 GB)

Example download process:

```bash
# Download metadata
wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip

# Download medium audio dataset
wget https://os.unil.cloud.switch.ch/fma/fma_medium.zip

# Extract files
unzip fma_metadata.zip -d ./fma_data
unzip fma_medium.zip -d ./fma_data
```

### 2. Filter FMA Dataset by Genre

Filter the dataset for weak beat genres:

```bash
python fma_filter.py --filter --audio-dir ./fma_data/fma_medium
```

**With custom genres**:

```bash
python fma_filter.py --filter \
  --audio-dir ./fma_data/fma_medium \
  --genres Ambient Drone Experimental Noise "New Age"
```

**Preserve directory structure**:

```bash
python fma_filter.py --filter \
  --audio-dir ./fma_data/fma_medium \
  --preserve-structure
```

**Save statistics to JSON**:

```bash
python fma_filter.py --filter \
  --audio-dir ./fma_data/fma_medium \
  --save-stats filtering_stats.json
```

### 3. Scan Local Music Folder

Create an inventory of audio files in a local folder:

```bash
python fma_filter.py --scan-local /path/to/your/music
```

This creates `local_music_inventory.json` with information about all audio files found.

### 4. Filter Local Music Folder

Filter local music by filename keywords:

```bash
python fma_filter.py --filter-local /path/to/your/music \
  --keywords ambient drone chill atmospheric
```

**With custom output directory**:

```bash
python fma_filter.py --filter-local /path/to/your/music \
  --keywords ambient experimental \
  --output-dir ./my_filtered_music
```

## Command-Line Options

### Actions (choose one):

- `--download-info`: Show download commands for FMA dataset
- `--filter`: Filter FMA dataset by genre
- `--scan-local FOLDER`: Scan local music folder and create inventory
- `--filter-local FOLDER`: Filter local music folder by keywords

### Configuration:

- `--data-dir DIR`: Directory for FMA data (default: `./fma_data`)
- `--output-dir DIR`: Output directory for filtered music (default: `./weak_beat_music`)
- `--audio-dir DIR`: Directory containing FMA audio files (required for `--filter`)
- `--genres GENRE [GENRE ...]`: List of genres to filter (default: weak beat genres)
- `--keywords WORD [WORD ...]`: Keywords for filtering local folders (required for `--filter-local`)
- `--dataset-size {small,medium,large,full}`: Dataset size for download info (default: `medium`)
- `--preserve-structure`: Preserve directory structure when copying files
- `--save-stats FILE`: Save statistics to JSON file

## Default Weak Beat Genres

The script filters for these genres by default:

- Ambient
- Drone
- Experimental
- Noise
- Easy Listening
- Sound Effects
- Field Recordings
- Spoken Word
- New Age

You can override these with the `--genres` option.

## Output

### Filtering FMA Dataset

The script will:
1. Load and parse the FMA metadata (`tracks.csv`)
2. Filter tracks by specified genres
3. Copy matching audio files to the output directory
4. Display statistics including:
   - Total tracks in dataset
   - Number of filtered tracks
   - Genre distribution
   - Copy statistics (success/missing/failed)

### Filtering Local Folder

The script will:
1. Scan the folder for audio files (mp3, wav, flac, m4a, ogg, aac, wma)
2. Filter by filename keywords (case-insensitive)
3. Copy matching files to output directory
4. Display statistics

## Examples

### Complete workflow for FMA dataset:

```bash
# Step 1: Get download commands
python fma_filter.py --download-info --dataset-size medium

# Step 2: Download (using displayed commands)
wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
wget https://os.unil.cloud.switch.ch/fma/fma_medium.zip
unzip fma_metadata.zip -d ./fma_data
unzip fma_medium.zip -d ./fma_data

# Step 3: Filter the dataset
python fma_filter.py --filter \
  --audio-dir ./fma_data/fma_medium \
  --output-dir ./weak_beat_music \
  --save-stats stats.json

# Step 4: Check results
ls -lh ./weak_beat_music/
cat stats.json
```

### Filter existing music collection:

```bash
# Scan your music folder first
python fma_filter.py --scan-local ~/Music

# Filter by keywords
python fma_filter.py --filter-local ~/Music \
  --keywords ambient "new age" meditation drone \
  --output-dir ./weak_beat_collection
```

## FMA Dataset Information

The Free Music Archive dataset includes:

- **Small**: 8,000 tracks, 30-second clips, 8 balanced genres (7.2 GB)
- **Medium**: 25,000 tracks, 30-second clips, 16 unbalanced genres (25 GB)
- **Large**: 106,574 tracks, 30-second clips, 161 unbalanced genres (93 GB)
- **Full**: 106,574 tracks, full length, 161 unbalanced genres (879 GB)

More information: [FMA GitHub Repository](https://github.com/mdeff/fma)

## Supported Audio Formats

The script supports the following audio formats when scanning local folders:

- MP3 (`.mp3`)
- WAV (`.wav`)
- FLAC (`.flac`)
- M4A (`.m4a`)
- OGG (`.ogg`)
- AAC (`.aac`)
- WMA (`.wma`)

## Troubleshooting

### "Metadata file not found"

Make sure you've downloaded and extracted the FMA metadata:
```bash
wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
unzip fma_metadata.zip -d ./fma_data
```

### "Audio directory not found"

Ensure the `--audio-dir` path is correct. After extracting fma_medium.zip, the path should be `./fma_data/fma_medium`.

### "No tracks found matching the specified genres"

Check the available genres in your dataset:
```python
import pandas as pd
tracks = pd.read_csv('./fma_data/fma_metadata/tracks.csv', header=[0, 1])
print(tracks[('track', 'genre_top')].unique())
```

### Missing audio files

Some tracks in the metadata might not have corresponding audio files. The script will report missing files in the statistics.

## License

This script is provided as-is for working with the FMA dataset. Please refer to the [FMA dataset license](https://github.com/mdeff/fma) for information about the dataset usage.

## Contributing

Feel free to submit issues or pull requests for improvements!

## Citation

If you use the FMA dataset in your research, please cite:

```
@inproceedings{fma_dataset,
  title = {{FMA}: A Dataset for Music Analysis},
  author = {Defferrard, Micha\"el and Benzi, Kirell and Vandergheynst, Pierre and Bresson, Xavier},
  booktitle = {18th International Society for Music Information Retrieval Conference (ISMIR)},
  year = {2017},
  archiveprefix = {arXiv},
  eprint = {1612.01840},
  url = {https://arxiv.org/abs/1612.01840},
}
```
