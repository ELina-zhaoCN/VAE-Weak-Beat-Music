#!/usr/bin/env bash
# =============================================================================
# BF-VAE → Rhythm Game  End-to-End One-Click Script
# =============================================================================
# Usage:
#   bash run_e2e_demo.sh                          # use default output_enhanced.wav
#   bash run_e2e_demo.sh /path/to/your_audio.wav  # custom VAE output
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GAME_DIR="$(dirname "$SCRIPT_DIR")/oops-i-tapped-it-again"
VENV="$SCRIPT_DIR/.venv"
VAE_MODEL="$SCRIPT_DIR/7.BF_VAE_v2/checkpoints/best_model_v2.pth"
VAE_OUTPUT="${1:-$SCRIPT_DIR/output_enhanced.wav}"   # <Your VAE Output Path>
INPUT_MUSIC="${2:-$SCRIPT_DIR/weak_beat_music/101765.mp3}"

echo "================================================================"
echo " BF-VAE → Rhythm Game  End-to-End Demo"
echo "================================================================"

# ── activate venv ─────────────────────────────────────────────────────────────
source "$VENV/bin/activate"
echo "[1/4] Virtual env activated"

# ── Step A: VAE beat enhancement ──────────────────────────────────────────────
echo ""
echo "[2/4] Running VAE beat enhancement..."
python -u "$SCRIPT_DIR/7.BF_VAE_v2/inference_v2.py" \
    --input      "$INPUT_MUSIC" \
    --checkpoint "$VAE_MODEL" \
    --output     "$VAE_OUTPUT" \
    --plot       "$SCRIPT_DIR/output_comparison.png" \
    2>/dev/null
echo "      ✅ Enhanced audio saved → $VAE_OUTPUT"

# ── Step B: Generate beatmap from VAE output ──────────────────────────────────
echo ""
echo "[3/4] Generating beatmap from enhanced audio..."
python "$GAME_DIR/tools/generate_beatmap.py" \
    "$VAE_OUTPUT" \
    --typescript \
    --difficulty medium \
    --name "VAE Enhanced Beat" \
    --lane-mode hybrid \
    --spacing 1.0 \
    2>/dev/null

TS_PATH="$GAME_DIR/lens-studio/MusicMaster/Assets/Scripts/SongLibrary.ts"
NOTE_COUNT=$(python3 -c "import json; d=json.load(open('$GAME_DIR/tools/output.json')); print(len(d['notes']))" 2>/dev/null)
echo "      ✅ SongLibrary.ts written with $NOTE_COUNT notes"

# ── Step C: Verify output ─────────────────────────────────────────────────────
echo ""
echo "[4/4] Verification"
echo "  output_enhanced.wav : $(du -sh "$VAE_OUTPUT" | cut -f1)"
echo "  output.json         : $(du -sh "$GAME_DIR/tools/output.json" | cut -f1)"
echo "  SongLibrary.ts      : $(wc -l < "$TS_PATH") lines"

BPM=$(python3 -c "import json; d=json.load(open('$GAME_DIR/tools/output.json')); print(d['bpm'])" 2>/dev/null)
echo ""
echo "================================================================"
echo " ✅ Pipeline complete"
echo "    Song BPM  : $BPM"
echo "    Notes     : $NOTE_COUNT"
echo "    SongLibrary.ts → ready for Lens Studio"
echo "================================================================"
echo ""
echo "Open in Lens Studio:"
echo "  $GAME_DIR/lens-studio/MusicMaster/MusicMaster.esproj"
echo ""
echo "Listen to enhanced audio:"
echo "  open \"$VAE_OUTPUT\""

# ── Exception repair commands (pre-configured) ─────────────────────────────────
cat << 'EXCEPTIONS'

================================================================
 EXCEPTION REPAIR COMMANDS
================================================================
# Fix 1 — librosa audio read failure (corrupt WAV):
#   ffmpeg -i output_enhanced.wav -ar 22050 -ac 1 output_enhanced_fixed.wav
#   bash run_e2e_demo.sh output_enhanced_fixed.wav

# Fix 2 — TS data format incompatibility (NaN/Infinity in notes):
#   python3 -c "
#     import json; f='oops-i-tapped-it-again/tools/output.json'
#     d=json.load(open(f)); d['notes']=[n for n in d['notes']
#     if all(isinstance(v,(int,float)) and abs(v)<1e9
#            for v in (n['beat'],n['lane']))]
#     json.dump(d,open(f,'w'),indent=2); print(len(d['notes']),'notes remain')"
#   python oops-i-tapped-it-again/tools/generate_beatmap.py \
#     oops-i-tapped-it-again/tools/output.json --typescript

# Fix 3 — game data loading error (AllSongs not exported):
#   grep -n "AllSongs" \
#     oops-i-tapped-it-again/lens-studio/MusicMaster/Assets/Scripts/SongLibrary.ts
#   # If missing, append manually:
#   echo 'export const AllSongs = [Song_VaeEnhancedBeat];' >> \
#     oops-i-tapped-it-again/lens-studio/MusicMaster/Assets/Scripts/SongLibrary.ts
================================================================
EXCEPTIONS
