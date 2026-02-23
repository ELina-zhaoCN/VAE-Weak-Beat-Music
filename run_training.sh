#!/bin/bash
# Complete Training Pipeline for Music VAE
# One-click execution script

set -e  # Exit on error

echo "========================================================================"
echo "Music VAE Training Pipeline"
echo "Learning to Generate Rhythmic Beats from Weak-Beat Music"
echo "========================================================================"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Step 1: Install dependencies
echo -e "\n${BLUE}Step 1: Installing dependencies...${NC}"
pip install -r requirements.txt

# Step 2: Check for data
echo -e "\n${BLUE}Step 2: Checking for training data...${NC}"
if [ ! -d "./weak_beat_music" ] || [ -z "$(ls -A ./weak_beat_music 2>/dev/null)" ]; then
    echo "Training data not found. You need to:"
    echo "  1. Download FMA dataset:"
    echo "     cd 1.filter_fma_weak_beat"
    echo "     python fma_filter.py --download-info"
    echo "  2. Filter weak-beat music:"
    echo "     python fma_filter.py --filter --audio-dir ../fma_data/fma_medium"
    echo ""
    echo "Or use your own music folder:"
    echo "     python fma_filter.py --filter-local /path/to/music --keywords ambient drone"
    echo ""
    read -p "Press Enter to continue anyway, or Ctrl+C to exit..."
else
    NUM_FILES=$(find ./weak_beat_music -type f \( -name "*.mp3" -o -name "*.wav" \) | wc -l)
    echo -e "${GREEN}✓ Found $NUM_FILES audio files in ./weak_beat_music/${NC}"
fi

# Step 3: Train model
echo -e "\n${BLUE}Step 3: Training VAE model...${NC}"
echo "Configuration:"
echo "  - Epochs: 100"
echo "  - Batch size: 16"
echo "  - Learning rate: 1e-4"
echo "  - Latent dimension: 128"
echo "  - KL weight: 1.0 (after warmup)"
echo "  - Beat weight: 0.5 (after warmup)"
echo "  - Warmup: 20 epochs"
echo ""

python train.py \
  --data_dir ./weak_beat_music \
  --epochs 100 \
  --batch_size 16 \
  --lr 1e-4 \
  --latent_dim 128 \
  --kl_weight 1.0 \
  --beat_weight 0.5 \
  --warmup_epochs 20 \
  --checkpoint_dir ./checkpoints \
  --log_dir ./logs \
  --save_interval 10

# Step 4: Evaluate
echo -e "\n${BLUE}Step 4: Evaluating trained model...${NC}"
if [ -f "./checkpoints/best_model.pt" ]; then
    python evaluate.py \
      --checkpoint ./checkpoints/best_model.pt \
      --data_dir ./weak_beat_music \
      --num_samples 5 \
      --output_dir ./evaluation
    
    echo -e "\n${GREEN}✓ Evaluation complete!${NC}"
    echo "Results saved in: ./evaluation/"
    echo ""
    echo "View results:"
    echo "  - Visualizations: ./evaluation/visualizations/"
    echo "  - Metrics: ./evaluation/evaluation_results.json"
    echo ""
    
    if command -v tensorboard &> /dev/null; then
        echo "To view training curves:"
        echo "  tensorboard --logdir=./logs"
    fi
else
    echo "Warning: No trained model found. Training may have failed."
fi

echo ""
echo "========================================================================"
echo -e "${GREEN}Training pipeline complete!${NC}"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Check training curves: tensorboard --logdir=./logs"
echo "  2. View evaluation results: open ./evaluation/visualizations/"
echo "  3. Generate more samples: python evaluate.py --checkpoint ./checkpoints/best_model.pt"
echo ""
