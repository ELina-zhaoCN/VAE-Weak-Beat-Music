#!/usr/bin/env bash
# =============================================================================
# 一键生成模型训练过程所有对比图（用于向老师展示）
# =============================================================================
# 用法: bash generate_all_plots.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
OUT_DIR="$SCRIPT_DIR/demo_plots"
MODEL="$SCRIPT_DIR/7.BF_VAE_v2/checkpoints/best_model_v2.pth"
DATA_DIR="$SCRIPT_DIR/weak_beat_music"
SPLIT_FILE="$SCRIPT_DIR/7.BF_VAE_v2/checkpoints/data_split.json"
HISTORY="$SCRIPT_DIR/7.BF_VAE_v2/checkpoints/history_v2.json"

mkdir -p "$OUT_DIR"
source "$VENV/bin/activate"

echo "================================================================"
echo " 生成模型训练过程对比图"
echo "================================================================"

# ── 1. 训练曲线 (从 history_v2.json) ─────────────────────────────────────
echo ""
echo "[1/4] 训练曲线 (Training Curves)..."
python "$SCRIPT_DIR/7.BF_VAE_v2/plot_train_history.py" \
    --history "$HISTORY" \
    --output  "$OUT_DIR/1_training_curves.png"
echo "      ✅ $OUT_DIR/1_training_curves.png"

# ── 2. 单曲增强对比图 (inference_v2 --plot) ───────────────────────────────
echo ""
echo "[2/4] 单曲增强对比 (Input vs Enhanced)..."
INPUT_MP3="$DATA_DIR/101765.mp3"
if [ ! -f "$INPUT_MP3" ]; then
    INPUT_MP3=$(find "$DATA_DIR" -name "*.mp3" -o -name "*.wav" 2>/dev/null | head -1)
fi
if [ -n "$INPUT_MP3" ] && [ -f "$INPUT_MP3" ]; then
    python "$SCRIPT_DIR/7.BF_VAE_v2/inference_v2.py" \
        --input      "$INPUT_MP3" \
        --checkpoint "$MODEL" \
        --output     "$OUT_DIR/2_enhanced_sample.wav" \
        --plot       "$OUT_DIR/2_inference_comparison.png" 2>/dev/null
    echo "      ✅ $OUT_DIR/2_inference_comparison.png"
else
    echo "      ⚠ 跳过: 未找到 weak_beat_music 中的音频"
fi

# ── 3. 测试集汇总图 (evaluate_v2) ─────────────────────────────────────────
echo ""
echo "[3/4] 测试集汇总 (Test Set Summary)..."
EVAL_TMP="$OUT_DIR/_eval_tmp"
if [ -f "$SPLIT_FILE" ] && [ -d "$DATA_DIR" ]; then
    mkdir -p "$EVAL_TMP"
    python "$SCRIPT_DIR/7.BF_VAE_v2/evaluate_v2.py" \
        --checkpoint "$MODEL" \
        --split_file "$SPLIT_FILE" \
        --data_dir   "$DATA_DIR" \
        --output_dir "$EVAL_TMP" \
        --n_samples  15 2>/dev/null
    if [ -f "$EVAL_TMP/test_summary_v2.png" ]; then
        cp "$EVAL_TMP/test_summary_v2.png" "$OUT_DIR/3_test_summary.png"
        echo "      ✅ $OUT_DIR/3_test_summary.png"
    fi
    rm -rf "$EVAL_TMP" 2>/dev/null || true
else
    echo "      ⚠ 跳过: 缺少 data_split.json 或 weak_beat_music"
fi

# ── 4. 单样本详细对比 (evaluate.py v1 风格，可选) ─────────────────────────
echo ""
echo "[4/4] 单样本详细对比 (Sample Comparison)..."
if [ -f "$SCRIPT_DIR/6.Test_Script/evaluate.py" ] && [ -d "$DATA_DIR" ]; then
    python "$SCRIPT_DIR/6.Test_Script/evaluate.py" \
        --checkpoint  "$MODEL" \
        --data_dir    "$DATA_DIR" \
        --output_dir  "$OUT_DIR/sample_vis" \
        --num_samples 3 \
        --save_audio 2>/dev/null || true
    VIS_DIR="$OUT_DIR/sample_vis/visualizations"
    if [ -d "$VIS_DIR" ]; then
        for f in "$VIS_DIR"/*_comparison.png; do
            [ -f "$f" ] && echo "      ✅ $f"
        done
    fi
else
    echo "      ⚠ 跳过: evaluate.py 或数据不可用"
fi

echo ""
echo "================================================================"
echo " 完成！所有对比图已保存到: $OUT_DIR"
echo "================================================================"
echo ""
echo " 1_training_curves.png   - 训练过程曲线 (Regularity, Recon, Beat, KL)"
echo " 2_inference_comparison.png - 单曲增强前后对比 (波形+Mel+节拍)"
echo " 3_test_summary.png     - 测试集汇总 (Before vs After, Δ分布)"
echo " sample_vis/            - 单样本详细对比 (波形+Mel+自相关)"
echo ""
echo " 展示顺序建议: 1 → 2 → 3 → sample_vis"
echo ""
