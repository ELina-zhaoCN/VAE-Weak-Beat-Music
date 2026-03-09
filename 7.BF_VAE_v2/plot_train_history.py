#!/usr/bin/env python3
"""
BF-VAE v2  ·  训练曲线可视化
================================
从 history_v2.json 读取训练历史，生成训练过程对比图。

Usage:
    python plot_train_history.py --history 7.BF_VAE_v2/checkpoints/history_v2.json --output training_curves.png
"""

import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_training_curves(history_path: str, output_path: str):
    with open(history_path) as f:
        history = json.load(f)

    epochs = [h['epoch'] for h in history]
    train_reg = [h['train']['regularity'] for h in history]
    val_reg   = [h['val']['regularity'] for h in history]
    train_recon = [h['train']['recon'] for h in history]
    val_recon   = [h['val']['recon'] for h in history]
    train_beat = [h['train']['beat'] for h in history]
    val_beat   = [h['val']['beat'] for h in history]
    kl_raw     = [h['val']['kl_raw'] for h in history]
    beat_w     = [h['beat_w'] for h in history]
    lr         = [h['lr'] for h in history]

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Beat Regularity (主要指标)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(epochs, train_reg, 'b-', linewidth=2, label='Train Regularity')
    ax.plot(epochs, val_reg,   'r-', linewidth=2, label='Val Regularity')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Beat Regularity')
    ax.set_title('Beat Regularity (↑ higher = stronger beats)', fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max(epochs))

    # 2. Reconstruction Loss
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(epochs, train_recon, 'b-', linewidth=2, label='Train Recon')
    ax.plot(epochs, val_recon,   'r-', linewidth=2, label='Val Recon')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title('Reconstruction Loss (↓ lower = better fidelity)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max(epochs))

    # 3. Beat Loss (1 - regularity)
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(epochs, train_beat, 'b-', linewidth=2, label='Train Beat Loss')
    ax.plot(epochs, val_beat,   'r-', linewidth=2, label='Val Beat Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Beat Loss')
    ax.set_title('Beat Loss (↓ lower = more regular)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max(epochs))

    # 4. KL & Beat Weight
    ax = fig.add_subplot(gs[1, 1])
    ax2 = ax.twinx()
    ax.plot(epochs, kl_raw, 'g-', linewidth=2, label='KL (nats/dim)')
    ax2.plot(epochs, beat_w, 'm--', linewidth=2, label='Beat Weight')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('KL raw', color='green')
    ax2.set_ylabel('Beat Weight', color='magenta')
    ax.set_title('KL Divergence & Beat Weight Warm-up', fontsize=11, fontweight='bold')
    ax.tick_params(axis='y', labelcolor='green')
    ax2.tick_params(axis='y', labelcolor='magenta')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max(epochs))

    fig.suptitle('BF-VAE v2 Training Process', fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--history', required=True, help='Path to history_v2.json')
    p.add_argument('--output',  default='training_curves.png', help='Output .png path')
    args = p.parse_args()
    plot_training_curves(args.history, args.output)
