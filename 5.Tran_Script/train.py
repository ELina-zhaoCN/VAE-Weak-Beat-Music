#!/usr/bin/env python3
"""
训练脚本：弱节拍音乐 VAE 模型
- 使用重建损失 + KL 散度 + 节拍损失
- 渐进式热身策略（β 和 γ 权重线性增加）
- 自动划分训练/验证集 (80/20)
- 保存最佳模型（基于验证集节拍损失）
- 支持 TensorBoard 记录
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import librosa
from tqdm import tqdm

# ------------------------------------------------------------
# 将项目模块路径加入系统路径（根据你的文件夹结构调整）
# ------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), '../2.Music_dataset'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../3.Model'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../4.Beat_losses'))

# 导入自定义模块（确保文件名和类名与实际一致）
from audio_dataset import AudioMelDataset as MusicDataset      # 数据加载类
from vae_model import MelSpectrogramVAE as VAE             # VAE 模型类（请确认类名）
from beat_loss import beat_loss               # 节拍损失函数


def parse_args():
    parser = argparse.ArgumentParser(description='训练 VAE 生成规律节拍音乐')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='音频文件夹路径（已筛选的弱节拍音乐）')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='潜在空间维度')
    parser.add_argument('--kl_weight', type=float, default=1.0,
                        help='KL 损失最终权重')
    parser.add_argument('--beat_weight', type=float, default=0.5,
                        help='节拍损失最终权重')
    parser.add_argument('--warmup_epochs', type=int, default=20,
                        help='热身轮数（权重线性增加到设定值）')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='TensorBoard 日志目录')
    return parser.parse_args()


def compute_bpm_consistency(recon_mel):
    """
    计算重建音乐的 BPM 一致性（作为节拍规律性的代理指标）
    输入: recon_mel (batch, 1, n_mels, n_frames)
    返回: 标量一致性得分 (0~1)
    """
    batch_size = recon_mel.size(0)
    low_freq = recon_mel[:, :, :10, :].mean(dim=2)  # (batch, 1, n_frames)
    low_freq = low_freq.squeeze(1).cpu().numpy()
    scores = []
    for i in range(batch_size):
        signal = low_freq[i]
        corr = np.correlate(signal, signal, mode='same')
        mid = len(corr) // 2
        pos_corr = corr[mid+1:]
        if len(pos_corr) == 0:
            scores.append(0.0)
            continue
        max_corr = np.max(pos_corr)
        norm = np.dot(signal, signal) + 1e-8
        score = max_corr / norm
        scores.append(score)
    return float(np.mean(scores))


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    # 加载数据集
    print('正在加载数据集...')
    full_dataset = MusicDataset(data_dir=args.data_dir)
    print(f'总样本数: {len(full_dataset)}')

    # 划分训练集和验证集 (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f'训练样本: {train_size}, 验证样本: {val_size}')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 初始化模型
    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    recon_criterion = nn.MSELoss()

    # 热身参数
    kl_weight = 0.0
    beat_weight = 0.0
    kl_step = args.kl_weight / args.warmup_epochs
    beat_step = args.beat_weight / args.warmup_epochs

    best_val_beat_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # 更新热身权重
        if epoch <= args.warmup_epochs:
            kl_weight = min(kl_weight + kl_step, args.kl_weight)
            beat_weight = min(beat_weight + beat_step, args.beat_weight)
        else:
            kl_weight = args.kl_weight
            beat_weight = args.beat_weight

        # ------------------ 训练阶段 ------------------
        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0
        train_beat = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs} [Train]')
        for batch in pbar:
            batch = batch.to(device)

            recon, mu, logvar = model(batch)
            # 调整重建张量的时间维度以匹配目标
            if recon.size(-1) != batch.size(-1):
                 recon = recon[:, :, :, :batch.size(-1)]   # 截取到目标帧数
            recon_loss = recon_criterion(recon, batch)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / batch.size(0)
            b_loss = beat_loss(recon)

            loss = recon_loss + kl_weight * kl_loss + beat_weight * b_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.size(0)
            train_recon += recon_loss.item() * batch.size(0)
            train_kl += kl_loss.item() * batch.size(0)
            train_beat += b_loss.item() * batch.size(0)

            pbar.set_postfix({
                'loss': loss.item(),
                'recon': recon_loss.item(),
                'kl': kl_loss.item(),
                'beat': b_loss.item()
            })

        train_loss /= len(train_dataset)
        train_recon /= len(train_dataset)
        train_kl /= len(train_dataset)
        train_beat /= len(train_dataset)

        # ------------------ 验证阶段 ------------------
        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0
        val_beat = 0.0
        bpm_scores = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, mu, logvar = model(batch)
                if recon.size(-1) != batch.size(-1):
                     recon = recon[:, :, :, :batch.size(-1)]
                     if recon.size(-1) != batch.size(-1):
                         recon = recon[:, :, :, :batch.size(-1)]
                recon_loss = recon_criterion(recon, batch)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl_loss = kl_loss / batch.size(0)
                b_loss = beat_loss(recon)

                loss = recon_loss + kl_weight * kl_loss + beat_weight * b_loss

                val_loss += loss.item() * batch.size(0)
                val_recon += recon_loss.item() * batch.size(0)
                val_kl += kl_loss.item() * batch.size(0)
                val_beat += b_loss.item() * batch.size(0)

                bpm_scores.append(compute_bpm_consistency(recon) * batch.size(0))

        val_loss /= len(val_dataset)
        val_recon /= len(val_dataset)
        val_kl /= len(val_dataset)
        val_beat /= len(val_dataset)
        avg_bpm = sum(bpm_scores) / len(val_dataset)

        # 打印结果
        print(f'\nEpoch {epoch}:')
        print(f'  Train - Loss: {train_loss:.4f} (recon: {train_recon:.4f}, kl: {train_kl:.4f}, beat: {train_beat:.4f})')
        print(f'  Val   - Loss: {val_loss:.4f} (recon: {val_recon:.4f}, kl: {val_kl:.4f}, beat: {val_beat:.4f})')
        print(f'  BPM一致性: {avg_bpm:.4f}  | 权重: β={kl_weight:.3f}, γ={beat_weight:.3f}')

        # TensorBoard 记录
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Recon/train', train_recon, epoch)
        writer.add_scalar('Recon/val', val_recon, epoch)
        writer.add_scalar('KL/train', train_kl, epoch)
        writer.add_scalar('KL/val', val_kl, epoch)
        writer.add_scalar('Beat/train', train_beat, epoch)
        writer.add_scalar('Beat/val', val_beat, epoch)
        writer.add_scalar('BPM/val', avg_bpm, epoch)
        writer.add_scalar('Weights/kl', kl_weight, epoch)
        writer.add_scalar('Weights/beat', beat_weight, epoch)

        # 保存最佳模型（基于验证集节拍损失）
        if val_beat < best_val_beat_loss:
            best_val_beat_loss = val_beat
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print(f'  ✔ 保存最佳模型 (val beat loss: {val_beat:.4f})')

    writer.close()
    print('训练完成！')


if __name__ == '__main__':
    main()