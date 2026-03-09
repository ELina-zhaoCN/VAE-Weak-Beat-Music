# BF-VAE Weak-Beat Music Enhancement System · Architecture (Text Version)

---

## I. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                        BF-VAE WEAK-BEAT MUSIC ENHANCEMENT SYSTEM                              │
│                              Layered System Architecture                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 4: APPLICATION LAYER                                                                  │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐              │
│  │ Beatmap Generator   │───►│ SongLibrary.ts      │───►│ Lens Studio         │              │
│  │ (generate_beatmap)  │    │ (TypeScript export) │    │ (Rhythm Game)        │              │
│  │ BPM, onset, lanes   │    │ AllSongs[]          │    │ MusicMaster.esproj  │              │
│  └─────────────────────┘    └─────────────────────┘    └─────────────────────┘              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                          ▲
                                          │ WAV, JSON
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 3: INFERENCE / SERVICE LAYER                                                          │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────┐ │
│  │ Beat         │   │ Chunking     │   │ VAE          │   │ Overlap-Add  │   │ Audio      │ │
│  │ Detector     │──►│ Engine       │──►│ Forward      │──►│ Merger       │──►│ Output     │ │
│  │ (weak-beat?) │   │ 10s+2.5s     │   │ (model)      │   │ crossfade    │   │ WAV        │ │
│  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └────────────┘ │
│         │                    │                  │                                          │
│         └────────────────────┴──────────────────┘                                          │
│                              │                                                              │
│                    inference_v2.py                                                           │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                          ▲
                                          │ Mel chunks
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 2: MODEL / CORE LAYER                                                                  │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                    MelSpectrogramVAE (vae_model.py)                                    │   │
│  │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                             │   │
│  │  │ Encoder     │────►│ Latent z    │────►│ Decoder     │                             │   │
│  │  │ Conv×3      │     │ 128-dim     │     │ ConvT×3     │                             │   │
│  │  └─────────────┘     └─────────────┘     └─────────────┘                             │   │
│  └───────────────────────────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐                  │
│  │ Recon Loss (MSE)    │  │ KL Divergence       │  │ Beat Loss           │                  │
│  │                     │  │ Free Bits           │  │ FFT autocorr        │                  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘                  │
│                                      train_v2.py                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                                          ▲
                                          │ Mel (1,128,431)
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1: DATA LAYER                                                                         │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐               │
│  │ FMA Dataset         │   │ Metadata            │   │ AudioMelDataset_v2  │               │
│  │ (raw audio)         │   │ tracks.csv          │   │ 22050Hz, Mel, norm  │               │
│  └──────────┬──────────┘   └──────────┬──────────┘   └──────────┬──────────┘               │
│             │                         │                         │                          │
│             └─────────────────────────┴─────────────────────────┘                          │
│                                       │                                                     │
│                             fma_filter.py                                                   │
│                                       │                                                     │
│  ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐               │
│  │ weak_beat_music/    │   │ data_split.json     │   │ best_model_v2.pth   │               │
│  │ MP3, WAV            │   │ train/val/test     │   │ checkpoints         │               │
│  └─────────────────────┘   └─────────────────────┘   └─────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│  EXTERNAL DEPENDENCIES                                                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│  PyTorch │ librosa │ soundfile │ numpy │ oops-i-tapped-it-again (beatmap) │ Lens Studio     │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## II. VAE Model Structure

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MelSpectrogramVAE Model Architecture                          │
└─────────────────────────────────────────────────────────────────────────────────┘

Input: (batch, 1, 128, 431)  ← Mel spectrogram, ~10s @ 22050Hz

┌──────────────────────────────────────────────────────────────────────────────────┐
│ ENCODER                                                                            │
├──────────────────────────────────────────────────────────────────────────────────┤
│  Conv2d(1→32, 4×4, stride=2)  + BatchNorm + ReLU    →  (32, 64, 215)              │
│  Conv2d(32→64, 4×4, stride=2) + BatchNorm + ReLU   →  (64, 32, 107)              │
│  Conv2d(64→128, 4×4, stride=2) + BatchNorm + ReLU   →  (128, 16, 53)              │
│  Flatten                                            →  108544                    │
│  Linear(108544 → 128)                               →  μ (mean)                   │
│  Linear(108544 → 128)                               →  log σ² (log variance)     │
│  Reparameterization: z = μ + σ * ε, ε ~ N(0,1)     →  latent z (128-dim)         │
└──────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ LATENT SPACE                                                                       │
│  Dimension: 128                                                                    │
└──────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ DECODER                                                                            │
├──────────────────────────────────────────────────────────────────────────────────┤
│  Linear(128 → 108544)                              →  (128, 16, 53)              │
│  ConvTranspose2d(128→64, 4×4, stride=2) + BN+ReLU  →  (64, 32, 106)              │
│  ConvTranspose2d(64→32, 4×4, stride=2)  + BN+ReLU  →  (32, 64, 212)              │
│  ConvTranspose2d(32→1, 4×4, stride=2)              →  (1, 128, 431)              │
└──────────────────────────────────────────────────────────────────────────────────┘

Output: (batch, 1, 128, 431)  ← Reconstructed Mel spectrogram
```

---

## III. Loss Function Structure

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Total Loss = Recon + β·KL + γ·Beat                                               │
└─────────────────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────┐
  │ Recon Loss (MSE)    │  MSE between reconstructed and input Mel
  │ Full-band / config  │
  └─────────────────────┘

  ┌─────────────────────┐
  │ KL Divergence       │  KL( q(z|x) || N(0,1) )
  │ Free Bits reg       │  kl_per_dim = max(kl_per_dim, 0.5)
  │ β = 0.01            │
  └─────────────────────┘

  ┌─────────────────────┐
  │ Beat Loss          │  loss = 1 - regularity_score
  │ FFT autocorr       │  regularity ∈ [0,1], higher = more regular beats
  │ γ warm-up 0→0.3     │
  └─────────────────────┘
```

---

## IV. Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Training Pipeline (train_v2.py)                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

  weak_beat_music/                    data_split.json
  (MP3/WAV files)                     (file-level split)
        │                                    │
        ▼                                    ▼
  ┌─────────────────────────────────────────────────────┐
  │ AudioMelDataset_v2                                  │
  │  - Load audio → 22050Hz mono                        │
  │  - Mel spectrogram (128×431)                        │
  │  - Normalize: clip(db/40, -1, 1)                   │
  └─────────────────────────────────────────────────────┘
        │
        ├── Train (70%)  ──► DataLoader ──► Per batch
        ├── Val   (15%)  ──► DataLoader       │
        └── Test  (15%)  ──► Eval only        │
                                              ▼
  ┌─────────────────────────────────────────────────────┐
  │ Per batch:                                           │
  │   mel_in → VAE(mel_in) → mel_out, μ, logvar         │
  │   loss = MSE(mel_out, mel_in)                        │
  │        + β·KL(μ, logvar)                             │
  │        + γ·(1 - beat_regularity(mel_out))            │
  │   backward → optimizer.step()                        │
  └─────────────────────────────────────────────────────┘
        │
        ▼
  best_model_v2.pth  (highest val regularity)
  history_v2.json   (per-epoch metrics)
```

---

## V. Inference Enhancement Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Inference Pipeline (inference_v2.py) · Supports arbitrary-length audio           │
└─────────────────────────────────────────────────────────────────────────────────┘

  Input: Arbitrary-length MP3/WAV
        │
        ▼
  ┌─────────────────────────────────────────────────────┐
  │ 1. Detect weak-beat                                  │
  │    chunk_audio → compute regularity per chunk        │
  │    mean < 0.45 → WEAK BEAT, recommend enhancement   │
  └─────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────┐
  │ 2. Chunking (Overlap-Add)                            │
  │    chunk_sec=10s, overlap=2.5s                      │
  │    Tail chunk: fade-out + zero-pad                  │
  └─────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────┐
  │ 3. Per-chunk VAE forward                             │
  │    chunk → audio_to_mel → VAE → mel_to_audio        │
  │    (Griffin-Lim inverse)                            │
  └─────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────┐
  │ 4. Overlap-Add merge                                 │
  │    Raised-cosine crossfade                           │
  │    Tail: silence padded region + global fade (3s)    │
  │    Output normalize: peak → 0.85                     │
  └─────────────────────────────────────────────────────┘
        │
        ▼
  Output: output_enhanced.wav
          output_comparison.png (optional)
```

---

## VI. End-to-End Application Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ run_e2e_demo.sh One-Click Pipeline                                               │
└─────────────────────────────────────────────────────────────────────────────────┘

  weak_beat_music/101765.mp3
        │
        ▼
  [Step A] inference_v2.py
        │
        ├── output_enhanced.wav
        └── output_comparison.png
        │
        ▼
  [Step B] generate_beatmap.py (oops-i-tapped-it-again)
        │
        ├── output.json (BPM, notes)
        └── SongLibrary.ts (TypeScript)
        │
        ▼
  [Step C] Lens Studio
        │
        └── MusicMaster.esproj → Rhythm game
```

---

## VII. Beat Loss Computation Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ Beat Regularity Score (beat_loss_v2.py)                                           │
└─────────────────────────────────────────────────────────────────────────────────┘

  Mel spectrogram (1, 128, 431)
        │
        ▼
  Extract low-freq band energy (e.g. dims 0~32) → time series
        │
        ▼
  FFT autocorrelation: correlate(energy, energy)
        │
        ▼
  Find dominant period peak (exclude lag=0)
        │
        ▼
  regularity = peak_value  (0~1)
        │
        ▼
  beat_loss = 1 - regularity  (differentiable, for backprop)
```

---

## VIII. Directory and Module Mapping

```
Final_model/
├── 1.Filter_fma_weak_beat/     # Data filtering
├── 2.Music_dataset/            # Dataset definitions
├── 3.Model/                    # VAE model (vae_model.py)
├── 4.Beat_losses/              # Beat Loss v1 (has bug)
├── 5.Tran_Script/              # Training script v1
├── 6.Test_Script/              # Evaluation script v1
├── 7.BF_VAE_v2/                # v2 full implementation
│   ├── audio_dataset_v2.py     # Dataset
│   ├── beat_loss_v2.py         # Fixed Beat Loss
│   ├── train_v2.py             # Training
│   ├── inference_v2.py         # Inference
│   ├── evaluate_v2.py          # Test-set evaluation
│   └── checkpoints/            # Model and history
├── run_e2e_demo.sh             # End-to-end script
└── oops-i-tapped-it-again/     # Rhythm game (external)
```
