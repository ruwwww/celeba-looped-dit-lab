# CelebA-HQ Looped-DiT Lab (~50M Parameters)

Implementasi arsitektur **Looped-DiT** (OpenSenseNova / arXiv:2609.40305) pada latent space CelebA-HQ $16 \times 16 \times 32$ kanal dengan frozen VAE f16c32.

## 1. Arsitektur Looped-DiT (~50M Parameter)
Looped-DiT membagi transformer block menjadi 3 tahap:
- **Pre-loop ($A$):** `pre = 2` block (dieksekusi 1 kali di awal).
- **Looped Core ($B$):** `core = 4` block yang di-share bobotnya dan di-loop sebanyak $N = 4$ kali per denoising step.
  - Menggunakan **Self-Modulating Attention** (Exclusive Self-Attention / XSA atau Attention Gate) untuk meregulasi update perhatian dalam loop.
- **Post-loop ($C$):** `post = 2` block (dieksekusi 1 kali sebelum output projection).
- **Total Parameter Fisik:** Hanya $2 + 4 + 2 = 8$ transformer block (~48M–50M parameter dengan hidden_size=576 atau hidden_size=512).
- **Kedalaman Efektif:** $2 + (4 \times 4) + 2 = 20$ layer! Kompak di bobot, tetapi berkapasitas komputasi model 20 layer.

## 2. Deep Supervision
- Setiap iterasi loop $r \in \{1, \dots, N\}$ memiliki intermediate state $h_r$ yang dapat didekode melalui post-loop block $C$.
- Pelatihan menggunakan **Deep Supervision** dengan bobot exit terdistribusi (exponential atau uniform) untuk melatih intermediate exits agar konvergen ke target velocity yang sama.

## 3. Spesifikasi Teknis
- **Latent Resolution:** $16 \times 16$, 32 kanal.
- **Conditioning:** 64 cluster DINOv2 dengan class dropout 15% untuk Classifier-Free Guidance (CFG scale 3.0–4.0).
- **Formulasi Flow:** Optimal Transport Flow Matching ($x_t = t \cdot x_1 + (1-t) \cdot x_0$, target $u_t = x_1 - x_0$).
- **Sampler:** Euler ODE Integrator dengan dukungan `num_loops` dinamis dan CFG.
- **Standard Pelatihan & Logging:** Menggunakan `tracker.py` (`metrics.jsonl`, `figures/loss_curve.png`, `figures/sample_step_*.png`).

## 4. Struktur Modul
- `models/looped_dit.py`: LoopedDiT backbone, DoubleStream / SingleStream Transformer Blocks dengan XSA dan adaLN-Zero / time modulation.
- `models/flow.py`: OT Flow Matching dengan Deep Supervision loss dan Euler ODE sampler.
- `models/__init__.py`: Export modul model.
- `tests/test_looped_dit.py`: Pytest suite (forward pass, loop exits, loss backward, 5-step Euler sampling).
- `train.py`: Training CLI terstandarisasi dengan ExperimentTracker, AMP bfloat16, AdamW, EMA, periodic figure generation.
- `sample.py`: CLI standalone sampler dengan parameter `--loops`, `--cfg-scale`, `--steps`.
