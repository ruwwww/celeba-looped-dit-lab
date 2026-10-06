# Looped-DiT Implementation Plan

## Contracts & Architecture Details

### 1. `models/looped_dit.py`
- Arsitektur transformer DiT yang membagi block ke `(pre=2, core=4, post=2)`.
- Default: `hidden_size=576`, `num_heads=9` (head_dim=64), `mlp_ratio=2.6667` (SwiGLU) atau `hidden_size=512`, `num_heads=8` dengan total parameter ~48M–50M.
- **XSA (Exclusive Self-Attention):** Pada attention output di loop core:
  $$\hat{v} = \frac{v}{\|v\|_2}, \quad \text{out} \leftarrow \text{out} - (\text{out} \cdot \hat{v})\hat{v}$$
  Memastikan atensi pada loop berikutnya hanya mengekstrak konteks baru antar-token tanpa mengulang self-value secara redundan.
- **Deep Supervision Exits:**
  - Metode `forward(x, t, y=None, num_loops=4, exit_loops=())`:
  - Jika `exit_loops` diberikan, kembalikan `(final_output, {r: exit_output_r})`.
- Modulasi timestep $t$ dan class label $y$ menggunakan adaLN-Zero atau adaptive projection layer.

### 2. `models/flow.py`
- Optimal Transport Flow Matching dengan Deep Supervision:
  - `deep_supervision_weights(num_loops=4, weighting="exponential")`
  - `training_loss(model, x1, y=None, exit_weights=None, t=None, x0=None)`
  - Menghitung MSE velocity loss: $\mathcal{L} = w_{\text{final}} \mathcal{L}_{\text{final}} + \sum_{r} w_r \mathcal{L}_r$.
- `euler_sample(model, shape, y=None, cfg_scale=4.0, steps=50, num_loops=4)`:
  - ODE integration dari $t=0$ ke $t=1$ dengan Classifier-Free Guidance.

### 3. `tests/test_looped_dit.py`
- Test 1: Shape forward pass `[B, 32, 16, 16]` -> `[B, 32, 16, 16]`.
- Test 2: Deep supervision exits menghasilkan dict tensor sesuai exit yang diminta.
- Test 3: Backward pass menghasilkan gradien valid dan finite untuk seluruh parameter.
- Test 4: 5-step Euler sampling menghasilkan output tensor yang valid.

### 4. `train.py` & `sample.py`
- Terintegrasi dengan `tracker.py` (`ExperimentTracker`):
  - Output direktori terstruktur: `checkpoints/`, `figures/`, `metrics.jsonl`, `config.json`.
  - Auto-generate `figures/loss_curve.png` dan sample grid berkala.
- `sample.py`: CLI sampler mandiri dengan decode via `load_frozen_vae`.
