#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/Coding3/celeba-looped-dit-lab

mkdir -p outputs/looped_dit_50m_v2

# Looped-DiT 47.5M Parameters (Stabilized V2):
# hidden_size=512, mlp_ratio=6.0, loop_split=(2, 4, 2)
# Physical blocks: 8 blocks (47.46M params)
# Loops: N=4 times for core blocks -> Effective depth: 2 + (4*4) + 2 = 20 blocks!
# Stabilized:
# 1. QK-Norm (RMSNorm on Query & Key in SelfAttention)
# 2. Token-aligned Exclusive Self-Attention (XSA) per head
# 3. Loop-boundary RMSNorm to prevent recurrent activation drift
# 4. Normalized Deep Supervision (weights sum to 2.0)
# 5. Bounded adaLN modulation (scale clamped to [-2, 2], gate clamped to [-4, 4])
# 6. Gradient clipping at 0.5

PYTHONPATH=. /home/kuroko/.conda/envs/ai/bin/python train.py \
  --cache cached_data/celeba_train_latents_clustered.pt \
  --output-dir outputs/looped_dit_50m_v2 \
  --batch-size 32 \
  --workers 0 \
  --max-steps 105000 \
  --hidden-size 512 \
  --mlp-ratio 6.0 \
  --loop-split 2 4 2 \
  --num-loops 4 \
  --lr 2e-4 \
  --grad-clip 0.5 \
  --cfg-scale 4.0 \
  --log-interval 200 \
  --checkpoint-interval 10000 \
  --sample-interval 1000 \
  --sample-steps 50 \
  --sample-count 6 \
  --amp-bf16 > outputs/looped_dit_50m_v2/train.log 2>&1
