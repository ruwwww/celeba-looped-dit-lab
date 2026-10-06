#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/Coding3/celeba-looped-dit-lab

mkdir -p outputs/looped_dit_50m

# Looped-DiT ~49.0M Parameters:
# hidden_size=512, mlp_ratio=6.0, loop_split=(2, 4, 2)
# Physical blocks: 8 blocks (49.03M params)
# Loops: N=4 times for core blocks -> Effective depth: 2 + (4*4) + 2 = 20 blocks!
# Training with Deep Supervision, XSA, and CFG scale 4.0
# Batch size 32 on cached latents, 105,000 steps (~120 epochs)

PYTHONPATH=. /home/kuroko/.conda/envs/ai/bin/python train.py \
  --cache cached_data/celeba_train_latents_clustered.pt \
  --output-dir outputs/looped_dit_50m \
  --batch-size 32 \
  --max-steps 105000 \
  --hidden-size 512 \
  --mlp-ratio 6.0 \
  --loop-split 2 4 2 \
  --num-loops 4 \
  --lr 2e-4 \
  --cfg-scale 4.0 \
  --log-interval 200 \
  --checkpoint-interval 10000 \
  --sample-interval 1000 \
  --sample-steps 50 \
  --sample-count 6 \
  --amp-bf16 > outputs/looped_dit_50m/train.log 2>&1
