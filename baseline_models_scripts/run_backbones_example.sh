#!/bin/bash
# Example: run the same experiment with several backbones one by one.

ENCODERS=(resnet18 resnet34 efficientnet-b0 mobilenet_v2)
for enc in "${ENCODERS[@]}"; do
  python train_ttd.py \
    --project-root /users/7/yu001011/csci5527/CSCI5527-final \
    --experiment Multi-Domain \
    --arch Unet \
    --encoder "$enc" \
    --encoder-weights imagenet \
    --epochs 100 \
    --batch-size 16 \
    --img-size 512 \
    --lr 1e-4 \
    --quiet
 done
