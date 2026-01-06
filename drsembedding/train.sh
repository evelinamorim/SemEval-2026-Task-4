#!/bin/bash

python train.py ../data/synthetic/preprocessed_synth.jsonl \
    --epochs 50 \
    --batch_size 32 \
    --hidden_dim 128 \
    --output_dim 64 \
    --lr 0.001 \
    --dropout 0.1 \
    --patience 10 \
    --device auto
