#!/bin/bash

python evaluate_five.py ../data/preprocessed_dev.jsonl \
    --checkpoint ./checkpoints_five/best_model.pt \
    --train_data ../data/synthetic/preprocessed_synth.jsonl \
    --show_errors \
    --component_analysis