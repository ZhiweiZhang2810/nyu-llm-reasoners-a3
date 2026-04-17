#!/bin/bash

# Experiment settings
STEPS=200
OUTPUT_DIR="./outputs"
mkdir -p $OUTPUT_DIR

# echo "Starting SFT Experiments..."
# for N in 128 256 512 1024
# do
#     echo "Running SFT with N=$N..."
#     uv run student/train.py --mode sft --num_examples $N --output_dir $OUTPUT_DIR
# done

echo "Starting GRPO Countdown Baseline & LR Sweep..."
for LR in 1e-5 5e-6 2e-6
do
    echo "Running GRPO Countdown with LR=$LR..."
    uv run student/train.py --mode grpo --dataset countdown --lr $LR --steps $STEPS --output_dir $OUTPUT_DIR
done

echo "Starting GRPO Baselining Experiments (using LR 1e-5)..."
echo "Running GRPO with no_baseline..."
uv run student/train.py --mode grpo --dataset countdown --lr 1e-5 --loss_type no_baseline --steps $STEPS --output_dir $OUTPUT_DIR
echo "Running GRPO with reinforce_with_baseline..."
uv run student/train.py --mode grpo --dataset countdown --lr 1e-5 --loss_type reinforce_with_baseline --steps $STEPS --output_dir $OUTPUT_DIR

echo "Starting GRPO Length Normalization Experiments..."
echo "Running GRPO with length_norm=masked_normalize..."
uv run student/train.py --mode grpo --dataset countdown --lr 1e-5 --loss_type reinforce_with_baseline --length_norm masked_normalize --steps $STEPS --output_dir $OUTPUT_DIR
# (masked_mean is covered by the baseline run above)

echo "Starting GRPO STD Normalization Experiments..."
echo "Running GRPO with no_std_norm..."
uv run student/train.py --mode grpo --dataset countdown --lr 1e-5 --loss_type reinforce_with_baseline --no_std_norm --steps $STEPS --output_dir $OUTPUT_DIR

echo "All experiments completed. Check $OUTPUT_DIR for plots."
