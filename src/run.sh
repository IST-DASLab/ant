#!/bin/bash

clear

#CUDA_VISIBLE_DEVICES=7 python train.py \
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 train.py \
    --opt dash \
    --micro_batch_size 32 \
    --train_batches 2000 \
    --model_device_index 0 \
    --k_input 3e-3 \
    --momentum 0.9 \
    --beta2 0.95 \
    --resume false \
    \
    --family transformer \
    --update_freq 1 \
    --backend pytorch \
    \
    ./out/test