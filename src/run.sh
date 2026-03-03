#!/bin/bash

## CUDA_VISIBLE_DEVICES=7 python train.py \
#torchrun --standalone --nproc_per_node=4 train.py \
#    --opt dash \
#    --family transformer \
#    --micro_batch_size 32 \
#    --train_batches 2000 \
#    --k_input 1e-3 \
#    --momentum 0.95 \
#    --beta2 0.95 \
#    --backend pytorch \
#    --update_freq 1 \
#    --model_device_index 0 ./out/test

python ./src/train.py \
    --opt dash \
    --micro_batch_size 128 \
    --train_batches 2000 \
    --model_device_index 0 \
    --k_input 3e-3 \
    --momentum 0.9 \
    --beta2 0.95 \
    --resume false \
    --backend pytorch ./out/test