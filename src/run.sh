#!/bin/bash

python train.py \
    --opt dash \
    --micro_batch_size 32 \
    --train_batches 2000 \
    --k_input 1e-3 \
    --momentum 0.95 \
    --beta2 0.95 \
    --model_device_index 0 ./out/test