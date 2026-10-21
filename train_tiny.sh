#!/usr/bin/env bash
set -euo pipefail

python train_pscvstereo.py --model tiny \
    --name pscvstereo_tiny \
    --mixed_precision \
    --batch_size 16 \
    --lr 0.0005 \
    --wdecay 0.00001 \
    --num_steps 50000 \
    --train_datasets vkitti middlebury_2014_no_middeval3 middlebury_2021 eth3d instereo2k tartan_air \
    --dataset_repeat 1 200 200 200 5 20 \
    --image_size 416 576 \
    --gpus 0 \
    --rand_seed 4312 \
    --do_flip v \
    --all_left_prob 0.0 \
    --v_shift_prob 0.0 \
    --ext_downsamp_prob 1.0

python train_pscvstereo.py --model tiny \
    --name pscvstereo_tiny_finetune \
    --mixed_precision \
    --restore_ckpt checkpoints/pscvstereo_tiny.pth \
    --batch_size 16 \
    --lr 0.0003 \
    --wdecay 0.00001 \
    --num_steps 30000 \
    --train_datasets vkitti middlebury_2014_no_middeval3 middlebury_2021 eth3d instereo2k tartan_air \
    --dataset_repeat 1 200 200 200 5 20 \
    --image_size 416 576 \
    --gpus 0 \
    --rand_seed 4312 \
    --do_flip v \
    --all_left_prob 0.0 \
    --v_shift_prob 0.5 \
    --ext_downsamp_prob 0.0