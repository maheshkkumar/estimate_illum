#!/bin/bash

path="./data"
batch_size=2
learning_rate=1e-4
loss="l1"
theta=1.0
epochs=30
date=$(date '+%d_%m_%Y_%H_%M_%S');
experiment="${date}_relit_baseline_${loss}_theta_${theta}_ep_${epochs}_bs_${batch_size}_no_crop_random_relight"
out="./results/${experiment}"

# training
srun python train_relighting_baseline.py -p ${path} -o ${out} --batch_size ${batch_size} --learning_rate ${learning_rate} --loss_type ${loss} --mask_probes --shift_range --theta ${theta} --epochs ${epochs} --random_relight