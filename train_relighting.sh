#!/bin/bash

path="./data"
batch_size=2
learning_rate=1e-4
chromesz=64
model="unet"
loss="l1"
theta=1.0
epochs=30
date=$(date '+%d_%m_%Y_%H_%M_%S');
experiment="${date}_relit_experiment_${loss}_${model}_theta_${theta}_ep_${epochs}_bs_${batch_size}_chrome_${chromesz}_no_crop"
out=".results/${experiment}"

# training
srun python train_relighting.py -p ${path} -o ${out} --batch_size ${batch_size} --learning_rate ${learning_rate} --chromesz ${chromesz} --loss_type ${loss} --mask_probes --shift_range --model ${model} --theta ${theta} --epochs ${epochs}