#!/bin/bash

path="./data"
batch_size=2
learning_rate=1e-4
loss="l1"
theta=1.0
epochs=30
chromesz=64
model="original"
date=$(date '+%d_%m_%Y_%H_%M_%S');
experiment="${date}_experiment_${loss}_${model}_theta_${theta}_ep_${epochs}_bs_${batch_size}_chrome_${chromesz}"
out=".results/${experiment}"

# training 
python train.py -p ${path} -o ${out} --batch_size ${batch_size} --learning_rate ${learning_rate} --chromesz ${chromesz} --loss_type ${loss} --mask_probes --shift_range --model ${model} --theta ${theta} --epochs ${epochs}