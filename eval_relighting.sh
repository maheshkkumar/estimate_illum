#!/bin/bash

path="./data"
batch_size=1
learning_rate=1e-4
chromesz=64
out="./evaluation"
weights="./results/"

# evaluation
echo "eval_relighting.py -o ${out} --batch_size ${batch_size} --mask_probes --shift_range --weights ${weights} --reference_prob"