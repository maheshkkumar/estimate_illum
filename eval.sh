#!/bin/bash

batch_size=1
chromesz=64
model="original"
out="./evaluation/"
weights="./results/test_model"

# evaluation
srun python eval.py -o ${out} --batch_size ${batch_size} --mask_probes --shift_range --weights ${weights} --model ${model}