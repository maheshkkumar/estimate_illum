#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate python3
echo "Environment activated"

batch_size=1
chromesz=64

model="original"
out="./evaluation/"
weights="./results/test_model"
srun python eval.py -o ${out} --batch_size ${batch_size} --mask_probes --shift_range --weights ${weights} --model ${model}