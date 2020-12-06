#!/bin/bash
#SBATCH --job-name=IllEst     	            # Name that will show up in squeue
#SBATCH --gres=gpu:1                        # Request 4 GPU "generic resource"
#SBATCH --time=7-00:00       		        # Max job time is 3 hours
#SBATCH --output=./logs/train/%N-%j.out   	# Terminal output to file named (hostname)-(jobid).out
#SBATCH --error=./logs/train/%N-%j.err		# Error log with format (hostname)-(jobid).out
#SBATCH --partition=long     		        # Partitions (long -- 7 days low priority, short -- 2 days intermediate priority, debug -- 4 hours high priority runtime)
#SBATCH --mail-type=all       		        # Send email on job start, end and fail
#SBATCH --mail-user=mahesh_reddy@sfu.ca
#SBATCH --qos=overcap
#SBATCH --nodelist=cs-venus-01

# The SBATCH directives above set options similarly to command line arguments to srun
# The job will appear on squeue and output will be written to the current directory
# You can do tail -f <output_filename> to track the job.
# You can kill the job using scancel <job_id> where you can find the <job_id> from squeue

# Your experiment setup logic here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate python3
echo "Environment activated"

path="./data"
batch_size=8
learning_rate=1e-4
loss="l1"
theta=1.0
epochs=40

# l1 loss and cropping
# date=$(date '+%d_%m_%Y_%H_%M_%S');
# model="${date}_experiment"
# out="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/${model}"
# srun python train.py -p ${path} -o ${out} --batch_size ${batch_size} --learning_rate ${learning_rate} --chromesz ${chromesz} --crop_images --loss_type "l1"

# l1 loss, cropping and shift range
# date=$(date '+%d_%m_%Y_%H_%M_%S');
# model="${date}_experiment"
# out="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/${model}"
# srun python train.py -p ${path} -o ${out} --batch_size ${batch_size} --learning_rate ${learning_rate} --chromesz ${chromesz} --crop_images --loss_type "l1" --shift_range

# l1 loss, croppping and masking probes
# date=$(date '+%d_%m_%Y_%H_%M_%S');
# model="${date}_experiment"
# out="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/${model}"
# srun python train.py -p ${path} -o ${out} --batch_size ${batch_size} --learning_rate ${learning_rate} --chromesz ${chromesz} --crop_images --mask_probes --loss_type "l1"

# l1 loss, croppping, masking probes and shift range
chromesz=64
model="resnet18"
date=$(date '+%d_%m_%Y_%H_%M_%S');
experiment="${date}_experiment_${loss}_${model}_theta_${theta}_ep_${epochs}_bs_${batch_size}_chrome_${chromesz}_no_crop"
out="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/${experiment}"
srun python train.py -p ${path} -o ${out} --batch_size ${batch_size} --learning_rate ${learning_rate} --chromesz ${chromesz} --loss_type ${loss} --mask_probes --shift_range --model ${model} --theta ${theta} --epochs ${epochs}

model="unet"
date=$(date '+%d_%m_%Y_%H_%M_%S');
experiment="${date}_experiment_${loss}_${model}_theta_${theta}_ep_${epochs}_bs_${batch_size}_chrome_${chromesz}_no_crop"
out="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/${experiment}"
srun python train.py -p ${path} -o ${out} --batch_size ${batch_size} --learning_rate ${learning_rate} --chromesz ${chromesz} --loss_type ${loss} --mask_probes --shift_range --model ${model} --theta ${theta} --epochs ${epochs}

# l1 loss, croppping, masking probes and shift range
chromesz=128
model="resnet18"
date=$(date '+%d_%m_%Y_%H_%M_%S');
experiment="${date}_experiment_${loss}_${model}_theta_${theta}_ep_${epochs}_bs_${batch_size}_chrome_${chromesz}_no_crop"
out="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/${experiment}"
srun python train.py -p ${path} -o ${out} --batch_size ${batch_size} --learning_rate ${learning_rate} --chromesz ${chromesz} --loss_type ${loss} --mask_probes --shift_range --model ${model} --theta ${theta} --epochs ${epochs}

model="unet"
date=$(date '+%d_%m_%Y_%H_%M_%S');
experiment="${date}_experiment_${loss}_${model}_theta_${theta}_ep_${epochs}_bs_${batch_size}_chrome_${chromesz}_no_crop"
out="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/${experiment}"
srun python train.py -p ${path} -o ${out} --batch_size ${batch_size} --learning_rate ${learning_rate} --chromesz ${chromesz} --loss_type ${loss} --mask_probes --shift_range --model ${model} --theta ${theta} --epochs ${epochs}

# l1 loss, croppping, masking probes and shift range
chromesz=256
model="resnet18"
date=$(date '+%d_%m_%Y_%H_%M_%S');
experiment="${date}_experiment_${loss}_${model}_theta_${theta}_ep_${epochs}_bs_${batch_size}_chrome_${chromesz}_no_crop"
out="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/${experiment}"
srun python train.py -p ${path} -o ${out} --batch_size ${batch_size} --learning_rate ${learning_rate} --chromesz ${chromesz} --loss_type ${loss} --mask_probes --shift_range --model ${model} --theta ${theta} --epochs ${epochs}

model="unet"
date=$(date '+%d_%m_%Y_%H_%M_%S');
experiment="${date}_experiment_${loss}_${model}_theta_${theta}_ep_${epochs}_bs_${batch_size}_chrome_${chromesz}_no_crop"
out="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/${experiment}"
srun python train.py -p ${path} -o ${out} --batch_size ${batch_size} --learning_rate ${learning_rate} --chromesz ${chromesz} --loss_type ${loss} --mask_probes --shift_range --model ${model} --theta ${theta} --epochs ${epochs}

# # l1 loss, croppping, masking probes and shift range
# chromesz=64
# model="vgg"
# date=$(date '+%d_%m_%Y_%H_%M_%S');
# experiment="${date}_experiment_${loss}_${model}_theta_${theta}_ep_${epochs}_bs_${batch_size}_chrome_${chromesz}_no_crop"
# out="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/${experiment}"
# srun python train.py -p ${path} -o ${out} --batch_size ${batch_size} --learning_rate ${learning_rate} --chromesz ${chromesz} --loss_type ${loss} --mask_probes --shift_range --model ${model} --theta ${theta} --epochs ${epochs}

# model="original"
# date=$(date '+%d_%m_%Y_%H_%M_%S');
# experiment="${date}_experiment_${loss}_${model}_theta_${theta}_ep_${epochs}_bs_${batch_size}_chrome_${chromesz}_no_crop"
# out="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/${experiment}"
# srun python train.py -p ${path} -o ${out} --batch_size ${batch_size} --learning_rate ${learning_rate} --chromesz ${chromesz} --loss_type ${loss} --mask_probes --shift_range --model ${model} --theta ${theta} --epochs ${epochs}

# # l1 loss, croppping, masking probes and shift range
# chromesz=128
# model="vgg"
# date=$(date '+%d_%m_%Y_%H_%M_%S');
# experiment="${date}_experiment_${loss}_${model}_theta_${theta}_ep_${epochs}_bs_${batch_size}_chrome_${chromesz}_no_crop"
# out="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/${experiment}"
# srun python train.py -p ${path} -o ${out} --batch_size ${batch_size} --learning_rate ${learning_rate} --chromesz ${chromesz} --loss_type ${loss} --mask_probes --shift_range --model ${model} --theta ${theta} --epochs ${epochs}

# model="original"
# date=$(date '+%d_%m_%Y_%H_%M_%S');
# experiment="${date}_experiment_${loss}_${model}_theta_${theta}_ep_${epochs}_bs_${batch_size}_chrome_${chromesz}_no_crop"
# out="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/${experiment}"
# srun python train.py -p ${path} -o ${out} --batch_size ${batch_size} --learning_rate ${learning_rate} --chromesz ${chromesz} --loss_type ${loss} --mask_probes --shift_range --model ${model} --theta ${theta} --epochs ${epochs}

# # l1 loss, croppping, masking probes and shift range
# chromesz=256
# model="vgg"
# date=$(date '+%d_%m_%Y_%H_%M_%S');
# experiment="${date}_experiment_${loss}_${model}_theta_${theta}_ep_${epochs}_bs_${batch_size}_chrome_${chromesz}_no_crop"
# out="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/${experiment}"
# srun python train.py -p ${path} -o ${out} --batch_size ${batch_size} --learning_rate ${learning_rate} --chromesz ${chromesz} --loss_type ${loss} --mask_probes --shift_range --model ${model} --theta ${theta} --epochs ${epochs}

# model="original"
# date=$(date '+%d_%m_%Y_%H_%M_%S');
# experiment="${date}_experiment_${loss}_${model}_theta_${theta}_ep_${epochs}_bs_${batch_size}_chrome_${chromesz}_no_crop"
# out="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/${experiment}"
# srun python train.py -p ${path} -o ${out} --batch_size ${batch_size} --learning_rate ${learning_rate} --chromesz ${chromesz} --loss_type ${loss} --mask_probes --shift_range --model ${model} --theta ${theta} --epochs ${epochs}

# l1 loss, croppping, masking probes and shift range
# model="VGG16"
# loss="l1"
# theta=0.1
# date=$(date '+%d_%m_%Y_%H_%M_%S');
# experiment="${date}_experiment_${loss}_${model}_theta_${theta}_swap_channels"
# out="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/${experiment}"
# srun python train.py -p ${path} -o ${out} --batch_size ${batch_size} --learning_rate ${learning_rate} --chromesz ${chromesz} --loss_type ${loss} --crop_images --mask_probes --shift_range --model ${model} --theta ${theta} --swap_channels
