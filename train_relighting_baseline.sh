#!/bin/bash
#SBATCH --job-name=IllEst     	            # Name that will show up in squeue
#SBATCH --gres=gpu:1                        # Request 4 GPU "generic resource"
#SBATCH --time=7-00:00       		        # Max job time is 3 hours
#SBATCH --output=./logs/train_relit/%N-%j.out   	# Terminal output to file named (hostname)-(jobid).out
#SBATCH --error=./logs/train_relit/%N-%j.err		# Error log with format (hostname)-(jobid).out
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
batch_size=2
learning_rate=1e-4

# l1 loss, croppping, masking probes and shift range
# loss="l1"
# theta=1.0
# epochs=30
# date=$(date '+%d_%m_%Y_%H_%M_%S');
# experiment="${date}_relit_baseline_${loss}_theta_${theta}_ep_${epochs}_bs_${batch_size}_no_crop"
# out="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/${experiment}"
# srun python train_relighting_baseline.py -p ${path} -o ${out} --batch_size ${batch_size} --learning_rate ${learning_rate} --loss_type ${loss} --mask_probes --shift_range --theta ${theta} --epochs ${epochs}

# l1 loss, croppping, masking probes and shift range
loss="l1"
theta=1.0
epochs=30
date=$(date '+%d_%m_%Y_%H_%M_%S');
experiment="${date}_relit_baseline_${loss}_theta_${theta}_ep_${epochs}_bs_${batch_size}_no_crop_random_relight"
out="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/${experiment}"
srun python train_relighting_baseline.py -p ${path} -o ${out} --batch_size ${batch_size} --learning_rate ${learning_rate} --loss_type ${loss} --mask_probes --shift_range --theta ${theta} --epochs ${epochs} --random_relight