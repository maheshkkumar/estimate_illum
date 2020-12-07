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
#SBATCH --nodelist=cs-venus-02

# The SBATCH directives above set options similarly to command line arguments to srun
# The job will appear on squeue and output will be written to the current directory
# You can do tail -f <output_filename> to track the job.
# You can kill the job using scancel <job_id> where you can find the <job_id> from squeue

# Your experiment setup logic here
source ~/miniconda3/etc/profile.d/conda.sh
conda activate python3
echo "Environment activated"

batch_size=1
chromesz=64

# l1 loss, croppping, masking probes and shift range
model="original"
out="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/02_12_2020_23_04_47_experiment_l1_original_theta_1.0_ep_40_bs_8_chrome_64_no_crop/models/qual"
weights="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/02_12_2020_23_04_47_experiment_l1_original_theta_1.0_ep_40_bs_8_chrome_64_no_crop/models/epoch_32_mae_0.10_rmse_0.14_psnr_17.16_ssim_0.45.pth"
srun python eval.py -o ${out} --batch_size ${batch_size} --mask_probes --shift_range --weights ${weights} --model ${model}

# l1 loss, croppping, masking probes and shift range
model="vgg"
out="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/30_11_2020_22_35_14_experiment_l1_vgg_theta_1.0_ep_40_bs_8_chrome_64_no_crop/models/qual"
weights="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/30_11_2020_22_35_14_experiment_l1_vgg_theta_1.0_ep_40_bs_8_chrome_64_no_crop/models/epoch_37_mae_0.07_rmse_0.11_psnr_19.54_ssim_0.56.pth"
srun python eval.py -o ${out} --batch_size ${batch_size} --mask_probes --shift_range --weights ${weights} --model ${model}

# l1 loss, croppping, masking probes and shift range
model="resnet18"
out="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/30_11_2020_22_38_10_experiment_l1_resnet18_theta_1.0_ep_40_bs_8_chrome_64_no_crop/models/qual"
weights="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/30_11_2020_22_38_10_experiment_l1_resnet18_theta_1.0_ep_40_bs_8_chrome_64_no_crop/models/epoch_38_mae_0.07_rmse_0.11_psnr_19.31_ssim_0.55.pth"
srun python eval.py -o ${out} --batch_size ${batch_size} --mask_probes --shift_range --weights ${weights} --model ${model}

# l1 loss, croppping, masking probes and shift range
model="unet"
out="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/01_12_2020_09_49_22_experiment_l1_unet_theta_1.0_ep_40_bs_8_chrome_64_no_crop/models/qual"
weights="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/01_12_2020_09_49_22_experiment_l1_unet_theta_1.0_ep_40_bs_8_chrome_64_no_crop/models/epoch_38_mae_0.07_rmse_0.11_psnr_19.59_ssim_0.57.pth"
srun python eval.py -o ${out} --batch_size ${batch_size} --mask_probes --shift_range --weights ${weights} --model ${model}