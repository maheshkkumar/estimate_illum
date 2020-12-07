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

path="./data"
batch_size=1
learning_rate=1e-4
chromesz=64

# l1 loss, croppping, masking probes and shift range
out="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/02_12_2020_21_02_30_relit_experiment_l1_unet_theta_1.0_ep_30_bs_2_chrome_64_no_crop/models/qual_v2"
weights="/project/aksoy-lab/datasets/Multi_Illum_Invariance/CMPT726/experiments/02_12_2020_21_02_30_relit_experiment_l1_unet_theta_1.0_ep_30_bs_2_chrome_64_no_crop/models/epoch_30_pmae_0.26_prmse_0.09_ppsnr_10.46_pssim_0.28_imae_0.08_irmse_0.02_ipsnr_18.81_issim_0.86.pth"
srun python eval_relighting.py -o ${out} --batch_size ${batch_size} --mask_probes --shift_range --weights ${weights} --reference_probe