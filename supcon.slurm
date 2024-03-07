#!/bin/bash

## ENVIRONMENT SETTINGS
#SBATCH --get-user-env=L          # Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=supcon
#SBATCH --time=10:00:00
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --output=slurm_log/%x.%j
#SBATCH --gres=gpu:a100:1             # Request 1 GPUs per node
#SBATCH --partition=gpu          # Submit job to the gpu queue

# SBATCH --gres=gpu:a100:1
# SBATCH --gres=gpu:rtx:1 
# SBATCH --gres=gpu:t4:1

## execute python script
python main_supcon.py --batch_size 256   --epochs 1   --learning_rate 1e-6   --weight_decay 1e-2   --dataset semi-aves   --data_folder ../CLIP-SSL/data/semi-aves/   --train_split fewshot15+real_t2t500.txt   --size 224   --temp 0.1   --cosine   --warm

# Job Environment variables
echo "Job ID: $SLURM_JOBID"
echo "Job submission directory: $SLURM_SUBMIT_DIR"
echo "Temp directory: $TMPDIR"
echo "Scratch directory: $SCRATCH"