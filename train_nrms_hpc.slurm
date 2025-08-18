#!/bin/bash
#SBATCH --account=p32491  ## YOUR ACCOUNT pXXXX or bXXXX
#SBATCH --partition=gengpu  ### PARTITION (buyin, short, normal, etc)
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1 ## how many computers do you need
#SBATCH --ntasks-per-node=8 ## how many cpus or processors do you need on each computer
#SBATCH --time=48:00:00 ## how long does this need to run (remember different partitions have restrictions on this parameter)
#SBATCH --mem-per-cpu=4G ## how much RAM do you need per node (this effects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=nrms-mind-training  ## When you run squeue -u NETID this is how you can identify the job
#SBATCH --mail-user=nicholas.hagar@northwestern.edu
#SBATCH --mail-type=ALL

module purge all
module load cuda/12.4.1-gcc-12.3.0
module load cudnn

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export POPROX_REC_DEVICE="cuda"

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting NRMS model training at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

echo "CUDA available: $(uv run python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA device count: $(uv run python -c 'import torch; print(torch.cuda.device_count())')"

# Run the training
cd models
uv run python -m poprox_recommender.training.train

echo "Training completed at $(date)"
