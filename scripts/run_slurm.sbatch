#!/bin/bash -l

#SBATCH --mail-type=FAIL
#SBATCH --mail-user=blauch@cmu.edu
#SBATCH -p gpu
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=25G
#SBATCH --exclude=mind-1-23,mind-1-26

cd /home/nblauch/git/familiarity_sims
conda activate familiarity

echo "$COMMAND"
$COMMAND
