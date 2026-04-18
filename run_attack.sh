#!/bin/bash
#SBATCH --job-name=adaptive_attack
#SBATCH --output=logs/adaptive_attack.out
#SBATCH --error=logs/adaptive_attack.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-08:00
#SBATCH --partition=gpu
#SBATCH -A aisecurity

mkdir -p logs
mkdir -p error

source /scratch/wds8wd/train_vemv/bin/activate


python -u adaptive_attack.py