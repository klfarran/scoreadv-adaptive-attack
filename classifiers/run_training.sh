#!/bin/bash
#SBATCH --job-name=classifier_A
#SBATCH --output=logs/job_A.out
#SBATCH --error=error/job_A.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -A aisecurity
#SBATCH --time=0-16:00
#SBATCH --partition=gpu

mkdir -p logs
mkdir -p error

source /scratch/wds8wd/train_vemv/bin/activate

python -u main.py \
  -bs 256 \
  -ne 400 \
  -nex 5 \
  -lr 0.01 \
  -wd 1e-3 \
  -nrm 1 \
  -es 50 \
  -w 12 \
  -t adv_detect \
  -sch 0 \
  -v vanilla