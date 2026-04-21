#!/bin/bash
#SBATCH --job-name=adaptive_attack
#SBATCH --output=logs/adaptive_attack.out
#SBATCH --error=logs/adaptive_attack.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --time=0-16:00
#SBATCH --partition=gpu
#SBATCH -A aisecurity

mkdir -p logs

module load gcc/11.4.0
module load cuda/11.8

export TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6;9.0"
export TORCH_EXTENSIONS_DIR=/sfs/weka/scratch/wds8wd/torch_extensions

rm -rf $TORCH_EXTENSIONS_DIR
mkdir -p $TORCH_EXTENSIONS_DIR

source /scratch/wds8wd/attk-venv/bin/activate

#pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 https://download.pytorch.org/whl/cu118

python -u adaptive_attack.py
