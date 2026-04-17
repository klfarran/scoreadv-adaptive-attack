import os
import torch

# Patch for CPU-only MUST come first
if not torch.cuda.is_available():
    import torch.utils.cpp_extension
    def _dummy_load(*args, **kwargs):
        class DummyModule:
            def __getattr__(self, name):
                return lambda *args, **kwargs: None
        return DummyModule()
    torch.utils.cpp_extension.load = _dummy_load

import numpy as np
from tqdm import tqdm
import gzip
from dataclasses import dataclass, field
from losses import get_optimizer
from models.ema import ExponentialMovingAverage
from utils import restore_checkpoint
import models
from models import utils as mutils
from models import ddpm, ncsnv2, ncsnpp
from sde_lib import VESDE
import datasets

# Load VE NCSN++ Deep Continuous model configuration
from configs.ve import cifar10_ncsnpp_deep_continuous as configs

# ========================================
# CONFIGURATION
# ========================================

# Checkpoint path
ckpt_filename = "exp/ve/checkpoint_12.pth"

# Input/output paths
input_dir = "./data_split/"
output_dir = "./data_split_scores/"

# Batch size for processing
batch_size = 128 if torch.cuda.is_available() else 32

# Number of noise levels
num_noise_levels = 7

# ========================================
# SETUP MODEL
# ========================================

print("="*60)
print("SCORE EXTRACTION SETUP")
print("="*60)

# Verify checkpoint exists
if not os.path.exists(ckpt_filename):
    raise FileNotFoundError(f"❌ Checkpoint not found: {ckpt_filename}")
print(f"✓ Found checkpoint at: {ckpt_filename}")

# Get configuration
config = configs.get_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.device = device
print(f"✓ Using device: {device}")

# Create SDE
sde = VESDE(sigma_min=config.model.sigma_min, 
            sigma_max=config.model.sigma_max, 
            N=config.model.num_scales)

# Setup data scalers
print("Setting up data scalers...")
sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)

# Create model
print("Creating score model...")
score_model = mutils.create_model(config)

# Setup optimizer and EMA
print("Setting up optimizer and EMA...")
optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)

# Load checkpoint
print(f"Loading checkpoint: {ckpt_filename}")
state = restore_checkpoint(ckpt_filename, state, config.device)
ema.copy_to(score_model.parameters())

score_model = score_model.to(device)

print("\n✓ Model loaded successfully!")
print(f"✓ Training step: {state['step']}")
print(f"✓ Device: {config.device}")
print(f"✓ Sigma range: [{config.model.sigma_min}, {config.model.sigma_max}]")
print("="*60 + "\n")

# Define 7 noise levels (geometric spacing)
noise_levels = np.geomspace(config.model.sigma_min, config.model.sigma_max, num=num_noise_levels)
print(f"Noise levels: {noise_levels}")
print()

# ========================================
# SCORE EXTRACTION FUNCTIONS
# ========================================

def extract_scores_from_images(images, labels, score_model, noise_levels, device, batch_size):
    """Extract scores for a list of images at multiple noise levels."""
    import time
    
    score_model.eval()
    
    scores_data = []
    num_images = len(images)
    
    # Process in batches
    with torch.no_grad():
        for start_idx in tqdm(range(0, num_images, batch_size), desc="  Processing batches"):
            batch_start = time.time()
            
            end_idx = min(start_idx + batch_size, num_images)
            batch_images = [images[i][0] for i in range(start_idx, end_idx)]
            batch_labels = [images[i][1] for i in range(start_idx, end_idx)]
            
            # Stack into batch and move to device
            batch_tensor = torch.stack(batch_images).to(device)
            
            # Extract scores at each noise level
            batch_scores = []
            for sigma_idx, sigma in enumerate(noise_levels):
                # Create sigma labels for the batch
                sigma_labels = torch.ones(batch_tensor.shape[0], device=device) * sigma
                
                # Compute score on images
                score = score_model(batch_tensor, sigma_labels)
                
                # Store score (keep on CPU to save memory)
                batch_scores.append(score.cpu())
            
            # Organize per-image
            for i in range(len(batch_images)):
                scores_data.append({
                    'image': batch_images[i].cpu(),  # ⭐ Ensure on CPU for storage
                    'label': batch_labels[i],
                    'scores': [batch_scores[sigma_idx][i] for sigma_idx in range(len(noise_levels))]
                })
            
            # Print timing for first batch
            if start_idx == 0:
                batch_time = time.time() - batch_start
                time_per_img = batch_time / len(batch_images)
                est_total_min = (time_per_img * num_images) / 60
                print(f"    First batch: {batch_time:.2f}s ({time_per_img:.2f}s/img)")
                print(f"    Estimated total: {est_total_min:.1f} minutes")
    
    return scores_data


def process_split(split_name, input_dir, output_dir, score_model, noise_levels, device, batch_size):
    """Process one split (train, val, or test)."""
    print(f"\n{'='*60}")
    print(f"Processing {split_name.upper()} split...")
    print(f"{'='*60}")
    
    split_input_dir = os.path.join(input_dir, split_name)
    split_output_dir = os.path.join(output_dir, split_name)
    
    # Process non-adversarial images
    print(f"\n  Loading non-adversarial images...")
    non_adv_path = os.path.join(split_input_dir, "non_adv", "images.pt.gz")
    with gzip.open(non_adv_path, "rb") as f:
        non_adv_images = torch.load(f)
    
    print(f"  ✓ Loaded {len(non_adv_images)} non-adversarial images")
    print(f"  Extracting scores for non-adversarial images...")
    
    non_adv_scores = extract_scores_from_images(
        non_adv_images, 
        [img[1] for img in non_adv_images],
        score_model, 
        noise_levels, 
        device, 
        batch_size
    )
    
    # Save non-adversarial with scores
    non_adv_output_dir = os.path.join(split_output_dir, "non_adv")
    os.makedirs(non_adv_output_dir, exist_ok=True)
    non_adv_output_path = os.path.join(non_adv_output_dir, "images_with_scores.pt.gz")
    
    print(f"  Saving to {non_adv_output_path}...")
    with gzip.open(non_adv_output_path, "wb") as f:
        torch.save(non_adv_scores, f)
    
    file_size_mb = os.path.getsize(non_adv_output_path) / (1024 * 1024)
    print(f"  ✓ Saved non-adversarial with scores ({file_size_mb:.1f} MB)")
    
    # Process each adversarial attack type
    attack_types = ["pgd", "fgsm", "cw", "autoattack"]
    
    for attack in attack_types:
        print(f"\n  Loading {attack} adversarial images...")
        adv_path = os.path.join(split_input_dir, "adv", attack, "images.pt.gz")
        
        with gzip.open(adv_path, "rb") as f:
            adv_images = torch.load(f)
        
        print(f"  ✓ Loaded {len(adv_images)} {attack} images")
        print(f"  Extracting scores for {attack} images...")
        
        adv_scores = extract_scores_from_images(
            adv_images,
            [img[1] for img in adv_images],
            score_model,
            noise_levels,
            device,
            batch_size
        )
        
        # Save adversarial with scores
        adv_output_dir = os.path.join(split_output_dir, "adv", attack)
        os.makedirs(adv_output_dir, exist_ok=True)
        adv_output_path = os.path.join(adv_output_dir, "images_with_scores.pt.gz")
        
        print(f"  Saving to {adv_output_path}...")
        with gzip.open(adv_output_path, "wb") as f:
            torch.save(adv_scores, f)
        
        file_size_mb = os.path.getsize(adv_output_path) / (1024 * 1024)
        print(f"  ✓ Saved {attack} with scores ({file_size_mb:.1f} MB)")


# ========================================
# MAIN PROCESSING
# ========================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("EXTRACTING SCORES FOR ADVERSARIAL DATASET")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Number of noise levels: {num_noise_levels}")
    print()
    
    # Process each split
    for split_name in ["train", "val", "test"]:
        process_split(
            split_name, 
            input_dir, 
            output_dir, 
            score_model, 
            noise_levels, 
            device, 
            batch_size
        )
    
    print("\n\n" + "="*60)
    print("✓ SCORE EXTRACTION COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")