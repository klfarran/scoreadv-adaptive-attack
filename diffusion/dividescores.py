"""
Split large training data files into smaller chunks for GitHub.
GitHub has a 100MB file size limit, so we'll split files larger than 85MB.
Copies all files (split or not) to maintain consistent structure.
"""

import os
import gzip
import torch
import shutil
from pathlib import Path

# ========================================
# CONFIGURATION
# ========================================

base_dir = "./data_split_scores/"
output_base_dir = "./data_score_splits/"

# GitHub file size limit is 100MB, we'll use 85MB to be safe
MAX_FILE_SIZE_MB = 80

# Process all three splits
SPLITS = ["train", "val", "test"]
ATTACK_TYPES = ["pgd", "fgsm", "cw", "autoattack"]

# ========================================
# HELPER FUNCTIONS
# ========================================

def get_file_size_mb(filepath):
    """Get file size in MB."""
    return os.path.getsize(filepath) / (1024 * 1024)

def split_data_list(data_list, chunk_size):
    """Split a list into chunks of specified size."""
    chunks = []
    for i in range(0, len(data_list), chunk_size):
        chunks.append(data_list[i:i + chunk_size])
    return chunks

def process_file(input_path, output_dir, max_size_mb):
    """
    Process a single file - either copy it or split it.
    
    Args:
        input_path: Path to input file
        output_dir: Directory for output files
        max_size_mb: Maximum file size in MB
    """
    print(f"\n  Processing {input_path}...")
    
    # Check if file exists
    if not os.path.exists(input_path):
        print(f"  ⚠️  File not found: {input_path}")
        return
    
    # Check file size
    file_size_mb = get_file_size_mb(input_path)
    print(f"  Current size: {file_size_mb:.1f} MB")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if file_size_mb <= max_size_mb:
        print(f"  ✓ File is under {max_size_mb}MB, copying as-is...")
        output_path = os.path.join(output_dir, "images_with_scores.pt.gz")
        shutil.copy2(input_path, output_path)
        print(f"  ✓ Copied to {output_path}")
        return
    
    # File is too big, need to split
    print(f"  File is too large, splitting...")
    
    # Load data
    print(f"  Loading data...")
    with gzip.open(input_path, "rb") as f:
        data = torch.load(f)
    
    print(f"  Loaded {len(data)} samples")
    
    # Estimate samples per chunk based on file size (with safety margin)
    samples_per_mb = len(data) / file_size_mb
    target_samples_per_chunk = int(samples_per_mb * max_size_mb * 0.85)
    
    print(f"  Estimated ~{target_samples_per_chunk} samples per chunk")
    
    # Split data
    chunks = split_data_list(data, target_samples_per_chunk)
    print(f"  Created {len(chunks)} chunks")
    
    # Save chunks
    for i, chunk in enumerate(chunks):
        output_path = os.path.join(output_dir, f"images_with_scores_part{i+1}.pt.gz")
        print(f"  Saving part {i+1}/{len(chunks)} ({len(chunk)} samples)...", end=" ")
        
        with gzip.open(output_path, "wb") as f:
            torch.save(chunk, f)
        
        chunk_size_mb = get_file_size_mb(output_path)
        
        if chunk_size_mb > 100:
            print(f"{chunk_size_mb:.1f} MB ⚠️  WARNING: Over 100MB!")
        else:
            print(f"{chunk_size_mb:.1f} MB ✓")
    
    print(f"  ✓ Split complete!")

# ========================================
# MAIN PROCESSING
# ========================================

if __name__ == "__main__":
    print("="*60)
    print("SPLITTING LARGE FILES FOR GITHUB")
    print("="*60)
    print(f"Input directory: {base_dir}")
    print(f"Output directory: {output_base_dir}")
    print(f"Max file size: {MAX_FILE_SIZE_MB} MB")
    print()
    
    for split_name in SPLITS:
        print("\n" + "="*60)
        print(f"Processing {split_name.upper()} split...")
        print("="*60)
        
        input_split_dir = os.path.join(base_dir, split_name)
        output_split_dir = os.path.join(output_base_dir, split_name)
        
        # Process non-adversarial data
        print(f"\n  --- Non-adversarial ---")
        non_adv_path = os.path.join(input_split_dir, "non_adv", "images_with_scores.pt.gz")
        output_non_adv_dir = os.path.join(output_split_dir, "non_adv")
        process_file(non_adv_path, output_non_adv_dir, MAX_FILE_SIZE_MB)
        
        # Process adversarial data
        for attack in ATTACK_TYPES:
            print(f"\n  --- {attack.upper()} ---")
            adv_path = os.path.join(input_split_dir, "adv", attack, "images_with_scores.pt.gz")
            output_adv_dir = os.path.join(output_split_dir, "adv", attack)
            process_file(adv_path, output_adv_dir, MAX_FILE_SIZE_MB)
    
    # Summary
    print("\n\n" + "="*60)
    print("✓ SPLITTING COMPLETE!")
    print("="*60)
    print(f"\nOutput directory: {output_base_dir}")
    print("\nDirectory structure:")
    print("  data_score_splits/")
    print("  ├── train/")
    print("  │   ├── non_adv/")
    print("  │   │   ├── images_with_scores_part1.pt.gz")
    print("  │   │   ├── images_with_scores_part2.pt.gz")
    print("  │   │   └── ...")
    print("  │   └── adv/")
    print("  │       ├── pgd/images_with_scores_part*.pt.gz")
    print("  │       ├── fgsm/images_with_scores_part*.pt.gz")
    print("  │       ├── cw/images_with_scores_part*.pt.gz")
    print("  │       └── autoattack/images_with_scores_part*.pt.gz")
    print("  ├── val/")
    print("  │   ├── non_adv/images_with_scores_part*.pt.gz")
    print("  │   └── adv/")
    print("  │       ├── pgd/images_with_scores.pt.gz (single file)")
    print("  │       ├── fgsm/images_with_scores.pt.gz")
    print("  │       ├── cw/images_with_scores.pt.gz")
    print("  │       └── autoattack/images_with_scores.pt.gz")
    print("  └── test/ (same as val)")
    print("\n✓ All files copied/split to maintain consistent structure")
    print("✓ Files under 85MB are copied as single files")
    print("✓ Files over 85MB are split into parts")