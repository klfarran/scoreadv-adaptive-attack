import gzip
import os
import torch
import time
import random

ATTACK_TYPES = ["pgd", "fgsm", "cw", "autoattack"]


def load_data_greedy(
    datasets_dir="./datasets/",
    target_counts=None,
    seed=42
):
    """
    Load data with fixed target counts using simple greedy allocation.
    
    Fast and simple - works well when you have plenty of data.
    
    Args:
        datasets_dir: Path to directory containing .pt.gz files
        target_counts: Dict like {
            "train": 6250,  # 6,250 per attack = 25,000 adversarial total
            "val": 625,     # 625 per attack = 2,500 adversarial total
            "test": 625     # 625 per attack = 2,500 adversarial total
        }
        seed: Random seed for reproducibility
    
    Returns:
        filtered_data: Dict with "train", "val", "test"
                      Each value is a dict with structure:
                      {
                          "non_adv": [(img, label), ...],
                          "adv": {
                              "pgd": [(img, label), ...],
                              "fgsm": [(img, label), ...],
                              "cw": [(img, label), ...],
                              "autoattack": [(img, label), ...]
                          }
                      }
    """
    random.seed(seed)
    
    if target_counts is None:
        target_counts = {}
    
    filtered_data = {}
    
    for data_name in ["train", "val", "test"]:
        print(f"\n{'='*60}")
        print(f"Processing {data_name.upper()} split...")
        print(f"{'='*60}")
        start_time = time.time()
        
        # Load all files for this split
        print(f"  Loading files from {datasets_dir}...")
        data = []
        files_to_load = [f for f in os.listdir(datasets_dir) 
                         if data_name in f and ".pt.gz" in f]
        files_to_load.sort()
        
        print(f"  Found {len(files_to_load)} files to load")
        
        for idx, f in enumerate(files_to_load, 1):
            filepath = os.path.join(datasets_dir, f)
            print(f"    Loading file {idx}/{len(files_to_load)}: {f}...", end=" ", flush=True)
            file_start = time.time()
            
            with gzip.open(filepath, "rb") as file:
                chunk = torch.load(file)
                data.extend(chunk)
            
            file_time = time.time() - file_start
            print(f"✓ ({len(chunk)} samples, {file_time:.1f}s)")
        
        print(f"  Total samples loaded: {len(data)}")
        
        # Filter: only samples with at least one successful attack
        valid_samples = [d for d in data if len(d.keys()) > 1]
        print(f"  Samples with successful attacks: {len(valid_samples)}")
        print(f"  Samples excluded (no attacks): {len(data) - len(valid_samples)}")
        print(f"  → Using {len(valid_samples)} samples for allocation")
        
        # Determine target per attack
        if data_name in target_counts:
            target = target_counts[data_name]
            print(f"  Target per attack (user-specified): {target}")
        else:
            # Use maximum available (bottleneck)
            print(f"  Computing maximum available...")
            attack_availability = {a: 0 for a in ATTACK_TYPES}
            for sample in valid_samples:
                for attack in ATTACK_TYPES:
                    if attack in sample:
                        attack_availability[attack] += 1
            target = min(attack_availability.values())
            print(f"  Target per attack (maximum): {target}")
        
        # Greedy allocation
        result = allocate_greedy(valid_samples, target)
        
        filtered_data[data_name] = result
        
        elapsed = time.time() - start_time
        print(f"\n  ✓ Completed in {elapsed:.1f}s")
        print(f"  Non-adversarial images: {len(result['non_adv'])}")
        print(f"  Adversarial images per attack:")
        for attack in sorted(ATTACK_TYPES):
            print(f"    {attack}: {len(result['adv'][attack])}")
        total_adv = sum(len(result['adv'][a]) for a in ATTACK_TYPES)
        print(f"  Total adversarial: {total_adv}")
        print(f"  Total images: {len(result['non_adv']) + total_adv}")
    
    return filtered_data


def allocate_greedy(samples, target):
    """
    Greedy allocation: iterate through samples and assign to least-filled attack.
    
    Simple, fast, and works well when you have plenty of data.
    
    Strategy:
    1. Shuffle samples for randomness
    2. For each sample, pick the attack with fewest allocations
    3. Stop when all attacks reach target
    
    Returns:
        result: Dict with structure:
            {
                "non_adv": [(img, label), ...],
                "adv": {
                    "pgd": [(img, label), ...],
                    "fgsm": [(img, label), ...],
                    "cw": [(img, label), ...],
                    "autoattack": [(img, label), ...]
                }
            }
    """
    print(f"  Running greedy allocation (target={target} per attack)...")
    
    # Shuffle samples for randomness (helps with diversity)
    shuffled_samples = samples.copy()
    random.shuffle(shuffled_samples)
    
    non_adv = []
    adv = {a: [] for a in ATTACK_TYPES}
    attack_counts = {a: 0 for a in ATTACK_TYPES}
    
    for sample in shuffled_samples:
        # Find which attacks this sample has
        available_attacks = [a for a in ATTACK_TYPES 
                            if a in sample and attack_counts[a] < target]
        
        if not available_attacks:
            continue  # Sample has no attacks we need
        
        # Pick the attack with fewest allocations (greedy)
        chosen_attack = min(available_attacks, key=lambda a: attack_counts[a])
        
        # Allocate to chosen attack
        non_adv.append(sample["original"])
        adv[chosen_attack].append(sample[chosen_attack])
        attack_counts[chosen_attack] += 1
        
        # Check if we're done
        if all(attack_counts[a] >= target for a in ATTACK_TYPES):
            print(f"    ✓ Reached target of {target} per attack")
            break
    
    # Verify we achieved target
    achieved = all(attack_counts[a] >= target for a in ATTACK_TYPES)
    if not achieved:
        print(f"    WARNING: Could not achieve target={target} for all attacks")
        print(f"    Achieved: {attack_counts}")
    
    print(f"    Final counts: {attack_counts}")
    print(f"    Unique originals used: {len(non_adv)}")
    
    result = {
        "non_adv": non_adv,
        "adv": adv
    }
    
    return result


def save_filtered_data(filtered_data, output_dir="./data_split/", overwrite=False):
    """
    Save the filtered data in organized directory structure.
    
    Args:
        filtered_data: Dict with "train", "val", "test"
        output_dir: Directory to save to
        overwrite: If True, skip confirmation prompt
    
    Directory structure:
        data_split/
        ├���─ train/
        │   ├── non_adv/
        │   │   └── images.pt.gz
        │   └── adv/
        │       ├── pgd/
        │       │   └── images.pt.gz
        │       ├── fgsm/
        │       │   └── images.pt.gz
        │       ├── cw/
        │       │   └── images.pt.gz
        │       └── autoattack/
        │           └── images.pt.gz
        ├── val/
        │   └── ... (same structure)
        └── test/
            └── ... (same structure)
    """
    # Check if data already exists
    if os.path.exists(output_dir) and not overwrite:
        print(f"\n{'='*60}")
        print(f"⚠️  WARNING: EXISTING DATA FOUND")
        print(f"{'='*60}")
        print(f"Output directory already exists: {output_dir}")
        print(f"Continuing will OVERWRITE existing data!")
        
        # List existing files
        print(f"\nExisting files:")
        total_size = 0
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(output_dir, split)
            if os.path.exists(split_dir):
                print(f"  {split}/")
                non_adv_path = os.path.join(split_dir, "non_adv", "images.pt.gz")
                if os.path.exists(non_adv_path):
                    size = os.path.getsize(non_adv_path) / (1024 * 1024)
                    total_size += size
                    print(f"    non_adv/images.pt.gz ({size:.1f} MB)")
                for attack in ATTACK_TYPES:
                    attack_path = os.path.join(split_dir, "adv", attack, "images.pt.gz")
                    if os.path.exists(attack_path):
                        size = os.path.getsize(attack_path) / (1024 * 1024)
                        total_size += size
                        print(f"    adv/{attack}/images.pt.gz ({size:.1f} MB)")
        
        print(f"\nTotal size: {total_size:.1f} MB")
        print()
        
        while True:
            response = input("Continue and overwrite? (yes/no): ").strip().lower()
            if response in ["yes", "y"]:
                print("Proceeding with overwrite...")
                break
            elif response in ["no", "n"]:
                print("\n❌ Aborted. No files were modified.")
                print("Tip: Use a different output_dir or set overwrite=True")
                return
            else:
                print("Please answer 'yes' or 'no'")
    
    print(f"\n{'='*60}")
    print(f"SAVING FILTERED DATA")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Structure: <split>/non_adv and <split>/adv/<attack_type>")
    print()
    
    for split_name, data in filtered_data.items():
        print(f"  Saving {split_name.upper()}...")
        
        # Create directories
        split_dir = os.path.join(output_dir, split_name)
        non_adv_dir = os.path.join(split_dir, "non_adv")
        adv_dir = os.path.join(split_dir, "adv")
        
        os.makedirs(non_adv_dir, exist_ok=True)
        os.makedirs(adv_dir, exist_ok=True)
        
        # Save non-adversarial images
        non_adv_path = os.path.join(non_adv_dir, "images.pt.gz")
        print(f"    Saving non-adversarial to {non_adv_path}...", end=" ", flush=True)
        start_time = time.time()
        with gzip.open(non_adv_path, "wb") as f:
            torch.save(data["non_adv"], f)
        elapsed = time.time() - start_time
        file_size_mb = os.path.getsize(non_adv_path) / (1024 * 1024)
        print(f"✓ ({len(data['non_adv'])} images, {file_size_mb:.1f} MB, {elapsed:.1f}s)")
        
        # Save adversarial images by attack type
        for attack in ATTACK_TYPES:
            attack_dir = os.path.join(adv_dir, attack)
            os.makedirs(attack_dir, exist_ok=True)
            
            attack_path = os.path.join(attack_dir, "images.pt.gz")
            print(f"    Saving {attack} to {attack_path}...", end=" ", flush=True)
            start_time = time.time()
            with gzip.open(attack_path, "wb") as f:
                torch.save(data["adv"][attack], f)
            elapsed = time.time() - start_time
            file_size_mb = os.path.getsize(attack_path) / (1024 * 1024)
            print(f"✓ ({len(data['adv'][attack])} images, {file_size_mb:.1f} MB, {elapsed:.1f}s)")
        
        print()
    
    print(f"✓ All filtered data saved to {output_dir}")


def load_filtered_data(input_dir="./data_split/"):
    """
    Load pre-filtered data from organized directory structure.
    
    Returns:
        filtered_data: Dict with "train", "val", "test"
                      Each is a dict with "non_adv" and "adv" keys
    """
    filtered_data = {}
    
    print(f"\n{'='*60}")
    print(f"LOADING FILTERED DATA FROM DISK")
    print(f"{'='*60}")
    
    for split_name in ["train", "val", "test"]:
        print(f"\n  Loading {split_name.upper()}...")
        
        split_dir = os.path.join(input_dir, split_name)
        
        # Load non-adversarial
        non_adv_path = os.path.join(split_dir, "non_adv", "images.pt.gz")
        print(f"    Loading non-adversarial from {non_adv_path}...", end=" ", flush=True)
        start_time = time.time()
        with gzip.open(non_adv_path, "rb") as f:
            non_adv = torch.load(f)
        elapsed = time.time() - start_time
        print(f"✓ ({len(non_adv)} images, {elapsed:.1f}s)")
        
        # Load adversarial by attack type
        adv = {}
        for attack in ATTACK_TYPES:
            attack_path = os.path.join(split_dir, "adv", attack, "images.pt.gz")
            print(f"    Loading {attack} from {attack_path}...", end=" ", flush=True)
            start_time = time.time()
            with gzip.open(attack_path, "rb") as f:
                adv[attack] = torch.load(f)
            elapsed = time.time() - start_time
            print(f"✓ ({len(adv[attack])} images, {elapsed:.1f}s)")
        
        filtered_data[split_name] = {
            "non_adv": non_adv,
            "adv": adv
        }
    
    print(f"\n✓ All filtered data loaded from {input_dir}")
    
    return filtered_data


def analyze_distribution(filtered_data):
    """Analyze class distribution in filtered data, including per-attack breakdowns."""
    print("\n" + "="*60)
    print("DISTRIBUTION ANALYSIS")
    print("="*60)
    
    from collections import Counter
    
    for split_name, data in filtered_data.items():
        print(f"\n{split_name.upper()}:")
        print("="*60)
        
        # Extract labels from non-adversarial images
        original_labels = [img[1] for img in data["non_adv"]]
        label_counts = Counter(original_labels)
        
        print(f"\n  NON-ADVERSARIAL IMAGES ({len(original_labels)} total):")
        print(f"  {'Class':<8} {'Count':<8} {'Percentage':<12}")
        print(f"  {'-'*28}")
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            pct = 100 * count / len(original_labels)
            print(f"  {label:<8} {count:<8} {pct:>6.1f}%")
        
        # Check uniformity
        expected = len(original_labels) / 10  # CIFAR-10 has 10 classes
        max_deviation = max(abs(label_counts.get(i, 0) - expected) for i in range(10))
        uniformity = 100 * (1 - max_deviation / expected) if expected > 0 else 0
        print(f"  {'-'*28}")
        print(f"  Distribution uniformity: {uniformity:.1f}%")
        
        # Analyze each attack type
        for attack in sorted(ATTACK_TYPES):
            adv_labels = [img[1] for img in data["adv"][attack]]
            adv_counts = Counter(adv_labels)
            
            print(f"\n  {attack.upper()} ADVERSARIAL ({len(adv_labels)} total):")
            print(f"  {'Class':<8} {'Count':<8} {'Percentage':<12} {'vs Expected':<12}")
            print(f"  {'-'*40}")
            
            expected_per_class = len(adv_labels) / 10
            for label in sorted(range(10)):
                count = adv_counts.get(label, 0)
                pct = 100 * count / len(adv_labels) if len(adv_labels) > 0 else 0
                deviation = count - expected_per_class
                print(f"  {label:<8} {count:<8} {pct:>6.1f}%      {deviation:>+6.1f}")
            
            # Calculate uniformity for this attack
            max_dev_attack = max(abs(adv_counts.get(i, 0) - expected_per_class) for i in range(10))
            uniformity_attack = 100 * (1 - max_dev_attack / expected_per_class) if expected_per_class > 0 else 0
            print(f"  {'-'*40}")
            print(f"  Distribution uniformity: {uniformity_attack:.1f}%")


def get_data_stats(datasets_dir="./datasets/"):
    """Print statistics about the saved adversarial data."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    for data_name in ["train", "val", "test"]:
        print(f"\n{data_name.upper()}:")
        
        files_to_load = [f for f in os.listdir(datasets_dir) 
                         if data_name in f and ".pt.gz" in f]
        files_to_load.sort()
        
        data = []
        for f in files_to_load:
            filepath = os.path.join(datasets_dir, f)
            with gzip.open(filepath, "rb") as file:
                data.extend(torch.load(file))
        
        print(f"  Total samples: {len(data)}")
        
        valid_samples = [d for d in data if len(d.keys()) > 1]
        print(f"  Samples with attacks: {len(valid_samples)}")
        
        # Attack availability
        attack_availability = {a: 0 for a in ATTACK_TYPES}
        for sample in data:
            for attack in ATTACK_TYPES:
                if attack in sample:
                    attack_availability[attack] += 1
        
        print(f"  Attack availability:")
        for attack in sorted(ATTACK_TYPES):
            pct = 100 * attack_availability[attack] / len(data) if len(data) > 0 else 0
            print(f"    {attack}: {attack_availability[attack]} ({pct:.1f}%)")
        
        if attack_availability:
            bottleneck = min(attack_availability, key=attack_availability.get)
            print(f"  Bottleneck: {bottleneck} ({attack_availability[bottleneck]} samples)")


if __name__ == "__main__":
    # Show stats
    get_data_stats()
    
    # Load with greedy approach (fast!)
    print("\n\n" + "="*60)
    print("LOADING DATA WITH GREEDY ALLOCATION")
    print("="*60)
    
    target_counts = {
        "train": 6250,   # 6,250 per attack = 25,000 adversarial (50,000 total)
        "val": 625,      # 625 per attack = 2,500 adversarial (5,000 total)
        "test": 625      # 625 per attack = 2,500 adversarial (5,000 total)
    }
    
    print(f"Target counts per attack:")
    for split, count in target_counts.items():
        total_adv = count * len(ATTACK_TYPES)
        total = total_adv * 2  # clean + adversarial
        print(f"  {split}: {count} per attack → {total_adv} adversarial + {total_adv} clean = {total} total")
    
    filtered_data = load_data_greedy(target_counts=target_counts)
    
    # Analyze distribution (now includes per-attack analysis!)
    analyze_distribution(filtered_data)
    
    # Save to organized directory structure
    save_filtered_data(filtered_data, output_dir="./data_split/")
    
    # Show final summary
    print("\n\n" + "="*60)
    print("FINAL DATASET SUMMARY")
    print("="*60)
    for split in ["train", "val", "test"]:
        data = filtered_data[split]
        n_clean = len(data["non_adv"])
        n_adv_total = sum(len(data["adv"][a]) for a in ATTACK_TYPES)
        
        print(f"\n{split.upper()}:")
        print(f"  Clean images: {n_clean}")
        print(f"  Adversarial images: {n_adv_total}")
        print(f"    Per attack type:")
        for attack in sorted(ATTACK_TYPES):
            print(f"      {attack}: {len(data['adv'][attack])}")
        print(f"  Total images: {n_clean + n_adv_total}")
        print(f"  Ratio: {100*n_clean/(n_clean+n_adv_total):.1f}% clean / {100*n_adv_total/(n_clean+n_adv_total):.1f}% adversarial")
    
    print("\n✓ Data saved to ./data_split/ with structure:")
    print("  data_split/")
    print("  ├── train/non_adv/images.pt.gz")
    print("  ├── train/adv/pgd/images.pt.gz")
    print("  ├── train/adv/fgsm/images.pt.gz")
    print("  ├── train/adv/cw/images.pt.gz")
    print("  ├── train/adv/autoattack/images.pt.gz")
    print("  └── (same for val and test)")