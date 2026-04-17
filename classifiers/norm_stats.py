import torch
from tqdm import tqdm

data = torch.load("./data/all_data.pt", weights_only=False)
train_data = data["train"]
NUM_NOISE_LEVELS = 7

# Compute stats for image channels
all_images = torch.stack([s.orig_image for s in train_data])
img_mean = all_images.mean(dim=[0, 2, 3])
img_std = all_images.std(dim=[0, 2, 3])
print(f"Image mean: {img_mean}")
print(f"Image std: {img_std}")

# Compute stats for each score level
score_means = []
score_stds = []
for noise_idx in range(NUM_NOISE_LEVELS):
    all_scores = torch.stack([s.orig_scores[noise_idx] for s in tqdm(train_data, desc=f"Score {noise_idx}")])
    score_mean = all_scores.mean(dim=[0, 2, 3])
    score_std = all_scores.std(dim=[0, 2, 3])
    score_means.append(score_mean)
    score_stds.append(score_std)
    print(f"Score {noise_idx} mean: {score_mean}")
    print(f"Score {noise_idx} std: {score_std}")

stats = {
    "img_mean": img_mean,
    "img_std": img_std,
    "score_means": score_means,
    "score_stds": score_stds,
}
torch.save(stats, "./data/norm_stats.pt")
print("Saved stats to ./data/norm_stats.pt")