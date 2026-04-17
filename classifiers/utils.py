import logging
import numpy as np
import torch

NOISE_LEVELS = torch.tensor(np.geomspace(0.01, 0.1, num=7))
NORM_STATS = torch.load("./data/norm_stats.pt", weights_only=False)
IMG_MEAN = NORM_STATS["img_mean"].view(3, 1, 1)
IMG_STD = NORM_STATS["img_std"].view(3, 1, 1)
SCORE_MEANS = [m.view(3, 1, 1) for m in NORM_STATS["score_means"]]
SCORE_STDS = [s.view(3, 1, 1) for s in NORM_STATS["score_stds"]]


class DataImage:
    def __init__(self, data_sample, attack_type=None):
        super().__init__()
        self.task = "img_classify"
        self.variant = "vanilla"
        self.attack_type = attack_type
        self.orig_image = data_sample["image"]
        self.orig_label = data_sample["label"]
        self.orig_scores = data_sample["scores"]

    def set_values(self, task, variant, scores_to_use=[]):
        self.task = task
        self.variant = variant
        self.scores_to_use = scores_to_use
        self.image = self.orig_image
        if self.variant == "scores":
            img = [[self.image[0]], [self.image[1]], [self.image[2]]]
            for s in self.scores_to_use:
                noise_idx = int(s.split("score")[-1])
                score = (
                    self.image + NOISE_LEVELS[noise_idx] * self.orig_scores[noise_idx]
                )
                img[0].append(score[0])
                img[1].append(score[1])
                img[2].append(score[2])
            self.image = torch.cat([torch.stack(img[i]) for i in range(3)], dim=0)

    @property
    def label(self):
        label = self.orig_label
        if self.task == "adv_detect":
            label = int(self.attack_type is not None)
        return label

    @property
    def is_adv(self):
        return int(self.attack_type is not None)


def setup_logger(exp_id, root_path, filename_params):
    logger = logging.getLogger(f"exp_{exp_id}_logger")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(
        f"{root_path}/training_{filename_params}_exp-{exp_id}.log"
    )
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(stream_handler)
    return logger


def normalize_data(data):
    all_imgs = []
    for key in data:
        all_imgs.extend([img.image for img in data[key]])
    all_imgs = torch.stack(all_imgs)
    num_samples, _, img_dim1, img_dim2 = all_imgs.shape
    mean = all_imgs.reshape(num_samples, 3, -1, img_dim1, img_dim2).mean((0, 2, 3, 4))
    std = all_imgs.reshape(num_samples, 3, -1, img_dim1, img_dim2).std((0, 2, 3, 4))
    for key in data:
        for i in range(len(data[key])):
            img_mean = mean.repeat_interleave(len(data[key][i].scores_to_use) + 1).reshape(-1, 1, 1)
            img_std = std.repeat_interleave(len(data[key][i].scores_to_use) + 1).reshape(-1, 1, 1)
            data[key][i].image = (data[key][i].image - img_mean) / (img_std + 1e-8)
    return data
