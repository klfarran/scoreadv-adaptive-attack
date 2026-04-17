import argparse
import gzip
import os
import random
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import torchattacks
import numpy as np
from tqdm import tqdm

ATTACK_TYPES = ["pgd", "fgsm", "cw", "autoattack"]
NUM_SAMPLES = {"train": 10000, "val": 1250, "test": 1250}


def load_data():
    data, filtered_data = {}, {}
    for data_name in ["train", "val", "test"]:
        data[data_name], filtered_data[data_name] = [], []
        for f in os.listdir("./datasets/"):
            if data_name not in f or ".pt.gz" not in f:
                continue
            with gzip.open(f"./datasets/{f}", "rb") as f:
                data[data_name].extend(torch.load(f))
        attack_count = {attack_type: 0 for attack_type in ATTACK_TYPES}
        filtered_data = {"train": [], "val": [], "test": []}
        for d in data[data_name]:
            if len(d.keys()) == 1:
                continue
            key = min([(attack_count[k], k) for k in d if k != "original"])[1]
            filtered_data[data_name].append(d["original"])
            filtered_data[data_name].append(d[key])
            attack_count[key] += 1
    return filtered_data


def generate_imgs(attack_type, model, train_data, val_data, test_data, batch_size):
    print("Generating attack type:", attack_type)
    if attack_type == "pgd":
        attack = torchattacks.PGD(model, eps=8 / 255, alpha=2 / 255, steps=10)
    elif attack_type == "fgsm":
        attack = torchattacks.FGSM(model, eps=8 / 255)
    elif attack_type == "cw":
        attack = torchattacks.CW(model, c=1, kappa=0, steps=50, lr=0.01)
    elif attack_type == "autoattack":
        attack = torchattacks.AutoAttack(
            model, norm="Linf", eps=8 / 255, version="standard", seed=42
        )
    new_data = {}
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        new_data[name] = []
        loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        for batch in tqdm(loader, desc=f"Running {name}"):
            adv_images = attack(batch[1], batch[2]).detach()
            adv_labels = model(adv_images).argmax(-1).cpu().tolist()
            new_data[name].extend(
                [
                    (batch[0][idx].item(), img.cpu(), adv_labels[idx])
                    for idx, img in enumerate(adv_images)
                    if adv_labels[idx] != batch[2][idx]
                ]
            )
    return new_data


def main(args):
    print("Loading model")
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True
    ).to(args.device)

    print("Loading CIFAR-10 dataset")
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_data = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_size, val_size = int(len(test_data) * 0.5), int(len(test_data) * 0.5)
    test_data, val_data = random_split(test_data, [test_size, val_size])

    train_data = [(idx, img[0], img[1]) for idx, img in enumerate(train_data)]
    val_data = [(idx, img[0], img[1]) for idx, img in enumerate(val_data)]
    test_data = [(idx, img[0], img[1]) for idx, img in enumerate(test_data)]

    new_data, final_data = {}, {}
    for attack_type in ATTACK_TYPES:
        new_data[attack_type] = generate_imgs(
            attack_type, model, train_data, val_data, test_data, args.batch_size
        )

    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        final_data[name] = {img[0]: {"original": img[1:]} for img in data}
        for attack_type in ATTACK_TYPES:
            for adv_sample in new_data[attack_type][name]:
                if adv_sample[0] not in final_data[name]:
                    final_data[name][adv_sample[0]] = {}
                final_data[name][adv_sample[0]][attack_type] = adv_sample[1:]
    for data_name in final_data:
        idx = 0
        while idx < len(final_data[data_name]):
            with gzip.open(f"./datasets/{data_name}_{idx//2500}.pt.gz", "wb") as f:
                torch.save(list(final_data[data_name].values())[idx : idx + 2500], f)
            idx += 2500


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Adversarial Data")
    parser.add_argument("--device", default=None, type=int)
    parser.add_argument("-bs", "--batch_size", default=2048, type=int)
    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if args.device else "cpu"

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    main(args)
