import argparse
import gzip
import os
import random
import fcntl
import csv
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from classifier import CifarClassifier
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DataImage, normalize_data, setup_logger
from sklearn.metrics import precision_score, recall_score, f1_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default=None, type=int)
    parser.add_argument("-bs", "--batch_size", default=128, type=int)
    parser.add_argument("-ne", "--num_epochs", default=100, type=int)
    parser.add_argument("-nex", "--num_exps", default=5, type=int)
    parser.add_argument("-lr", "--lr", default=1e-3, type=float)
    parser.add_argument("-wd", "--wd", default=1e-3, type=float)
    parser.add_argument("-nrm", "--normalize", default=1, type=int)
    parser.add_argument("-es", "--early_stop", default=20, type=int)
    parser.add_argument("-w", "--width", default=2, type=int)
    parser.add_argument("-sch", "--scheduler", default=0, type=int)
    parser.add_argument("-do", "--dropout", default=0.5, type=float)
    parser.add_argument("-fc", "--fixed_coeff", default=True, action="store_true")
    parser.add_argument(
        "-t",
        "--task",
        default="img_classify",
        type=str,
        choices=["img_classify", "adv_detect"],
    )
    # score0 corresponds to 1st noise level, score1 to 2nd noise level, ...
    parser.add_argument(
        "-v",
        "--variant",
        type=str,
        default=["vanilla"],
        choices=["vanilla", "scores"],
    )
    parser.add_argument(
        "-sc",
        "--scores",
        type=str,
        nargs='+',
        default=[],
        choices=[f"score{i}" for i in range(7)],
    )
    
    return parser.parse_args()


def collate_fn(batch):
    batched_vals = {"images": [], "labels": [], "is_adv": []}
    for sample in batch:
        batched_vals["images"].append(sample.image)
        batched_vals["labels"].append(sample.label)
        batched_vals["is_adv"].append(sample.is_adv)
    batched_vals["images"] = torch.stack(batched_vals["images"], dim=0)
    batched_vals["labels"] = torch.tensor(batched_vals["labels"]).long()
    batched_vals["is_adv"] = torch.tensor(batched_vals["is_adv"]).long()
    return batched_vals


def load_data(task, variant, scores, normalize):
    if not os.path.exists("./data/all_data.pt"):
        data = {"train": [], "val": [], "test": []}
        for data_set in data:
            for file in tqdm(
                os.listdir(f"./data/{data_set}/non_adv"), desc=f"Loading {data_set} set"
            ):
                with gzip.open(f"./data/{data_set}/non_adv/{file}", "rb") as f:
                    cur_samples = torch.load(f, weights_only=False)
                data[data_set].extend([DataImage(sample) for sample in cur_samples])
            torch.save(data, "./data/all_data.pt")

            for attack_type in tqdm(
                os.listdir(f"./data/{data_set}/adv"), desc=f"Loading {data_set} set"
            ):
                for file in os.listdir(f"./data/{data_set}/adv/{attack_type}"):
                    with gzip.open(
                        f"./data/{data_set}/adv/{attack_type}/{file}", "rb"
                    ) as f:
                        cur_samples = torch.load(f, weights_only=False)
                    data[data_set].extend(
                        [
                            DataImage(sample, attack_type=attack_type)
                            for sample in cur_samples
                        ]
                    )
            torch.save(data, "./data/all_data.pt")

    data = torch.load("./data/all_data.pt", weights_only=False)
    for data_set in data:
        for i in range(len(data[data_set])):
            data[data_set][i].set_values(task, variant, scores)
    if normalize:
        data = normalize_data(data)
    train_loader = DataLoader(
        data["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        data["val"], batch_size=args.batch_size, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        data["test"], batch_size=args.batch_size, collate_fn=collate_fn
    )
    return train_loader, val_loader, test_loader


def eval_classifier(classifier, loader, device):
    classifier.eval()
    eval_pred, eval_labels, eval_is_adv = [], [], []
    eval_loss, eval_acc = 0, 0
    with torch.no_grad():
        for batch in loader:
            pred = classifier(batch["images"].to(device))
            eval_loss += F.cross_entropy(pred, batch["labels"].to(device)).detach().cpu()
            eval_acc += (pred.argmax(-1).detach().cpu() == batch["labels"]).sum()
            eval_pred.append(pred.argmax(-1).detach().cpu())
            eval_labels.append(batch["labels"])
            eval_is_adv.append(batch["is_adv"])
    eval_pred = torch.concat(eval_pred)
    eval_labels = torch.concat(eval_labels)
    eval_is_adv = torch.concat(eval_is_adv)
    
    correct = (eval_pred == eval_labels)
    clean_acc = correct[eval_is_adv == 0].float().mean().item()
    adv_acc = correct[eval_is_adv == 1].float().mean().item()
    
    eval_acc = eval_acc / len(loader.dataset)
    eval_loss = eval_loss / len(loader)
    eval_recall = recall_score(eval_labels, eval_pred, average='weighted', zero_division=0)
    eval_precision = precision_score(eval_labels, eval_pred, average='weighted', zero_division=0)
    eval_f1 = f1_score(eval_labels, eval_pred, average='weighted', zero_division=0)
    return eval_loss, eval_acc, clean_acc, adv_acc, eval_recall, eval_precision, eval_f1


def main(args):
    best_final_eval_loss = 1e10
    best_train_accs, best_eval_accs, best_clean_accs, best_adv_accs = [], [], [], []
    args.root_path = f"./results/{args.task}/{args.variant}"
    scores_str = "-".join(args.scores) if args.scores else "none"
    args.filename_params = (
        f"epochs={args.num_epochs}_batch={args.batch_size}_lr={args.lr}_wd={args.wd}_scores={scores_str}_norm={args.normalize}_width={args.width}_sch={args.scheduler}_do={args.dropout}_fc={int(args.fixed_coeff)}"
    )
    os.makedirs(args.root_path, exist_ok=True)
    with open(f"{args.root_path}/results_{args.filename_params}.txt", "a+") as f:
        f.write(f"{datetime.now()}\nStart experiment\n")
    acc_list, recall_list, precision_list, f1_list = [], [], [], []
    clean_acc_list, adv_acc_list = [], []
    print("Loading data")
    train_loader, val_loader, test_loader = load_data(args.task, args.variant, args.scores, args.normalize)
    for exp in range(args.num_exps):
        train_accs, eval_accs, clean_accs, adv_accs = [], [], [], []
        logger = setup_logger(exp, args.root_path, args.filename_params)
        logger.info(f"Running experiment {exp} on device {args.device}")
        if args.variant == "scores":
            num_score_channels = len(args.scores) * 3
        else:
            num_score_channels = 0
        
        # Create classifier
        classifier = CifarClassifier(
            depth=28,                      # Standard depth
            widen_factor=args.width,       # NOW width is the widen_factor!
            num_score_channels=num_score_channels,
            num_classes=10,
            dropRate=args.dropout,         # Regularization
            scores=args.scores if args.fixed_coeff else None
        ).to(args.device)
        optimizer = torch.optim.SGD(
            classifier.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9, nesterov=True
        )
        if args.scheduler == 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=5,      
                T_mult=2,    
                eta_min=0 
            )

        best_eval_loss = 1e10
        best_eval_acc = 0
        logger.info("Start training")
        for epoch in range(args.num_epochs):
            classifier.train()
            train_loss, train_acc = 0, 0
            for batch in train_loader:
                optimizer.zero_grad()
                pred = classifier(batch["images"].to(args.device))
                loss = F.cross_entropy(pred, batch["labels"].to(args.device))
                train_loss += loss.detach().cpu()
                train_acc += (pred.argmax(-1).detach().cpu() == batch["labels"]).sum()
                loss.backward()
                optimizer.step()
            scheduler.step()
            train_acc = train_acc / len(train_loader.dataset)
            train_loss = train_loss / len(train_loader)
            log_line = f"Epoch {epoch}: train_loss = {train_loss.item()} | train_acc = {train_acc.item()}"

            eval_loss, eval_acc, clean_acc, adv_acc, eval_recall, eval_precision, eval_f1 = eval_classifier(
                classifier, val_loader, args.device
            )
            log_line += (
                f" | eval_loss = {eval_loss.item()} | eval_acc = {eval_acc.item()} | clean_acc = {clean_acc} | adv_acc = {adv_acc} | eval_recall = {eval_recall} | eval_precision = {eval_precision} | eval_f1 = {eval_f1}"
            )

            train_accs.append(train_acc.item())
            eval_accs.append(eval_acc.item())
            clean_accs.append(clean_acc)
            adv_accs.append(adv_acc)

            if eval_acc > best_eval_acc:
                log_line += "\nSaved best model"
                best_eval_acc = eval_acc
                torch.save(
                    classifier.state_dict(),
                    f"{args.root_path}/model_{args.filename_params}_exp-{exp}.pt",
                )
            logger.info(log_line)
        if best_eval_loss < best_final_eval_loss:
            best_final_eval_loss = best_eval_loss
            best_train_accs = train_accs
            best_eval_accs = eval_accs
            best_clean_accs = clean_accs
            best_adv_accs = adv_accs
        classifier.load_state_dict(
            torch.load(
                f"{args.root_path}/model_{args.filename_params}_exp-{exp}.pt",
                map_location="cpu",
            )
        )
        classifier = classifier.to(args.device)
        test_loss, test_acc, test_clean_acc, test_adv_acc, test_recall, test_precision, test_f1 = eval_classifier(classifier, test_loader, args.device)
        logger.info(f"test_loss = {test_loss} | test_acc = {test_acc} | test_clean_acc = {test_clean_acc} | test_adv_acc = {test_adv_acc} | test_recall = {test_recall} | test_precision = {test_precision} | test_f1 = {test_f1}")
        with open(f"{args.root_path}/results_{args.filename_params}.txt", "a+") as f:
            f.write(f"Experiment {exp}: test_acc = {test_acc} | test_clean_acc = {test_clean_acc} | test_adv_acc = {test_adv_acc} | test_recall = {test_recall} | test_precision = {test_precision} | test_f1 = {test_f1}\n")
        acc_list.append(test_acc)
        recall_list.append(test_recall)
        precision_list.append(test_precision)
        f1_list.append(test_f1)
        clean_acc_list.append(test_clean_acc)
        adv_acc_list.append(test_adv_acc)
    with open(f"{args.root_path}/results_{args.filename_params}.txt", "a+") as f:
        f.write(f"Summary: acc = {np.mean(acc_list)} ± {np.std(acc_list)} | recall = {np.mean(recall_list)} ± {np.std(recall_list)} | precision = {np.mean(precision_list)} ± {np.std(precision_list)} | f1_list = {np.mean(f1_list)} ± {np.std(f1_list)}\n")
    os.makedirs("./results/", exist_ok=True)
    csv_path = "./results/summary3.csv"

    torch.save({
        "train_accs": best_train_accs,
        "eval_accs": best_eval_accs,
        "clean_accs": best_clean_accs,
        "adv_accs": best_adv_accs,
    }, f"{args.root_path}/epoch_metrics_{args.filename_params}.pt")
    
    
    with open(csv_path, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        file_exists = os.path.getsize(csv_path) > 0 
        writer = csv.DictWriter(f, fieldnames=[
            "task", "variant", "scores", "epochs", "batch_size", "width", "lr", "wd", "normalize", "scheduler",
            "acc_mean", "acc_std", "clean_acc_mean", "clean_acc_std",
            "adv_acc_mean", "adv_acc_std", "recall_mean", "recall_std",
            "precision_mean", "precision_std", "f1_mean", "f1_std"
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "task": args.task,
            "variant": args.variant,
            "scores": scores_str,
            "epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "width": args.width,
            "lr": args.lr,
            "wd": args.wd,
            "normalize": args.normalize,
            "scheduler": args.scheduler,
            "acc_mean": np.mean(acc_list),
            "acc_std": np.std(acc_list),
            "clean_acc_mean": np.mean(clean_acc_list),
            "clean_acc_std": np.std(clean_acc_list),
            "adv_acc_mean": np.mean(adv_acc_list),
            "adv_acc_std": np.std(adv_acc_list),
            "recall_mean": np.mean(recall_list),
            "recall_std": np.std(recall_list),
            "precision_mean": np.mean(precision_list),
            "precision_std": np.std(precision_list),
            "f1_mean": np.mean(f1_list),
            "f1_std": np.std(f1_list),
        })
        fcntl.flock(f, fcntl.LOCK_UN)
        

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    args = parse_args()
    # args.device = f"cuda:{args.device}" if args.device is not None else "cpu"
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.device)
    main(args)
