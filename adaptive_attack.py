import os
import torch
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from classifiers.classifier import CifarClassifier
from diffusion.load_diffusion_model import load_diff_model
from combined_model import CombinedModel

# define loss functions 
def classification_loss(logits, labels):
    return F.cross_entropy(logits, labels)

def detection_loss(logits, is_adv_labels):
    return F.cross_entropy(logits, is_adv_labels)

# goal is to maximize classification loss to fool the classifier D, and 
# minimize detection loss (look clean) to fool the classifier C
def combined_loss(logits_C, logits_D, y_class, adv_label, alpha=0.5):
    return (
        classification_loss(logits_D, y_class)
        - alpha * detection_loss(logits_C, adv_label)
    )
    
    
# pgd attack 
# model: CombinedModel
# x: clean images (B, C, H, W)
# y: true class labels
# eps: max perturbation
# alpha: step size per iteration
# steps: number of PGD iterations
# beta: weight for store consistency
def pgd_attack(model, x, y, eps, alpha, steps, beta=0.1):
    # get clean scores once (no grad)
    with torch.no_grad():
        _, _, clean_scores = model(x)

    # initialize adversarial image
    x_adv = x.clone().detach().requires_grad_(True)

    # pgd loop
    for _ in range(steps):
        # forward pass in combined model
        logits_C, logits_D, adv_scores = model(x_adv)

        # define the "clean" target for the detector to be 0 
        adv_label = torch.zeros_like(y)

        # main attack loss
        loss_main = combined_loss(
            logits_C=logits_C,
            logits_D=logits_D,
            y_class=y,
            adv_label=adv_label,
            alpha=0.5
        )

        # penalize changes in diffusion scores
        score_loss = F.mse_loss(adv_scores, clean_scores)

        # if beta is too high -> weaker attack, more realistic
        # if bets is too low -> stronger attack, more detectable
        loss = loss_main + beta * score_loss

        # zero gradients (prevent gradients from previous steps piling up)
        if x_adv.grad is not None:
            x_adv.grad.zero_()

        loss.backward()

        with torch.no_grad():
            # maximize loss- move pixels in direction that maximizes loss
            x_adv = x_adv + alpha * x_adv.grad.sign()

            # project into valid image pizel range
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            x_adv = torch.clamp(x_adv, 0, 1)

        # re enable gradients for the next iteration
        x_adv.requires_grad_(True)

    # return adversarial image batch
    return x_adv

# evaluation 
def evaluate(model, x, x_adv, y, device, threshold=0.5):
    model.eval()
    
    with torch.no_grad():
        logits_C_adv, logits_D_adv, adv_scores = model(x_adv)
        _, _, clean_scores = model(x)
        
        prob_D = F.softmax(logits_D_adv, dim=1)
        prob_C = F.softmax(logits_C_adv, dim=1)
        
        #argmax accuracy 
        pred_D = logits_D_adv.argmax(dim=1)
        acc_argmax = (pred_D == y).float().mean().item()

        pred_C = logits_C_adv.argmax(dim=1)
        
        # confidence-based accuracy 
        true_class_prob = prob_D.gather(1, y.unsqueeze(1)).squeeze()
        confidence_true_class_rate = (true_class_prob > threshold).float().mean().item()
        
        #detector "clean" confidence (assumes class = 0 is clean)
        clean_prob = prob_C[:,0]
        
        #attack success- success = model not confident in true class & looks clean
        success = (true_class_prob < threshold) & (clean_prob > threshold)
        success_rate = success.float().mean().item()
        
        #mse
        img_mse = F.mse_loss(x_adv, x).item()
        score_mse = F.mse_loss(adv_scores, clean_scores).item()

        #new stuff 
        missclassification_rate = (pred_D != y).float().mean().item()
        detector_evasion_rate = (pred_C == 0).float().mean().item()
        joint_success = ((pred_D != y) & (pred_C == 0)).float().mean().item()
        
    return {
        "classification_accuracy_argmax": acc_argmax,
        "confidence_true_class_rate": confidence_true_class_rate,
        "attack_success": success_rate,
        "missclassification_rate": missclassification_rate,
        "detector_evasion_rate": detector_evasion_rate,
        "joint_success": joint_success,
        "img_mse": img_mse,
        "score_mse": score_mse
    }


def run_experiment(
    model,
    transform,
    batch_size=32,
    num_samples=25000,
    eps=8/255,
    alpha=2/255,
    steps=10,
    device="cuda"
):
    # use cifar 10
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    
    indices = torch.randperm(len(dataset))[:num_samples]
    subset = Subset(dataset, indices)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    
    model = model.to(device)
    model.eval()

    all_adv = []
    all_clean = []
    all_labels = []

    acc_clean_total = 0
    acc_adv_total = 0
    conf_adv_total = 0
    success_total = 0
    missclass_rate_total = 0
    det_evasion_rate_total = 0
    joint_success_total = 0
    img_mse_total = 0
    score_mse_total = 0
    n_batches = 0
    
    #pgd loop
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # clean evaluation
        metrics_clean = evaluate(model, x, x, y, device)

        # adversarial generation
        x_adv = pgd_attack(
            model=model,
            x=x,
            y=y,
            eps=eps,
            alpha=alpha,
            steps=steps
        )

        # adversarial evaluation
        metrics_adv = evaluate(model, x, x_adv, y, device)

        all_adv.append(x_adv.cpu())
        all_clean.append(x.cpu())
        all_labels.append(y.cpu())

        acc_clean_total += metrics_clean["classification_accuracy_argmax"]
        conf_adv_total += metrics_adv["confidence_true_class_rate"]
        acc_adv_total += metrics_adv["classification_accuracy_argmax"]
        success_total += metrics_adv["attack_success"]
        missclass_rate_total += metrics_adv["missclassification_rate"]
        det_evasion_rate_total += metrics_adv["detector_evasion_rate"]
        joint_success_total += metrics_adv["joint_success"]
        img_mse_total += metrics_adv["img_mse"]
        score_mse_total += metrics_adv["score_mse"]

        n_batches += 1
        
    results = {
        "clean_acc_argmax": acc_clean_total / n_batches,
        "adv_acc_argmax": acc_adv_total / n_batches,
        "adv_confidence_true_class": conf_adv_total / n_batches,
        "attack_success": success_total / n_batches,
        "missclassification_rate": missclass_rate_total / n_batches,
        "detector_evasion_rate": det_evasion_rate_total / n_batches,
        "joint_success": joint_success_total / n_batches,
        "img_mse": img_mse_total / n_batches,
        "score_mse": score_mse_total / n_batches
    }
    
    #print results
    print("\n===== RESULTS =====")
    for k, v in results.items():
        print(f"{k}: {v:.6f}")
     
    # save results   
    os.makedirs("results", exist_ok=True)
    txt_path = f"results/adaptive_attack_results.txt"
    with open(txt_path, "w") as f:
        f.write("===== RESULTS =====\n")
        for k, v in results.items():
            f.write(f"{k}: {v:.6f}\n")

    print(f"Saved results to: {txt_path}")
    
    
    # save adversarial + clean dataset
    adv_dataset = torch.cat(all_adv)
    clean_dataset = torch.cat(all_clean)
    labels = torch.cat(all_labels)
    
    torch.save({
        "adv_data": adv_dataset,
        "labels": labels
    },  "results/adv_dataset.pt")

    torch.save({
        "data": clean_dataset,
        "labels": labels
    },  "results/clean_dataset.pt")

    return adv_dataset, labels

    
if __name__ == "__main__": 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    start_time = time.perf_counter()

    # load checkpoints
    diff_ckpt = "checkpoints/diffmodel_checkpoint.pth"
    C_ckpt = "checkpoints/C_adv_detect.pt"
    D_ckpt = "checkpoints/D_img_classify.pt"

    score_model, noise_levels = load_diff_model(diff_ckpt, device)
    
    classifier_C = CifarClassifier(
        depth=28,
        widen_factor=5,
        num_score_channels=9,
        num_classes=10,
        dropRate=0.5,
        scores=["score6", "score3", "score0"],
    ).to(device)

    classifier_D = CifarClassifier(
        depth=28,
        widen_factor=5,
        num_score_channels=9,
        num_classes=10,
        dropRate=0.5,
        scores=["score6", "score3", "score0"],
    ).to(device)
    
    classifier_C.load_state_dict(torch.load(C_ckpt, map_location=device))
    classifier_D.load_state_dict(torch.load(D_ckpt, map_location=device))
 
    stats = torch.load("data/norm_stats.pt")
    img_mean = stats["img_mean"]
    img_std = stats["img_std"]
    score_means = stats["score_means"]
    score_stds = stats["score_stds"]
 
    transform = transforms.ToTensor()
    
    
    model = CombinedModel(
        score_model=score_model,
        classifier_C=classifier_C,
        classifier_D=classifier_D,
        noise_levels=noise_levels,
        img_mean=img_mean,
        img_std=img_std,
        score_means=score_means,
        score_stds=score_stds,
        score_indices=(6, 3, 0)
    ).to(device)

    # run experiment 
    adv_data, labels = run_experiment(
        model=model,
        transform=transform,
        device=device
    )
      
    total_time = time.perf_counter() - start_time
    with open("results/total_runtime.txt", "w") as f:
        f.write(f"Total runtime (seconds): {total_time:4f}\n")
