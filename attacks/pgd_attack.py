import torch
import torch.nn.functional as F
from losses import attack_loss


def pgd_attack(model, x, y, eps, alpha, steps, beta=0.1):
    # get clean scores once (no grad)
    with torch.no_grad():
        _, _, clean_scores = model(x)

    x_adv = x.clone().detach().requires_grad_(True)

    for _ in range(steps):
        logits_C, logits_D, adv_scores = model(x_adv)

        # detector target: "clean"
        adv_label = torch.zeros_like(y)

        # main attack loss
        loss_main = attack_loss(
            logits_C=logits_C,
            logits_D=logits_D,
            y_class=y,
            adv_label=adv_label,
            alpha=0.5
        )

        # score consistency term (from proposal)
        score_loss = F.mse_loss(adv_scores, clean_scores)

        loss = loss_main + beta * score_loss

        # zero gradients
        if x_adv.grad is not None:
            x_adv.grad.zero_()

        loss.backward()

        with torch.no_grad():
            # maximize loss
            x_adv = x_adv + alpha * x_adv.grad.sign()

            # projection
            x_adv = torch.clamp(x_adv, 0, 1)
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)

        x_adv.requires_grad = True

    return x_adv