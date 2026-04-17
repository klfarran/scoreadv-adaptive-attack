import torch
import torch.nn.functional as F 

def combined_attack(
    x_init,   # initial image or noise- controlled in main
    score_model,
    classifier_C,
    classifier_D,
    noise_levels,
    target_class,
    num_steps=50,
    step_size=0.01,
    lambda_cls=1.0,
    lambda_det=1.0,
    device="cuda"
):

    x = x_init.clone().detach().to(device)
    x.requires_grad = True

    score_model.eval()
    classifier_C.eval()
    classifier_D.eval()

    for step in range(num_steps):

        sigma = noise_levels[step % len(noise_levels)]
        sigma_tensor = torch.ones(x.shape[0], device=device) * sigma

        score = score_model(x, sigma_tensor)

        # Classifier C (clean label)
        logits_C = classifier_C(x)
        target_C = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        loss_C = F.cross_entropy(logits_C, target_C)
        
        # Classifier D (targeted attack)
        logits_D = classifier_D(x)
        target_D = torch.ones(x.shape[0], dtype=torch.long, device=device) * target_class
        loss_D = F.cross_entropy(logits_D, target_D)

        total_loss = lambda_cls * loss_D + lambda_det * loss_C

        # backward pass
        grad = torch.autograd.grad(total_loss, x)[0]

        # update x
        with torch.no_grad():
            x = x + step_size * score - step_size * grad

        x.requires_grad = True

        if step % 10 == 0:
            print(f"Step {step} | L_D: {loss_D.item():.4f} | L_C: {loss_C.item():.4f}")

    return x.detach()