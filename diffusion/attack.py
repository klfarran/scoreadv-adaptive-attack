import torch
import torch.nn.functional as F 

def attack(
    x_init,    # initial image or noise- controlled in main
    score_model,
    classifier,
    noise_levels,
    target_class,
    is_detector=False,     #true for classifier C, false for D
    num_steps=50,
    step_size=0.01,
    lambda_cls=1.0,
    lambda_det=1.0,
    device="cuda"    
):
    """
    Diffusion-guided adversarial attack
    
    classifier = our CifarClassifier, either C or D
    """
    
    x = x_init.clone().detach().to(device)
    x.requires_grad = True
    
    score_model.eval()
    classifier.eval()
    
    for step in range(num_steps):
        
        # pick noise level
        sigma = noise_levels[step % len(noise_levels)]
        sigma_tensor = torch.ones(x.shape[0], device=device) * sigma
        
        # score term
        score = score_model(x, sigma_tensor)
        
        # classifier loss
        logits = classifier(x)
        
        if is_detector:
            target = torch.zeros(x.shape[0], dtype=torch.long, device=device)
        else:
            target = torch.ones(x.shape[0], dtype=torch.long, device=device) * target_class
        
        loss = F.cross_entropy(logits, target)
        
        # backward pass
        grad = torch.autograd.grad(loss, x)[0]
        
        # update x
        with torch.no_grad():
            x = x + step_size * score - step_size * grad
            # does this need to be clamped? need to look at the data range 
            
        x.requires_grad = True
        
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
            
    return x.detach()