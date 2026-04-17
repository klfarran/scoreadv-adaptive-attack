import torch 
import torch.nn as nn 


class CombinedModel(nn.Module):
    def __init__(
        self,
        score_model,
        classifier_C,
        classifier_D,
        noise_levels,
        score_indices=(6, 3, 0),
    ):
        super().__init__()
        
        self.score_model = score_model
        self.classifier_C = classifier_C
        self.classifier_D = classifier_D

        self.noise_levels = noise_levels
        self.score_indices = score_indices
    
    def forward(self, x):
        # pass through diffusion model to get score maps 
        score_maps = []
        for i in self.score_indices:
            sigma = self.noise_levels[i]
            sigma_batch = torch.full(
                (x.shape[0],),
                float(sigma),
                device=x.device
            )

            # want to detatch to prevent gradients from entering the diffuion model
            # want this because we want to treat diff model as a fixed feature extractor
            score = self.score_model(x, sigma_batch).deteach()
            score_maps.append(score)

        # (B, 3*k, H, W)
        scores = torch.cat(score_maps, dim=1)


        # build classifier input
        x_in = torch.cat([x, scores], dim=1)


        # forward through both classifiers
        logits_C = self.classifier_C(x_in)
        logits_D = self.classifier_D(x_in)

        return logits_C, logits_D, scores