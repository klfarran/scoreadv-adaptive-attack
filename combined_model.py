import torch 
import torch.nn as nn 


class CombinedModel(nn.Module):
    def __init__(
        self,
        score_model,
        classifier_C,
        classifier_D,
        noise_levels,
        img_mean,
        img_std,
        score_means,
        score_stds,
        score_indices=(6, 3, 0),
    ):
        super().__init__()
        
        self.score_model = score_model
        self.classifier_C = classifier_C
        self.classifier_D = classifier_D

        self.noise_levels = noise_levels
        self.score_indices = score_indices
        
        # normalization buffers
        self.register_buffer("img_mean", img_mean.float())
        self.register_buffer("img_std", img_std.float())
    
        # stack score stats into tensors of shape (num_levels, 3)
        self.register_buffer(
            "score_means",
            torch.stack([m.float() for m in score_means])
        )
        self.register_buffer(
            "score_stds",
            torch.stack([s.float() for s in score_stds])
        )
    
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

            # i think that we want to detatch to prevent gradients from entering the diffuion model
            # want this because we want to treat diff model as a fixed feature extractor
            score = self.score_model(x, sigma_batch).detach()
            score_maps.append(score)

        scores = torch.cat(score_maps, dim=1)

        # build classifier input (concatenate image + scores)
        x_in = torch.cat([x, scores], dim=1)

        # normalization tensors
        means = [self.img_mean]
        stds = [self.img_std]

        for i in self.score_indices:
            means.append(self.score_means[i])
            stds.append(self.score_stds[i])

        # shape → (1, total_channels, 1, 1)
        mean = torch.cat(means).view(1, -1, 1, 1)
        std = torch.cat(stds).view(1, -1, 1, 1)
        
        assert x_in.shape[1] == mean.shape[1]

        # normalize input
        x_in = (x_in - mean) / (std + 1e-8)

        # forward through both classifiers
        logits_C = self.classifier_C(x_in)
        logits_D = self.classifier_D(x_in)

        return logits_C, logits_D, scores