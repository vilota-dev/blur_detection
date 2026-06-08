import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )
        self.shortcut = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))

class BlurClassifier(nn.Module):
    def __init__(self, backbone, embed_dim, num_classes=4, laplacian_input_dim=196):
        super().__init__()
        self.backbone = backbone
        linear_input_dim = 2 * embed_dim
        self.laplacian_projector = nn.Sequential(
            nn.Linear(laplacian_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        fused_input_dim = linear_input_dim + 128
        self.classifier_head = nn.Sequential(
            nn.Linear(fused_input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            ResidualBlock(2048, 1024),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, patch_mask, laplacian_tensor):
        features = self.backbone.forward_features(x)
        cls_token = features["x_norm_clstoken"]
        patch_tokens = features["x_norm_patchtokens"]
        
        mask_weights = patch_mask.float().unsqueeze(-1)
        weighted_patches = patch_tokens * mask_weights
        summed_patches = weighted_patches.sum(dim=1)
        valid_patch_count = mask_weights.sum(dim=1) + 1e-6
        masked_patch_mean = summed_patches / valid_patch_count

        dino_feature = torch.cat([cls_token, masked_patch_mean], dim=1)

        pooled_laplacian = F.adaptive_avg_pool2d(laplacian_tensor, (14, 14))
        flat_laplacian = pooled_laplacian.view(pooled_laplacian.size(0), -1)
        laplacian_features = self.laplacian_projector(flat_laplacian)

        fused_input = torch.cat([dino_feature, laplacian_features], dim=1)

        logits = self.classifier_head(fused_input)

        return logits