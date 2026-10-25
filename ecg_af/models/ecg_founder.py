from pathlib import Path

import torch
from torch import nn

from ..util import GLOBAL_CONFIG
from .net1d import Net1D


class AFECGFounder1D(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = GLOBAL_CONFIG["models"]["ecg_founder"]
        self.in_channels = cfg["in_channels"]
        net1d_cfg = cfg["net1d"]
        n_classes = cfg["n_classes"]

        self.backbone = Net1D(
            in_channels=len(self.in_channels),
            n_classes=n_classes,
            **net1d_cfg,
        )

        pretrained_path = Path(__file__).parent.parent / cfg["pretrained_path"]
        checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"]
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("dense.")}
        missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
        GLOBAL_CONFIG.logger.info(
            "ECGFounder loaded from %s | missing=%d unexpected=%d (dense.* expected missing)",
            pretrained_path,
            len(missing),
            len(unexpected),
        )


        in_features = self.backbone.dense.in_features
        self.backbone.dense = nn.Linear(in_features, n_classes)

        if cfg["linear_prob"]:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("dense."):
                    param.requires_grad = False

    def forward(self, x):
        out = self.backbone(x[:, self.in_channels, :])
        return out.squeeze(-1)
