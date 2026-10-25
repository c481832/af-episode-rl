from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..data.preprocessing import resample_signal
from ..util import GLOBAL_CONFIG


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(12, 256, kernel_size=2, stride=2, bias=False),
                    nn.Identity(),
                    nn.GroupNorm(256, 256),
                    nn.GELU(),
                ),
                *[
                    nn.Sequential(
                        nn.Conv1d(256, 256, kernel_size=2, stride=2, bias=False),
                        nn.GELU(),
                    )
                    for _ in range(3)
                ],
            ]
        )

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x


class ConvPositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        conv = nn.Conv1d(768, 768, kernel_size=128, padding=64, groups=16)
        self.pos_conv = nn.Sequential(nn.utils.parametrizations.weight_norm(conv, dim=2))

    def forward(self, x):
        y = self.pos_conv(x.transpose(1, 2))[:, :, :-1]
        return nn.functional.gelu(y).transpose(1, 2)


class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = nn.Linear(768, 768)
        self.v_proj = nn.Linear(768, 768)
        self.q_proj = nn.Linear(768, 768)
        self.out_proj = nn.Linear(768, 768)

    def forward(self, x):
        return nn.functional.multi_head_attention_forward(
            x,
            x,
            x,
            768,
            12,
            torch.empty(0, device=x.device),
            torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
            None,
            None,
            False,
            0.0,
            self.out_proj.weight,
            self.out_proj.bias,
            False,
            None,
            False,
            None,
            use_separate_proj_weight=True,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
        )[0]


class TransformerEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = SelfAttention()
        self.self_attn_layer_norm = nn.LayerNorm(768)
        self.fc1 = nn.Linear(768, 3072)
        self.fc2 = nn.Linear(3072, 768)
        self.final_layer_norm = nn.LayerNorm(768)

    def forward(self, x):
        x = self.self_attn_layer_norm(x + self.self_attn(x))
        return self.final_layer_norm(x + self.fc2(nn.functional.gelu(self.fc1(x))))


class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer() for _ in range(12)])
        self.layer_norm = nn.LayerNorm(768)

    def forward(self, x):
        x = self.layer_norm(x).transpose(0, 1)
        for layer in self.layers:
            x = layer(x)
        return x.transpose(0, 1)


class ECGFMEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.post_extract_proj = nn.Linear(256, 768)
        self.conv_pos = ConvPositionalEncoding()
        self.layer_norm = nn.LayerNorm(256)
        self.encoder = TransformerEncoder()

    def extract_features(self, source):
        features = self.post_extract_proj(
            self.layer_norm(self.feature_extractor(source).transpose(1, 2))
        )
        return {"x": self.encoder(features + self.conv_pos(features))}


class NoiseDetectionHead(nn.Module):
    def __init__(self, encoder_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, encoder_out):
        return self.mlp(encoder_out.mean(dim=1)).squeeze(-1)


class NoiseDetectorModel(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = GLOBAL_CONFIG["models"]["noise_detector"]
        checkpoint_path = Path(__file__).parent.parent / cfg["checkpoint_path"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["model_state_dict"]
        hidden_dim = state_dict["head.mlp.0.weight"].shape[0]

        self.encoder_model = ECGFMEncoder()
        self.head = NoiseDetectionHead(768, hidden_dim)
        self.load_state_dict(state_dict, strict=False)
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, x):
        x = x[:, 0, :]
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.clamp(std, min=1e-8)
        source = torch.zeros(
            (len(x), 12, x.shape[-1]), dtype=x.dtype, device=x.device
        )
        source[:, 1, :] = x
        features = self.encoder_model.extract_features(source)["x"]
        return self.head(features)


@torch.inference_mode()
def noise_probability(model, signal, fs, start=0):
    """Sigmoid score for a 5 s window (high = clean, low = noisy), matching the notebook."""
    if fs <= 0:
        raise ValueError("fs must be positive")

    cfg = GLOBAL_CONFIG["models"]["noise_detector"]
    segment_seconds = int(GLOBAL_CONFIG["modes"]["noise"]["window_seconds"])
    target_fs = int(cfg["target_fs"])

    n_samples = int(segment_seconds * fs)
    ecg = np.asarray(signal, dtype=np.float32).reshape(-1)
    segment = ecg[start:start + n_samples]
    if len(segment) < n_samples:
        segment = np.pad(segment, (0, n_samples - len(segment)))
    segment = resample_signal(segment, orig_fs=int(fs), target_fs=target_fs).astype(
        np.float32, copy=False
    )

    device = next(model.parameters()).device
    source = torch.from_numpy(segment).reshape(1, 1, -1).to(device)
    logit = model(source)
    return float(torch.sigmoid(logit).item())


def is_noise(model, signal, fs, start=0, threshold=None):
    """Flag a window as noise when its score falls below the threshold (notebook: < 0.05)."""
    if threshold is None:
        threshold = float(GLOBAL_CONFIG["models"]["noise_detector"]["threshold"])
    return noise_probability(model, signal, fs, start=start) < threshold
