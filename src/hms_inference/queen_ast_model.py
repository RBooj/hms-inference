from __future__ import annotations

import torch
import torch.nn as nn
from transformers import ASTFeatureExtractor, ASTModel, AutoFeatureExtractor, ASTForAudioClassification


class ASTQueenClassifier(nn.Module):
    """
    End-to-end AST binary classifier.

    audio -> AST backbone -> CLS token -> small classifier head -> 1 logit
    """

    def __init__(
        self,
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0.448",
        dropout: float = 0.2,
        mean: float = None,
        std: float = None
    ):
        super().__init__()

        self.model_name = model_name
        self.extractor = ASTFeatureExtractor.from_pretrained(model_name, do_normalize=False, mean=mean, std=std)
        self.backbone = ASTModel.from_pretrained(model_name)

        hidden_dim = self.backbone.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def unfreeze_last_n_encoder_layers(self, n: int) -> None:
        """
        Freeze entire backbone, then unfreeze only the last n encoder layers.
        Also unfreeze final layernorm/pool-adjacent parameters if present.
        """
        self.freeze_backbone()

        # HuggingFace AST is ViT-like internally
        encoder_layers = getattr(self.backbone.encoder, "layer", None)
        if encoder_layers is None:
            raise AttributeError("Could not find backbone.encoder.layer on ASTModel")

        if n <= 0:
            return

        for layer in encoder_layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True

        # Common useful extras
        if hasattr(self.backbone, "layernorm"):
            for p in self.backbone.layernorm.parameters():
                p.requires_grad = True

    def forward(self, waveforms_16k: list[torch.Tensor]) -> torch.Tensor:
        """
        waveforms_16k: list of 1D float32 tensors sampled at 16kHz
        returns logits of shape [batch]
        """
        if not waveforms_16k:
            raise ValueError("forward received empty waveform list")

        inputs = self.extractor(
            [wav.detach().cpu().numpy() for wav in waveforms_16k],
            sampling_rate=16000,
            return_tensors="pt",
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        out = self.backbone(**inputs)

        # Use CLS token representation
        x = out.last_hidden_state[:, 0, :]
        logits = self.classifier(x).squeeze(-1)
        return logits
