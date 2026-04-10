from __future__ import annotations

from dataclasses import dataclass
import torch
from transformers import ASTFeatureExtractor, ASTModel


@dataclass
class ASTEmbeddingResult:
    embedding: torch.Tensor
    hidden_dim: int
    model_name: str


class ASTEmbedder:
    """
    AST -> Embedding extractor
    Use model's pooled output if available
    Otherwise mean-pool last_hidden_state
    """

    def __init__(
        self,
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0.448",
        device: str | None = None,
    ):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.extractor = ASTFeatureExtractor.from_pretrained(model_name)
        self.model = ASTModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def embed(self, waveform_16k: torch.Tensor) -> ASTEmbeddingResult:
        """
        waveform_16k: shape [num_samples], 16kHz mono
        """

        # transformers expects numpy arrays or lists
        # feature extractor returns torch tensors if asked
        inputs = self.extractor(
            waveform_16k.cpu().numpy(), sampling_rate=16000, return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        out = self.model(**inputs)

        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            emb = out.pooler_output[0]
        else:
            last_hidden = out.last_hidden_state[0]
            emb = last_hidden.mean(dim=0)

        return ASTEmbeddingResult(
            embedding=emb.detach().cpu(),
            hidden_dim=int(emb.shape[0]),
            model_name=self.model_name,
        )

    @torch.inference_mode()
    def embed_batch(self, waveforms_16k: list[torch.Tensor]) -> ASTEmbeddingResult:
        if not waveforms_16k:
            raise ValueError("embed_batch received an empty waveform list.")

        for i, wav in enumerate(waveforms_16k):
            if wav.ndim != 1:
                raise ValueError(f"Waveform at index {i} is not 1D: shape={wav.shape}")
            if wav.numel() == 0:
                raise ValueError(f"Waveform at index {i} is empty.")

        inputs = self.extractor(
                [wav.cpu().numpy() for wav in waveforms_16k],
                sampling_rate=16000,
                return_tensors="pt",
                )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
                ):
            out = self.model(**inputs)

        # choose one embedding definition and keep it fixed
        emb = out.last_hidden_state[:, 0, :]   # [batch, hidden_dim]

        return ASTEmbeddingResult(
                embedding=emb.detach().cpu(),
                hidden_dim=int(emb.shape[-1]),
                model_name=self.model_name,
                )
