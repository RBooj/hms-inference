from __future__ import annotations

from dataclasses import dataclass
import torch
from transformers import AutoFeatureExtractor, ASTModel


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
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0",
        device: str | None = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
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
            embedding=emb.detech().cpu(),
            hidden_dim=int(emb.shape[0]),
            model_name=self.model_name,
        )
