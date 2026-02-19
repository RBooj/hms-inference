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
        model_name: str = "MIT/ast-finetuned-audioset-14-14-0.443",
        device: str | None = None,
    ):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = ASTModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def embed(self, waveform_16k: torch.Tensor) -> ASTEmbeddingResult:
        """
        waveform_16k: shape [num_samples], 16kHz mono
        """
        print(
            f"[ast_embedder] waveform_16k: shape={tuple(waveform_16k.shape)}, dtype={waveform_16k.dtype}, device={waveform_16k.device}"
        )

        # transformers expects numpy arrays or lists
        # feature extractor returns torch tensors if asked
        inputs = self.extractor(
            waveform_16k.cpu().numpy(), sampling_rate=16000, return_tensors="pt"
        )
        print(f"[ast_embedder] extractor outputs:")
        for k, v in inputs.items():
            print(
                f"    - {k}: shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}"
            )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        out = self.model(**inputs)
        print("[ast_embedder] model outputs available fields:", out.keys())
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            print(
                f"[ast_embedder] last_hidden_state: shape={tuple(out.last_hidden_state.shape)}, dtype={out.last_hidden_state.dtype}, device={out.last_hidden_state.device}"
            )

        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            emb = out.pooler_output[0]
            print(
                f"[ast_embedder] pooler_output: shape={tuple(out.pooler_output.shape)}, dtype={out.pooler_output.dtype}, device={out.pooler_output.device}"
            )
        else:
            last_hidden = out.last_hidden_state[0]
            emb = last_hidden.mean(dim=0)

        return ASTEmbeddingResult(
            embedding=emb.detach().cpu(),
            hidden_dim=int(emb.shape[0]),
            model_name=self.model_name,
        )
