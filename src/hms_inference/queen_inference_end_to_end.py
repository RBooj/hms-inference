from __future__ import annotations

from pathlib import Path

import torch
import torchaudio

from hms_inference.queen_ast_model import ASTQueenClassifier


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path.cwd() / "data" / "models" / "queen_ast_finetune_best.pt"


def load_audio_mono_16k(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    return wav.squeeze(0).to(torch.float32)


def slice_waveform_chunk(
    waveform_16k: torch.Tensor,
    start_s: float,
    end_s: float,
    sample_rate: int = 16000,
) -> torch.Tensor:
    start_idx = int(round(start_s * sample_rate))
    end_idx = int(round(end_s * sample_rate))
    return waveform_16k[start_idx:end_idx]


def load_model(model_path: Path = MODEL_PATH) -> ASTQueenClassifier:
    checkpoint = torch.load(model_path, map_location=DEVICE)
    config = checkpoint["config"]

    model = ASTQueenClassifier(
        model_name=config["model_name"],
        dropout=config["dropout"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_chunk_probability(
    model: ASTQueenClassifier,
    wav_path: str,
    start_s: float,
    end_s: float,
) -> float:
    wav = load_audio_mono_16k(wav_path)
    chunk = slice_waveform_chunk(wav, start_s, end_s)

    logits = model([chunk])
    prob = torch.sigmoid(logits)[0].item()
    return float(prob)


if __name__ == "__main__":
    model = load_model()
    # example:
    # prob = predict_chunk_probability(model, "path/to/file.wav", 0.0, 10.0)
    # print(prob)
