from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from hms_inference.audio_io import load_audio_mono_16k
from hms_inference.ast_embedder import ASTEmbedder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("wav_path", help="Path to a .wav file")
    p.add_argument(
        "--out", default="embedding.npy", help="Where to save the embedding (npy)"
    )
    p.add_argument(
        "--model",
        default="MIT/ast-finetuned-audioset-10-10-0",
        help="HuggingFace model id",
    )
    args = p.parse_args()

    wav_path = Path(args.wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)

    waveform = load_audio_mono_16k(str(wav_path))
    embedder = ASTEmbedder(model_name=args.model)

    result = embedder.embed(waveform)
    emb = result.embedding.numpy().astype(np.float32)

    np.save(args.out, emb)

    meta = {
        "wav_path": str(wav_path),
        "model": result.model_name,
        "embedding_dim": result.hidden_dim,
        "saved_to": args.out,
    }
    print(json.dumps(meta, indent=2))
    print("First 10 values:", emb[:10])


if __name__ == "__main__":
    main()
