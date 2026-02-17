from __future__ import annotations

import torch
import torchaudio

def load_audio_mono_16k(path: str) -> torch.Tensor:
