from __future__ import annotations

import torch
import torchaudio


def load_audio_mono_16k(path: str) -> torch.Tensor:
    """
    Load audio file as single-channel wav sampled at 16kHz
    Return waveform shape [num_samples] (float32, -1..1)
    """
    # wav: [channels, samples]
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        # re-encode to mono channel
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    wav = wav.squeeze(0).to(torch.float32)
    return wav
