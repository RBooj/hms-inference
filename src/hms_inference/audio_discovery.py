from pathlib import Path

PROJECT_ROOT = Path.cwd()
DATA_ROOT = PROJECT_ROOT / "data" / "UrBAN" / "data"
AUDIO_ROOT_2021 = DATA_ROOT / "audio" / "beehives_2021"
AUDIO_ROOT_2022 = DATA_ROOT / "audio" / "beehives_2022"


def find_wavs(audio_root: Path) -> list[Path]:
    """
    Find wav files in project data directory
    """
    wavs = []
    for ext in ("*.wav", "*.WAV"):
        wavs.extend(audio_root.rglob(ext))
    return sorted(wavs)


wavs_2021 = find_wavs(AUDIO_ROOT_2021)
wavs_2022 = find_wavs(AUDIO_ROOT_2022)

print("2021 wavs found: ", len(wavs_2021))
print("2022 wavs found: ", len(wavs_2022))
print("Example (2021): ", wavs_2021[0] if wavs_2021 else None)
print("Example (2022): ", wavs_2022[0] if wavs_2022 else None)
