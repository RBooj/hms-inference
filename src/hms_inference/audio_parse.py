import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class AudioMeta:
    hive_id: int
    recording_start: datetime


FILENAME_RE = re.compile(
    r"""
        (?P<date>\d{2}-\d{2}-\d{4})         # dd-mm-yyy
        _
        (?P<hour>\d{2})h(?P<minute>\d{2})   # HHhMM
        _
        (?P<hive>hive-\d+)                  # hive-####
        \.(wav)                             # extention
        $
        """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_urban_wav_name(path: Path) -> AudioMeta:
    m = FILENAME_RE.search(path.name)
    print(m)
    if not m:
        raise ValueError(f"Unrecognized UrBAN wav filename: {path.name}")

    date_str = m.group("date")
    hour = int(m.group("hour"))
    minute = int(m.group("minute"))

    hive_str = m.group("hive")
    hive_id = int(hive_str.split("-")[1])

    recording_start = datetime.strptime(date_str, "%d-%m-%Y").replace(
        hour=hour, minute=minute
    )
    return AudioMeta(hive_id=hive_id, recording_start=recording_start)


print(f"Looking in: {Path.cwd()} for file: {Path("11-08-2021_20h00_HIVE-3631.WAV")}")
test = Path("11-08-2021_20h00_HIVE-3631.WAV")
print(parse_urban_wav_name(test))
