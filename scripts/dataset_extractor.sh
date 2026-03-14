#!/bin/bash
set -euo pipefail

ARCHIVE_DIR_2021="./data/UrBAN/data/audio/beehives_2021"
ARCHIVE_DIR_2022="./data/UrBAN/data/audio/beehives_2022"

echo "[UrBAN Extractor] Extracting 2021 and 2022 audio files"
echo "[UrBAN Extractor] The archives will be deleted after extraction."

extract_archives_in_dir() {
    local archive_dir="$1"
    local label="$2"

    shopt -s nullglob
    local archives=("$archive_dir"/*.tar.gz)
    local l_wavs=("$archive_dir"/*.wav)
    local u_wavs=("$archive_dir"/*.WAV)
    local total_l_wavs=${#l_wavs[@]}
    local total_u_wavs=${#u_wavs[@]}
    local total_wavs=$(( total_l_wavs + total_u_wavs ))
    local total_archives=${#archives[@]}
    local count=0

    echo "--------------------------------------------------------------------------------"
    echo "[UrBAN Extractor] $label: discovered $total_archives archives in $archive_dir"
    echo "[UrBAN Extractor[ $label: discovered $total_wavs existing wavs in $archive_dir"

    if [ "$total_archives" -eq 0 ]; then
        echo "[UrBAN Extractor] $label: nothing to extract."
        return 0
    fi

    for archive in "$archives[@]}"; do
        count=$((count + 1))
        local archive_name=$(basemane "$archive")

        echo "[UrBAN Extractor] $label [$count/$total_archives] Extracting: $archive_name"

        if tar -xzf "$archive" -C "$archive_dir"; then
            echo "[UrBAN Extractor] $label [$count/$total_archives] Extraction successful. Removing archive"
            rm "$archive"
            echo "[UrBAN Extractor] $label [$count/$total_archives] Removed: $archive_name"
            echo "[UrBAN Extractor] $label Checking disk usage after extraction:"
            du -sh "$archive_dir"
        else
            echo "[UrBAN Extractor] $label [$count/$total_archives] Extraction failed. Archive not deleted"
            return 1
        fi
    done
}

extract_archives_in_dir "$ARCHIVE_DIR_2021" "2021"
extract_archives_in_dir "$ARCHIVE_DIR_2022" "2022"

echo
echo "[UrBAN Extractor] All extractions completed."

