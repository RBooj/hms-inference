#!/bin/bash
set -e

ARCHIVE_DIR_2021="./data/UrBAN/data/audio/beehives_2021"
ARCHIVE_DIR_2022="./data/UrBAN/data/audio/beehives_2022"

echo "[UrBAN Extractor] Extracting 2021 and 2022 audio files. The archives will be deleted after extraction."

shopt -s nullglob
files_2021_archives=("$ARCHIVE_DIR_2021"/*.tar.gz)
count_2021_archives=${#files_2021_archives[@]}

files_2022_archives=("$ARCHIVE_DIR_2022"/*.tar.gz)
count_2022_archives=${#files_2022_archives[@]}

echo "Discovered $count_2021_archives archives from 2021"
echo "Discovered $count_2022_archives archives from 2022"

count=0
for archive in "$ARCHIVE_DIR_2021"/*.tar.gz; do
    echo "[UrBAN Extractor] Extracting archive: $archive ..."

    tar -xzf "$archive" -C "$ARCHIVE_DIR_2021"

    echo "[UrBAN Extractor] Extraction of [$archive] complete. Deleting..."
    if [ $? -eq 0 ]; then
        rm "$archive"
        echo "Archive removed."
    else
        echo "Extraction failed, archive kept."
    fi
    count=count+1
    remaining=$((count_2021_archives-count))
    echo "Extracted $count archives. ${""} remain."
done

for archive in "$ARCHIVE_DIR_2022"/*.tar.gz; do
    echo "[UrBAN Extractor] Extracting archive: $archive ..."

    tar -xzf "$archive" -C "$ARCHIVE_DIR_2022"

    echo "[UrBAN Extractor] Extraction of [$archive] complete. Deleting..."
    if [$? -eq 0]; then
        rm "$archive"
        echo "Archive removed."
    else
        echo "Extraction failed, archive kept."
    fi
done
