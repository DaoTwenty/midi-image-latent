#!/bin/bash
# Download all datasets for MIDI Image VAE
#
# Usage:
#   bash scripts/download_data.sh [dataset]
#   dataset: all | lakh | maestro | pop909   (default: all)
#
# Environment variables:
#   DATA_DIR — root directory for data (default: data/)
#
# Note: run this script on a login node; compute nodes have no outbound internet.

set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
DATASET="${1:-all}"

# ── helpers ────────────────────────────────────────────────────────────────────

log() { echo "[download_data] $*"; }
die() { echo "[download_data] ERROR: $*" >&2; exit 1; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "Required command '$1' not found. Please install it."
}

# ── dataset downloaders ────────────────────────────────────────────────────────

download_lakh() {
    # Lakh MIDI Dataset (LMD full version)
    # ~1.7 GB compressed, ~4 GB extracted, ~178K .mid files
    # Source: http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz
    log "Downloading Lakh MIDI Dataset (LMD full) ..."
    require_cmd wget
    require_cmd tar

    local lakh_dir="$DATA_DIR/lakh"
    local archive="$DATA_DIR/lmd_full.tar.gz"

    mkdir -p "$lakh_dir"

    if [ -f "$archive" ]; then
        log "Archive already present at $archive — skipping download."
    else
        wget \
            --no-verbose \
            --show-progress \
            -O "$archive" \
            "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz"
    fi

    log "Extracting $archive ..."
    tar xzf "$archive" -C "$lakh_dir" --strip-components=1

    rm -f "$archive"

    local n_files
    n_files=$(find "$lakh_dir" -name '*.mid' | wc -l)
    log "Lakh: $n_files .mid files extracted to $lakh_dir"
}

download_maestro() {
    # MAESTRO v3 — classical piano performances in MIDI format
    # ~58 MB (MIDI only), 1276 .midi files organised by year
    # Source: Google Magenta / Magenta Storage
    log "Downloading MAESTRO v3 ..."
    require_cmd wget
    require_cmd unzip

    local maestro_dir="$DATA_DIR/maestro"
    local archive="$DATA_DIR/maestro.zip"

    # Skip if already downloaded
    if [ -d "$maestro_dir/maestro-v3.0.0" ]; then
        local n_existing
        n_existing=$(find "$maestro_dir" -name '*.midi' | wc -l)
        log "MAESTRO already present ($n_existing .midi files in $maestro_dir) — skipping."
        return 0
    fi

    mkdir -p "$maestro_dir"

    wget \
        --no-verbose \
        --show-progress \
        -O "$archive" \
        "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"

    log "Extracting $archive ..."
    unzip -q "$archive" -d "$maestro_dir/"
    rm -f "$archive"

    local n_files
    n_files=$(find "$maestro_dir" -name '*.midi' | wc -l)
    log "MAESTRO: $n_files .midi files extracted to $maestro_dir"
}

download_pop909() {
    # POP909 — 909 pop songs with melody, bridge, accompaniment tracks
    # ~20 MB, 909 directories each containing a .mid file
    # Source: MIDI-Unprocessed GitHub release
    log "Downloading POP909 ..."
    require_cmd wget
    require_cmd unzip

    local pop_dir="$DATA_DIR/pop909"
    local archive="$DATA_DIR/pop909.zip"

    if [ -d "$pop_dir" ] && [ "$(find "$pop_dir" -name '*.mid' | wc -l)" -gt 900 ]; then
        local n_existing
        n_existing=$(find "$pop_dir" -name '*.mid' | wc -l)
        log "POP909 already present ($n_existing .mid files in $pop_dir) — skipping."
        return 0
    fi

    mkdir -p "$pop_dir"

    # POP909 is distributed as a GitHub release archive
    wget \
        --no-verbose \
        --show-progress \
        -O "$archive" \
        "https://github.com/music-x-lab/POP909-Dataset/archive/refs/heads/master.zip"

    log "Extracting $archive ..."
    # The zip contains POP909-Dataset-master/ with a POP909/ subdirectory
    unzip -q "$archive" -d "$DATA_DIR/_pop909_tmp/"
    # Move the actual MIDI files into the canonical location
    if [ -d "$DATA_DIR/_pop909_tmp/POP909-Dataset-master/POP909" ]; then
        cp -r "$DATA_DIR/_pop909_tmp/POP909-Dataset-master/POP909/." "$pop_dir/"
    else
        # Fallback: just move everything
        cp -r "$DATA_DIR/_pop909_tmp/POP909-Dataset-master/." "$pop_dir/"
    fi
    rm -rf "$DATA_DIR/_pop909_tmp" "$archive"

    local n_files
    n_files=$(find "$pop_dir" -name '*.mid' | wc -l)
    log "POP909: $n_files .mid files extracted to $pop_dir"
}

# ── main ───────────────────────────────────────────────────────────────────────

log "Data root: $DATA_DIR"
log "Dataset(s): $DATASET"

mkdir -p "$DATA_DIR"

case "$DATASET" in
    all)
        download_lakh
        download_maestro
        download_pop909
        ;;
    lakh)
        download_lakh
        ;;
    maestro)
        download_maestro
        ;;
    pop909)
        download_pop909
        ;;
    *)
        die "Unknown dataset '$DATASET'. Valid options: all | lakh | maestro | pop909"
        ;;
esac

log "Done."
