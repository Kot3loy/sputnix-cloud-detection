#!/usr/bin/env bash
# split_demo_data.sh — разбиение синтетического датасета на train/val
# для быстрого демо-обучения (используется в README и DEMO_SCRIPT.md).
#
# Использование:  bash split_demo_data.sh [папка-источник] [папка-назначения]
set -euo pipefail

SRC="${1:-data/synthetic}"
DST="${2:-data/processed_demo}"

N_TOTAL=$(ls "$SRC"/images/*.png 2>/dev/null | wc -l)
N_VAL=$(( N_TOTAL / 5 ))          # 20% в валидацию
N_TRAIN=$(( N_TOTAL - N_VAL ))

mkdir -p "$DST/train/images" "$DST/train/masks" "$DST/val/images" "$DST/val/masks"

i=0
for img in "$SRC"/images/*.png; do
    name=$(basename "$img")
    if [ "$i" -lt "$N_TRAIN" ]; then
        cp "$img" "$DST/train/images/"
        cp "$SRC/masks/$name" "$DST/train/masks/"
    else
        cp "$img" "$DST/val/images/"
        cp "$SRC/masks/$name" "$DST/val/masks/"
    fi
    i=$(( i + 1 ))
done

echo "train: $(ls "$DST/train/images" | wc -l) пар, val: $(ls "$DST/val/images" | wc -l) пар -> $DST"
