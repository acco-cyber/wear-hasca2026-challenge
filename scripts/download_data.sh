#!/bin/bash
# WEAR HASCA 2026 - resumable foreground downloader (chunk-safe)
# Usage: timeout 580 bash download_data.sh [wave]
# Wave 1: small files + test set | Wave 2: train inertial | Wave 3: train videomae
cd /home/z/my-project/data
export PATH="/home/z/.local/bin:$PATH"
export KAGGLE_API_TOKEN=KGAT_d6f6367b0cee135abd9dbe0d7cf96388
COMP=3rd-wear-dataset-challenge-hasca-2026

WAVE=${1:-1}

dl() {
  local f="$1" out="$2"
  if [ -f "$out" ]; then echo "SKIP $out"; return 0; fi
  mkdir -p "$(dirname "$out")"
  for i in 1 2 3; do
    if timeout 240 kaggle competitions download -c $COMP -f "$f" -p /tmp/dl_tmp --force 2>/dev/null; then
      # kaggle may save as $f.zip or $f
      local src="/tmp/dl_tmp/$(basename $f)"
      if [ -f "${src}.zip" ]; then
        unzip -o -q "${src}.zip" -d "$(dirname $out)" && rm -f "${src}.zip"
      elif [ -f "$src" ]; then
        mv "$src" "$out"
      fi
      if [ -f "$out" ]; then echo "OK $out"; return 0; fi
    fi
    echo "RETRY$i $f"; sleep 3
  done
  echo "FAIL $f"; return 1
}

mkdir -p /tmp/dl_tmp

if [ "$WAVE" = "1" ]; then
  dl "participant_meta_data.txt" "participant_meta_data.txt"
  dl "sample_submission.csv" "sample_submission.csv"
  dl "test/test_meta_data.csv" "test/test_meta_data.csv"
  dl "test/test_inertial_data.npy" "test/test_inertial_data.npy"
elif [ "$WAVE" = "2" ]; then
  for s in 0 0_2 1 10 11 12 13 14 14_2 15 16 17 18 19 2 20 21 3 4 5 6 7 8 9; do
    dl "train/inertial_feat/sbj_${s}.csv" "train/inertial_feat/sbj_${s}.csv"
  done
elif [ "$WAVE" = "3" ]; then
  for s in 0 0_2 1 10 11 12 13 14 14_2 15 16 17 18 19 2 20 21 3 4 5 6 7 8 9; do
    dl "train/videomae_feat/sbj_${s}.npy" "train/videomae_feat/sbj_${s}.npy"
  done
fi
echo "WAVE $WAVE DONE"
