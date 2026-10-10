#!/usr/bin/env bash
set -euo pipefail

if ! command -v ncu >/dev/null 2>&1; then
  echo 'ERROR: ncu is not on PATH' >&2
  exit 1
fi
if [[ ! -x ./spmv_profile ]]; then
  echo 'ERROR: build first: nvcc -O3 -lineinfo -std=c++17 -arch=sm_89 spmv_profile.cu -o spmv_profile' >&2
  exit 1
fi

NCU="$(command -v ncu)"
mkdir -p ncu_results

for case_name in Uniform Extreme; do
  for algo in csr jds; do
    label="${case_name}_${algo}"
    echo "========== profiling ${label} =========="
    sudo "$NCU" \
      --profile-from-start off \
      --cache-control none \
      --launch-count 1 \
      --page details \
      --section SpeedOfLight \
      --section MemoryWorkloadAnalysis \
      --section LaunchStats \
      --section Occupancy \
      --section WarpStateStats \
      --export "ncu_results/${label}" \
      --force-overwrite \
      ./spmv_profile --case "$case_name" --profile "$algo" \
      2>&1 | tee "ncu_results/${label}.txt"
  done
done

echo 'Done: see ncu_results/*.txt and ncu_results/*.ncu-rep'