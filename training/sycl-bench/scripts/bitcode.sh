#!/bin/bash

alias sycl="clang++ -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -fsycl-device-only -I../include/  -I../SYnergy/include -D __ENABLED_SYNERGY -D SYNERGY_ENABLE_PROFILING"

mkdir -p bitcode
for file in ../single-kernel/*.cpp; do
  echo "SYCL source: $file"
  base_name=$(basename $file)
  name="${base_name%.*}"

  sycl $file -o bitcode/$name.bc
  # llvm-dis $name.bc
done
