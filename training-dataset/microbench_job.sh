#!/bin/bash

#SBATCH --account=
#SBATCH --partition=
#SBATCH --mail-user=
#SBATCH --job-name=micro-bench
#SBATCH --time 00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,nvgpufreq
#SBATCH --mail-type=ALL
#SBATCH --exclusive

runs=$1
mem_freq=$2
core_freq=$3

srun --output=./logs/ArithLocalMixed_${mem_freq}_${core_freq}.log ./sycl-bench/build/ArithLocalMixed --device=gpu --num-runs=$runs --size=65536 --num-iters=20000 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=./logs/ArithMixedUnitOp_${mem_freq}_${core_freq}.log ./sycl-bench/build/ArithMixedUnitOp --device=gpu --num-runs=$runs --size=500000 --num-iters=50000 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=./logs/ArithMixedUnitType_${mem_freq}_${core_freq}.log ./sycl-bench/build/ArithMixedUnitType --device=gpu --num-runs=$runs --size=500000 --num-iters=25000 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=./logs/ArithSingleUnit_${mem_freq}_${core_freq}.log ./sycl-bench/build/ArithSingleUnit --device=gpu --num-runs=$runs --size=500000 --num-iters=500000 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=./logs/GlobalMemory_${mem_freq}_${core_freq}.log ./sycl-bench/build/GlobalMemory --device=gpu --num-runs=$runs --size=1048576 --num-iters=1000000 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=./logs/L2Unit_${mem_freq}_${core_freq}.log ./sycl-bench/build/L2Unit --device=gpu --num-runs=$runs --size=1000000 --num-iters=131072 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=./logs/LocalMemory_${mem_freq}_${core_freq}.log ./sycl-bench/build/LocalMemory --device=gpu --num-runs=$runs --size=1000000 --num-iters=1000000 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=./logs/GlobalMemory2_${mem_freq}_${core_freq}.log ./sycl-bench/build/GlobalMemory2 --device=gpu --num-runs=$runs --size=262144 --num-iters=200 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=./logs/Stencil_${mem_freq}_${core_freq}.log ./sycl-bench/build/Stencil --device=gpu --num-runs=$runs --size=8192 --num-iters=15 --memory-freq=${mem_freq} --core-freq=${core_freq}
