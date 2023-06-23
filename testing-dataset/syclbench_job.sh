#!/bin/bash

#SBATCH --account=lig8_dev
#SBATCH --job-name=sycl-bench
#SBATCH --partition=m100_usr_prod
#SBATCH --time 01:00:00
#SBATCH --nodes=1             
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1,nvgpufreq
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdantonio@unisa.it
#SBATCH --exclusive

runs=$1
mem_freq=$2
core_freq=$3

srun --output=./logs/bit_compression_${mem_freq}_${core_freq}.log ./sycl-bench/build/bit_compression --device=gpu --size=524288 --num-iters=100000 --num-runs=$1 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=./logs/black_scholes_${mem_freq}_${core_freq}.log ./sycl-bench/build/black_scholes --device=gpu --size=524288 --num-iters=100000 --num-runs=$1 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=./logs/box_blur_${mem_freq}_${core_freq}.log ./sycl-bench/build/box_blur --device=gpu --num-iters=200 --num-runs=$1 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=./logs/ftle_${mem_freq}_${core_freq}.log ./sycl-bench/build/ftle --device=gpu --size=1048576 --num-iters=500000 --num-runs=$1 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=./logs/geometric_mean_${mem_freq}_${core_freq}.log ./sycl-bench/build/geometric_mean  --device=gpu --size=16384 --num-iters=200000 --num-runs=$1 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=./logs/kmeans_${mem_freq}_${core_freq}.log ./sycl-bench/build/kmeans  --device=gpu --size=32768 --num-iters=500000 --num-runs=$1 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=./logs/knn_${mem_freq}_${core_freq}.log ./sycl-bench/build/knn --device=gpu --size=8192 --num-iters=200 --num-runs=$1 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=./logs/lin_reg_error_${mem_freq}_${core_freq}.log ./sycl-bench/build/lin_reg_error --device=gpu --num-iters=2000 --num-runs=$1 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=./logs/matrix_mul_${mem_freq}_${core_freq}.log ./sycl-bench/build/matrix_mul --device=gpu --size=5000 --num-iters=1 --num-runs=$1 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=./logs/matrix_transpose_local_mem_${mem_freq}_${core_freq}.log ./sycl-bench/build/matrix_transpose_local_mem --device=gpu --size=4096 --num-iters=20000 --num-runs=$1 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=./logs/median_${mem_freq}_${core_freq}.log ./sycl-bench/build/median  --device=gpu --num-iters=2000 --num-runs=$1 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=./logs/merse_twister_${mem_freq}_${core_freq}.log ./sycl-bench/build/merse_twister --device=gpu --size=524288 --num-iters=50000 --num-runs=$1 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=./logs/mol_dyn_${mem_freq}_${core_freq}.log ./sycl-bench/build/mol_dyn  --device=gpu --size=16384 --num-iters=200000 --num-runs=$1 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=./logs/nbody_local_mem_${mem_freq}_${core_freq}.log ./sycl-bench/build/nbody_local_mem --device=gpu --size=8192 --num-iters=500 --num-runs=$1 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=./logs/scalar_prod_${mem_freq}_${core_freq}.log ./sycl-bench/build/scalar_prod --device=gpu --size=8192 --num-iters=200000 --num-runs=$1 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=./logs/sinewave_${mem_freq}_${core_freq}.log ./sycl-bench/build/sinewave --device=gpu --size=8192 --num-iters=10000 --num-runs=$1 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=./logs/sobel_${mem_freq}_${core_freq}.log ./sycl-bench/build/sobel  --device=gpu --num-iters=2000 --num-runs=$1 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=./logs/vec_add_${mem_freq}_${core_freq}.log ./sycl-bench/build/vec_add  --device=gpu --size=1048576 --num-iters=100000 --num-runs=$1 --memory-freq=${mem_freq} --core-freq=${core_freq}