#!/bin/bash

#SBATCH --account=lig8_dev
#SBATCH --job-name=micro-bench
#SBATCH --partition=m100_usr_prod
#SBATCH --time 00:45:00                 # format: HH:MM:SS
#SBATCH --nodes=1                       
#SBATCH --ntasks=1	
#SBATCH --gres=gpu:1,nvgpufreq
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdantonio@unisa.it
#SBATCH --exclusive

runs=$1
mem_freq=$2
core_freq=$3
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

srun --output=$SCRIPT_DIR/logs/ArithLocalMixed_${mem_freq}_${core_freq}.log $SCRIPT_DIR/sycl-bench/build/ArithLocalMixed --device=gpu --num-runs=$runs --size=65536 --num-iters=20000 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=$SCRIPT_DIR/logs/ArithMixedUnitOp_${mem_freq}_${core_freq}.log $SCRIPT_DIR/sycl-bench/build/ArithMixedUnitOp --device=gpu --num-runs=$runs --size=500000 --num-iters=50000 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=$SCRIPT_DIR/logs/ArithMixedUnitType_${mem_freq}_${core_freq}.log $SCRIPT_DIR/sycl-bench/build/ArithMixedUnitType --device=gpu --num-runs=$runs --size=500000 --num-iters=25000 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=$SCRIPT_DIR/logs/ArithSingleUnit_${mem_freq}_${core_freq}.log $SCRIPT_DIR/sycl-bench/build/ArithSingleUnit --device=gpu --num-runs=$runs --size=500000 --num-iters=500000 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=$SCRIPT_DIR/logs/GlobalMemory_${mem_freq}_${core_freq}.log $SCRIPT_DIR/sycl-bench/build/GlobalMemory --device=gpu --num-runs=$runs --size=1048576 --num-iters=1000000 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=$SCRIPT_DIR/logs/L2Unit_${mem_freq}_${core_freq}.log $SCRIPT_DIR/sycl-bench/build/L2Unit --device=gpu --num-runs=$runs --size=1000000 --num-iters=131072 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=$SCRIPT_DIR/logs/LocalMemory_${mem_freq}_${core_freq}.log $SCRIPT_DIR/sycl-bench/build/LocalMemory --device=gpu --num-runs=$runs --size=1000000 --num-iters=1000000 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=$SCRIPT_DIR/logs/GlobalMemory2_${mem_freq}_${core_freq}.log $SCRIPT_DIR/sycl-bench/build/GlobalMemory2 --device=gpu --num-runs=$runs --size=262144 --num-iters=200 --memory-freq=${mem_freq} --core-freq=${core_freq}
srun --output=$SCRIPT_DIR/logs/Stencil_${mem_freq}_${core_freq}.log $SCRIPT_DIR/sycl-bench/build/Stencil --device=gpu --num-runs=$runs --size=8192 --num-iters=15 --memory-freq=${mem_freq} --core-freq=${core_freq}
