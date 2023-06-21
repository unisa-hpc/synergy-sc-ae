#!/bin/bash

if [ -z "$1" ]
  then
    echo "Provide the absolute path to the DPC++ compiler folder as first argument"
	return
fi

sampling=$2
if [ -z "$2" ]
then
  sampling=1
fi

DPCPP_CLANG=$1/clang++
DPCPP_LIB=$1/../lib/
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export LD_LIBRARY_PATH=$DPCPP_LIB:$LD_LIBRARY_PATH

cmake -DCMAKE_CXX_COMPILER=$DPCPP_CLANG \
  -DSYCL_IMPL=LLVM-CUDA -DSYCL_BENCH_CUDA_ARCH=sm_70 -DENABLED_TIME_EVENT_PROFILING=ON\
  -DENABLED_SYNERGY=ON -DSYNERGY_CUDA_SUPPORT=ON -DSYNERGY_KERNEL_PROFILING=ON -DSYNERGY_SYCL_IMPL=DPC++ \
  -S $SCRIPT_DIR/sycl-bench -B $SCRIPT_DIR/sycl-bench/build
cmake --build $SCRIPT_DIR/sycl-bench/build -j


echo "Running microbenchmarks..."
mkdir -p $SCRIPT_DIR/logs

mem_frequencies=$(nvidia-smi -i 0 --query-supported-clocks=mem --format=csv,noheader,nounits)
core_frequencies=$(nvidia-smi -i 0 --query-supported-clocks=gr --format=csv,noheader,nounits)

runs=5
for mem_freq in $mem_frequencies; do
  i=-1
	for core_freq in $core_frequencies; do
    i=$((i+1))
    if [ $((i % sampling)) != 0 ]
    then
      continue
    fi
    
		$SCRIPT_DIR/sycl-bench/build/ArithLocalMixed --device=gpu --num-runs=$runs --size=65536 --num-iters=20000 --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/ArithLocalMixed_${mem_freq}_${core_freq}.log
    $SCRIPT_DIR/sycl-bench/build/ArithMixedUnitOp  --device=gpu --num-runs=$runs --size=500000 --num-iters=50000 --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/ArithMixedUnitOp_${mem_freq}_${core_freq}.log 
    $SCRIPT_DIR/sycl-bench/build/ArithMixedUnitType --device=gpu --num-runs=$runs --size=500000 --num-iters=25000 --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/ArithMixedUnitType_${mem_freq}_${core_freq}.log
    $SCRIPT_DIR/sycl-bench/build/ArithSingleUnit --device=gpu --num-runs=$runs --size=500000 --num-iters=500000 --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/ArithSingleUnit_${mem_freq}_${core_freq}.log
    $SCRIPT_DIR/sycl-bench/build/GlobalMemory --device=gpu --num-runs=$runs --size=1048576 --num-iters=1000000 --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/GlobalMemory_${mem_freq}_${core_freq}.log
    $SCRIPT_DIR/sycl-bench/build/GlobalMemory2 --device=gpu --num-runs=$runs --size=262144 --num-iters=200 --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/GlobalMemory2_${mem_freq}_${core_freq}.log
    $SCRIPT_DIR/sycl-bench/build/L2Unit --device=gpu --num-runs=$runs --size=1000000 --num-iters=131072 --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/L2Unit_${mem_freq}_${core_freq}.log
    $SCRIPT_DIR/sycl-bench/build/LocalMemory --device=gpu --num-runs=$runs --size=1000000 --num-iters=1000000 --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/LocalMemory_${mem_freq}_${core_freq}.log
    $SCRIPT_DIR/sycl-bench/build/Stencil --device=gpu --num-runs=$runs --size=8192 --num-iters=15 --memory-freq=${mem_freq} --core-freq=${core_freq} > $SCRIPT_DIR/logs/Stencil_${mem_freq}_${core_freq}.log
	done
done

## TODO: sampling always takes into account default freq