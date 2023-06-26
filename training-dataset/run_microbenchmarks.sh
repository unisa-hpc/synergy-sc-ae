#!/bin/bash
CXX_COMPILER=""
CXX_FLAGS=""
sampling=1
cuda_arch=""
# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cxx_compiler=*)
            CXX_COMPILER="${1#*=}"
            shift
            ;;
        --cxx_flags=*)
            CXX_FLAGS="${1#*=}"
            shift
            ;;
        --cuda_arch=*)
            cuda_arch="${1#*=}"
            shift
            ;;
        --freq_sampling=*)
            sampling="${1#*=}"
            shift
            ;;
        *)
            echo "Invalid argument: $1"
              return 1 2>/dev/null
              exit 1
            ;;
    esac
done

if [ -z "$CXX_COMPILER" ]
  then
    echo "Provide the absolute path to the DPC++ compiler as --cxx_compiler argument"
    return 1 2>/dev/null
    exit 1
fi

if [ -z "$cuda_arch" ]
  then
    echo "Provide the cuda architecture as --cuda_arch argument (e.g: sm_70)"
    return 1 2>/dev/null
    exit 1
fi


DPCPP_CLANG=$CXX_COMPILER
BIN_DIR=$(dirname $DPCPP_CLANG)
DPCPP_LIB=$BIN_DIR/../lib/
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export LD_LIBRARY_PATH=$DPCPP_LIB:$LD_LIBRARY_PATH
cmake -DCMAKE_CXX_COMPILER=$DPCPP_CLANG \
  -DCMAKE_CXX_FLAGS="-Wno-unknown-cuda-version -Wno-linker-warnings -Wno-sycl-target $CXX_FLAGS" \
  -DSYCL_IMPL=LLVM-CUDA -DSYCL_BENCH_CUDA_ARCH=$cuda_arch -DENABLED_TIME_EVENT_PROFILING=ON \
  -DENABLED_SYNERGY=ON -DSYNERGY_CUDA_SUPPORT=ON -DSYNERGY_KERNEL_PROFILING=ON -DSYNERGY_SYCL_IMPL=DPC++ \
  -S $SCRIPT_DIR/sycl-bench -B $SCRIPT_DIR/sycl-bench/build
cmake --build $SCRIPT_DIR/sycl-bench/build -j


echo "Running micro-benchmarks..."
mkdir -p $SCRIPT_DIR/logs

mem_frequencies=$(nvidia-smi -i 0 --query-supported-clocks=mem --format=csv,noheader,nounits)
core_frequencies=$(nvidia-smi -i 0 --query-supported-clocks=gr --format=csv,noheader,nounits)

nvsmi_out=$(nvidia-smi  -q | grep "Default Applications Clocks" -A 2 | tail -n +2)
def_core=$(echo $nvsmi_out | awk '{print $3}')
def_mem=$(echo $nvsmi_out | awk '{print $7}')

sampled_freq=()
i=-1
for core_freq in $core_frequencies; do
  i=$((i+1))
  if [ $((i % sampling)) != 0 ]
  then
    continue
  fi
  sampled_freq+=($core_freq)
done
sampled_freq+=($def_core)

runs=5
mem_freq=$def_mem
for core_freq in "${sampled_freq[@]}"; do
    echo "Running micro-benchmarks for frequency $core_freq"
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