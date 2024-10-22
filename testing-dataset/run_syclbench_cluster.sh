#!/bin/bash

CXX_COMPILER=""
CXX_FLAGS=""
runs=5

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
        --num_runs=*)
            runs="${1#*=}"
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

mkdir -p $SCRIPT_DIR/logs

mem_frequencies=$(nvidia-smi -i 0 --query-supported-clocks=mem --format=csv,noheader,nounits)
core_frequencies=$(nvidia-smi -i 0 --query-supported-clocks=gr --format=csv,noheader,nounits)

nvsmi_out=$(nvidia-smi  -q | grep "Default Applications Clocks" -A 2 | tail -n +2)
def_core=$(echo $nvsmi_out | awk '{print $3}')
def_mem=$(echo $nvsmi_out | awk '{print $7}')

echo "Running SYCL-Bench..."

for core in $core_frequencies; do
    echo "Running benchmarks for frequency $core_freq"
    sbatch ${SCRIPT_DIR}/syclbench_job.sh $runs $def_mem $core
done