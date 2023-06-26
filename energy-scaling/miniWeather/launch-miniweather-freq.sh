#!/bin/bash

CXX_COMPILER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cxx_compiler=*)
            CXX_COMPILER="${1#*=}"
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

DPCPP_CLANG=$CXX_COMPILER
BIN_DIR=$(dirname $DPCPP_CLANG)
DPCPP_LIB=$BIN_DIR/../lib/
export LD_LIBRARY_PATH=$DPCPP_LIB:$LD_LIBRARY_PATH

configurations=("default" "min_edp" "min_ed2p" "es_50" "pl_50")
mkdir -p logs
for conf in ${configurations[@]}; do
	for nodes in 1 2 4 8 16; do
		sbatch --array=1-10 --nodes=$nodes --output=./logs/miniweather_${conf}_ws_${nodes}_%a.log ./miniweather-wsjob-freq.sh ${nodes} parallel_for_${conf}_${nodes}
	done
done

