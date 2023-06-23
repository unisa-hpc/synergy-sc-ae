#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "core-freq" > $SCRIPT_DIR/gpu-freq.csv
echo `nvidia-smi --query-supported-clocks=gr --format=csv,noheader,nounits` | tr " " "\n" >> $SCRIPT_DIR/gpu-freq.csv

nvsmi_out=$(nvidia-smi  -q | grep "Default Applications Clocks" -A 2 | tail -n +2)
def_core=$(echo $nvsmi_out | awk '{print $3}')

# python3 $SCRIPT_DIR/cloverLeaf/predict.py $def_core
python3 $SCRIPT_DIR/miniWeather/predict.py $def_core
