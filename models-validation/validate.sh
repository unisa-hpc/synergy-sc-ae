#!/bin/bash

if [ -z "$1" ]
  then
    echo "Provide the name of the dataset folder as first argument"
	return
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

nvsmi_out=$(nvidia-smi  -q | grep "Default Applications Clocks" -A 2 | tail -n +2)
def_core=$(echo $nvsmi_out | awk '{print $3}')

python3 $SCRIPT_DIR/models.py $1 $def_core