#!/bin/bash

provided=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --provided_data*)
            provided=true
            shift
            ;;
        *)
            echo "Invalid argument: $1"
            return 1
            ;;
    esac
done

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

nvsmi_out=$(nvidia-smi  -q | grep "Default Applications Clocks" -A 2 | tail -n +2)
def_core=$(echo $nvsmi_out | awk '{print $3}')

if [ $provided = true ]
then
  python3 $SCRIPT_DIR/models.py provided-data 1312
else
  python3 $SCRIPT_DIR/models.py data $def_core
fi
