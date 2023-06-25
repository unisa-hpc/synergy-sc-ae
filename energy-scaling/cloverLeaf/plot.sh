#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

provided=false
gpus_per_node=4
while [[ $# -gt 0 ]]; do
    case "$1" in
        --provided_data*)
            provided=true
            shift
            ;;
        --ngpus=*)
            gpus_per_node="${1#*=}"
            shift
            ;;
        *)
            echo "Invalid argument: $1"
            return 1 2>/dev/null
            exit 1
            ;;
    esac
done

if [ $provided = true ]
then  
  python3 $SCRIPT_DIR/parse.py provided-logs 4
  python3 $SCRIPT_DIR/plot.py 4
else  
  python3 $SCRIPT_DIR/parse.py logs $gpus_per_node
  python3 $SCRIPT_DIR/plot.py $gpus_per_node
fi
