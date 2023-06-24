#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

provided=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --provided_data*)
            provided=true
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
  python3 $SCRIPT_DIR/parse.py provided-logs
  python3 $SCRIPT_DIR/plot.py
else
  python3 $SCRIPT_DIR/parse.py logs
  python3 $SCRIPT_DIR/plot.py
fi
