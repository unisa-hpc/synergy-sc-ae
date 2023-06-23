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
            return 1
            ;;
    esac
done

if [ $provided = true ]
then
  python3 $SCRIPT_DIR/miniWeather/parse.py provided-logs
  python3 $SCRIPT_DIR/miniWeather/plot.py
  
  python3 $SCRIPT_DIR/cloverLeaf/parse.py provided-logs
  python3 $SCRIPT_DIR/cloverLeaf/plot.py
else
  python3 $SCRIPT_DIR/miniWeather/parse.py logs
  python3 $SCRIPT_DIR/miniWeather/plot.py
  
  python3 $SCRIPT_DIR/cloverLeaf/parse.py logs
  python3 $SCRIPT_DIR/cloverLeaf/plot.py
fi
