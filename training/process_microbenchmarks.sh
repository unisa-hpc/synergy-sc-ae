#!/bin/bash

logs_folder=logs

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "Parsing logs..."
python3 $SCRIPT_DIR/postprocess/parse.py $SCRIPT_DIR/$logs_folder $SCRIPT_DIR/parsed
python3 $SCRIPT_DIR/postprocess/metrics.py $SCRIPT_DIR/parsed $SCRIPT_DIR/parsed_metrics

echo "Creating dataset for training..."
python3 $SCRIPT_DIR/postprocess/merge.py $SCRIPT_DIR/parsed_metrics $SCRIPT_DIR/micro/features-normalized $SCRIPT_DIR/../validation/training-data

rm -r $SCRIPT_DIR/parsed
rm -r $SCRIPT_DIR/parsed_metrics