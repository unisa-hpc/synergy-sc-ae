#!/bin/bash

logs_folder=logs

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

nvsmi_out=$(nvidia-smi  -q | grep "Default Applications Clocks" -A 2 | tail -n +2)
def_core=$(echo $nvsmi_out | awk '{print $3}')
def_mem=$(echo $nvsmi_out | awk '{print $7}')

echo "Parsing logs..."
python3 $SCRIPT_DIR/postprocess/parse.py $SCRIPT_DIR/$logs_folder $SCRIPT_DIR/parsed
python3 $SCRIPT_DIR/postprocess/metrics.py $SCRIPT_DIR/parsed $SCRIPT_DIR/parsed_metrics

echo "Merging data..."
python3 $SCRIPT_DIR/postprocess/merge.py $SCRIPT_DIR/parsed_metrics $SCRIPT_DIR/features-normalized $SCRIPT_DIR/merged-normalized

rm -r $SCRIPT_DIR/parsed
rm -r $SCRIPT_DIR/parsed_metrics

echo "Creating dataset for validation..."
python3 $SCRIPT_DIR/postprocess/separate.py $SCRIPT_DIR/merged-normalized $SCRIPT_DIR/../models-validation/data/testing-data
python3 $SCRIPT_DIR/postprocess/extract.py $SCRIPT_DIR/merged-normalized $SCRIPT_DIR/../models-validation/data/testing-data $def_mem $def_core

echo "Plotting characterization of benchmarks..."
python3 $SCRIPT_DIR/postprocess/plot.py $SCRIPT_DIR/merged-normalized/ $SCRIPT_DIR/plots $def_mem $def_core
