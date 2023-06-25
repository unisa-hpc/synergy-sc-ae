#!/bin/bash

provided=false
logs_folder=logs
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

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ $provided = true ]
then
  logs_folder=provided-logs
  def_core=1312
  def_mem=877
else 
  nvsmi_out=$(nvidia-smi  -q | grep "Default Applications Clocks" -A 2 | tail -n +2)
  def_core=$(echo $nvsmi_out | awk '{print $3}')
  def_mem=$(echo $nvsmi_out | awk '{print $7}')
fi

echo "Parsing logs..."
python3 $SCRIPT_DIR/postprocess/parse.py $SCRIPT_DIR/$logs_folder $SCRIPT_DIR/parsed
python3 $SCRIPT_DIR/postprocess/metrics.py $SCRIPT_DIR/parsed $SCRIPT_DIR/parsed_metrics

echo "Merging data..."
python3 $SCRIPT_DIR/postprocess/merge.py $SCRIPT_DIR/parsed_metrics $SCRIPT_DIR/features-normalized $SCRIPT_DIR/merged-normalized

rm -r $SCRIPT_DIR/parsed
rm -r $SCRIPT_DIR/parsed_metrics

if [ $provided = false ]
then
  echo "Creating dataset for validation..."
  python3 $SCRIPT_DIR/postprocess/separate.py $SCRIPT_DIR/merged-normalized $SCRIPT_DIR/../models-validation/data/testing-data
  python3 $SCRIPT_DIR/postprocess/extract.py $SCRIPT_DIR/merged-normalized $SCRIPT_DIR/../models-validation/data/testing-data $def_mem $def_core
fi

echo "Plotting characterization of benchmarks..."
python3 $SCRIPT_DIR/postprocess/plot.py $SCRIPT_DIR/merged-normalized/ $SCRIPT_DIR/plots $def_mem $def_core
