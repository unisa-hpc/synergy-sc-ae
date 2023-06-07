#!/bin/bash

if [ -z "$1" ]
  then
    echo "First argument should be prefix folder"
	return
fi

folder=$1 

if [ -z "$2" ]
  then
    echo "Second argument should be 'micro' or 'sycl_bench'"
	return
fi

type=$2

echo "Parsing logs..."
./logs_parser.py $folder/logs/ $folder/parsed/
echo "Computing energy metrics..."
./add_energy_metrics.py $folder/parsed/ $folder/parsed-edp/

echo "Merging with features..."
merge_script="./merge_${type}.py"
$merge_script $folder/parsed-edp/ $folder/features-count/ $folder/merged-count/
$merge_script $folder/parsed-edp/ $folder/features-normalized/ $folder/merged-normalized/ norm

echo "Generating multi-objective plots..."
./multi_obj_plot.py $folder/merged-count/ $folder/plots-multiobj/
echo "Generating EDP and ED2P plots..."
./edp_ed2p_plot.py $folder/merged-count/ $folder/plots-edp/ $folder/plots-ed2p/