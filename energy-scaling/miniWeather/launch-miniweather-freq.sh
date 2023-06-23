#!/bin/bash

configurations=("default" "min_edp" "min_ed2p" "es_50" "pl_50")
mkdir -p logs
for conf in ${configurations[@]}; do
	for nodes in 1 2 4 8 16; do
		sbatch --array=1-10 --nodes=$nodes --output=./logs/miniweather_${conf}_ws_${nodes}_%a.log ./miniweather-wsjob-freq.sh ${nodes} parallel_for_${conf}_${nodes}
	done
done

