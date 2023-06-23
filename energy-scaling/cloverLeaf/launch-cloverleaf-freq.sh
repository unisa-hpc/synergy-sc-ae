#!/bin/bash


# configurations=("max_perf" "min_energy" "min_edp" "min_ed2p")
configurations=("default" "min_edp" "min_ed2p" "es_50" "pl_50")
mkdir -p logs
for conf in ${configurations[@]}; do
	for nodes in 1 2 4 8 16; do 
		sbatch --array=1-10 --nodes=$nodes --output=./logs/cloverleaf_${conf}_ws_${nodes}_%a.log ./cloverleaf-wsjob-freq.sh ${nodes} clover_leaf_${conf}
	done
done
