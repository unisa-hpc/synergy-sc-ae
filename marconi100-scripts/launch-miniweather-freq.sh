#!/bin/bash

 # configurations=("default" "max_perf" "min_energy" "min_edp" "min_ed2p")
 configurations=("default" "max_perf" "min_energy" "min_edp" "min_ed2p" "es_25" "es_50" "es_75" "pl_25" "pl_50" "pl_75")
for conf in ${configurations[@]}; do
	for nodes in 1 2 4 8 16; do
        	#sbatch --array=1-2 --nodes=$nodes --output=miniweather_${conf}_ss_${nodes}_%a.log ./miniweather-ssjob-freq.sh ${nodes} parallel_for_${conf}_1
		sbatch --array=1-10 --nodes=$nodes --output=miniweather_${conf}_ws_${nodes}_%a.log ./miniweather-wsjob-freq.sh ${nodes} parallel_for_${conf}_${nodes}
	done
done

