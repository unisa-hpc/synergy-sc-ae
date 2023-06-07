#!/bin/bash

core_freq=$(nvidia-smi -i 0 --query-supported-clocks=gr --format=csv,noheader,nounits)
vers=(20 40 80 120)

for v in ${vers[@]}; do
	for core in $core_freq; do
		sbatch --array=1-10 --output=ligen_${v}_${core}_%a.log ./ligen-job.sh 877 $core $v
	done
done
