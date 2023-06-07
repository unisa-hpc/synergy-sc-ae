#!/bin/bash

mem_freq=$(nvidia-smi -i 0 --query-supported-clocks=mem --format=csv,noheader,nounits)
core_freq=$(nvidia-smi -i 0 --query-supported-clocks=gr --format=csv,noheader,nounits)

for mem in $mem_freq; do 
	for core in $core_freq; do
		sbatch ./bench-job.sh 5 $mem $core
	done
done
