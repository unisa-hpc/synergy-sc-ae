#!/bin/bash

configurations=("default")

for conf in $configurations; do
	for nodes in 1 2 4 8 16; do
        	sbatch --array=1-5 --nodes=$nodes --output=miniweather_${conf}_ss_${nodes}_%a.log ./miniweather-ssjob.sh ${nodes}
		sbatch --array=1-5 --nodes=$nodes --output=miniweather_${conf}_ws_${nodes}_%a.log ./miniweather-wsjob.sh ${nodes}
	done
done
