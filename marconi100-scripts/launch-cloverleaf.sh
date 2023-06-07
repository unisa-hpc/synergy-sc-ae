#!/bin/bash

configurations=("default")


for conf in $configurations; do
	for nodes in 1 2 4 8 16; do 
		sbatch --array=1-5 --nodes=$nodes --output=cloverleaf_${conf}_ss_${nodes}_%a.log ./cloverleaf-ssjob.sh ${nodes}
		sbatch --array=1-5 --nodes=$nodes --output=cloverleaf_${conf}_ws_${nodes}_%a.log ./cloverleaf-wsjob.sh ${nodes}
	done
done
