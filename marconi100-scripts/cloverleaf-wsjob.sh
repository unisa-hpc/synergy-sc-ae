#!/bin/bash
#SBATCH --account=lig8_dev
#SBATCH --job-name=cloverleaf_ws_test
#SBATCH --partition=m100_usr_prod
#SBATCH --time=00:10:00
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --chdir=/m100/home/userexternal/lcarpent/results/cloverleaf-logs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdantonio@unisa.it
#SBATCH --exclusive

nodes=$1
proc=$(( $nodes*4 ))
size=$(( $proc*32 ))
mpirun -n $proc -gpu /m100/home/userexternal/lcarpent/cloverleaf_sycl_default/build/clover_leaf --file /m100/home/userexternal/lcarpent/cloverleaf_sycl_default/InputDecks/clover_bm${size}_short.in
