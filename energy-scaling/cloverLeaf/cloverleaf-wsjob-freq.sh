#!/bin/bash
#SBATCH --account=lig8_dev
#SBATCH --job-name=cloverleaf_ws_test
#SBATCH --partition=m100_usr_prod
#SBATCH --time=00:05:00
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gres=nvgpufreq
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdantonio@unisa.it
#SBATCH --exclusive

nodes=$1
executable=$2
proc=$(( $nodes*4 ))
size=$(( $proc*32 ))
mpirun -n $proc -gpu ./executables/${executable} --file ./cloverleafApp/input_files/clover_bm${size}_short.in