#!/bin/bash
#SBATCH --account=
#SBATCH --partition=
#SBATCH --gpus-per-node=
#SBATCH --ntasks-per-node=
#SBATCH --mail-user=
#SBATCH --job-name=cloverleaf_ws_test
#SBATCH --time=00:05:00
#SBATCH --gres=nvgpufreq
#SBATCH --mail-type=ALL
#SBATCH --exclusive

nodes=$1
executable=$2
proc=$(( $nodes*4 ))
size=$(( $proc*32 ))
mpirun -n $proc -gpu ./executables/${executable} --file ./cloverleafApp/input_files/clover_bm${size}_short.in
