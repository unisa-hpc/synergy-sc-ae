#!/bin/bash
#SBATCH --account=
#SBATCH --partition=
#SBATCH --gpus-per-node=
#SBATCH --ntasks-per-node=
#SBATCH --mail-user=
#SBATCH --job-name=miniweather_ws_test
#SBATCH --time=00:05:00
#SBATCH --gres=nvgpufreq
#SBATCH --mail-type=ALL
#SBATCH --exclusive

nodes=$1
executable=$2
proc=$(( $nodes*4 ))

mpirun -n $proc -gpu ./executables/${executable}
