#!/bin/bash
#SBATCH --account=lig8_dev
#SBATCH --job-name=miniweather_ws_test
#SBATCH --partition=m100_usr_prod
#SBATCH --time=00:05:00
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gres=nvgpufreq
#SBATCH --chdir=/m100/home/userexternal/lcarpent/results/miniweather-logs-freq-new
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdantonio@unisa.it
#SBATCH --exclusive

nodes=$1
executable=$2
proc=$(( $nodes*4 ))

mpirun -n $proc -gpu /m100/home/userexternal/lcarpent/miniWeather-executable-versions/${executable}
