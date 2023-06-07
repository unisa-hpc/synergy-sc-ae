#!/bin/bash
#SBATCH --account=lig8_dev
#SBATCH --job-name=ligen_energy
#SBATCH --partition=m100_usr_prod
#SBATCH --time=00:00:30
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=nvgpufreq
#SBATCH --chdir=/m100/home/userexternal/lcarpent/results/ligen-logs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdantonio@unisa.it
#SBATCH --exclusive

mem_freq=$1
core_freq=$2
vers=$3

cat /m100/home/userexternal/lcarpent/ligen/chemlib/ligen_${vers}.mol2 | /m100/home/userexternal/lcarpent/ligen/build/one-purpose-apps/ligen-dock-and-score -p /m100/home/userexternal/lcarpent/ligen/chemlib/pocket.ini -s wang98 --sycl=gpu -m ${mem_freq} -c ${core_freq}
