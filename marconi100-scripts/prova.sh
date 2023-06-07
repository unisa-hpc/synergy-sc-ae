#!/bin/bash
#SBATCH --account=lig8_dev
#SBATCH --job-name=clover_leaf_test
#SBATCH --partition=m100_usr_prod
#SBATCH --time 00:05:00                 # format: HH:MM:SS
#SBATCH --nodes=1                       # 1 node
#SBATCH --ntasks=2                      # tasks out of 128
#SBATCH --gres=gpu:2
#SBATCH --qos=m100_qos_dbg
#SBATCH --chdir=/m100/home/userexternal/lcarpent/cloverleaf_sycl/build
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdantonio@unisa.it
mpirun -n 2 -gpu ./clover_leaf --file ../InputDecks/clover_bm2_short.in
