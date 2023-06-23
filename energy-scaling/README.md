# Energy scaling
This folder contains all the source code and scripts needed to generate the data for plotting the multi-node energy scaling results.
## MiniWeather 
### Step to reproduce on multi-node with the energy slurm plugin
- Run `source ./extract_features.sh --cxx_compiler=<DPC++ compiler path>` to extract the features for each kernel in MiniWeather
- Run `source ./predict.sh` to generate the predicted frequencies for each kernel and energy metric (es_50, pl_50, min_edp, min_ed2p, default)
- Run `./generate.py --cxx_compiler=<DPC++ compiler path> --cuda_arch=<cuda architecture e.g sm_70> -cxx_flags=<optional cxx compiler flags>` to generate an executable *parallel_for_energyMetric_numNodes* for each combination of target *energyMetric* (es_50, pl_50, min_edp, min_ed2p, default) and *numNodes* (1, 2, 4, 8, 16).
- Complete the script `miniweather-wsjob-freq.sh` with the following element:
    - `#SBATCH --account=<cluster_account_name>`
    - `#SBATCH --partition=<cluster_partition_name>`
    - `#SBATCH --gpus-per-node=<num_of_gpus_per_node>`
    - `#SBATCH --ntasks-per-node=<num_of_process_for_each_node>` it must be equal to the number of gpus per node
    - `#SBATCH --gres=nvgpufreq` to enable the freqeuncy scaling through the slurm plugin
    - `#SBATCH --mail-user=<user_email>`
    - `#SBATCH --exclusive`
- run `source launch-miniweather-freq.sh` script to launch the Miniweather application on 1,2,4,8,16 nodes.

## CloverLeaf
### Step to reproduce on multi-node with the energy slurm plugin
- Run `source ./extract_features.sh --cxx_compiler=<DPC++ compiler path>` to extract the features for each kernel in Cloverleaf
- Run `source ./predict.sh` to generate the predicted frequencies for each kernel and energy metric (es_50, pl_50, min_edp, min_ed2p, default)
- Run `./generate.py --cxx_compiler=<DPC++ compiler path> --cuda_arch=<cuda architecture e.g sm_70> -cxx_flags=<optional cxx compiler flags> ` to generate an executable *clover_leaf_energyMetric* for each  *energyMetric* (es_50, pl_50, min_edp, min_ed2p, default)
- Complete the script `cloverleaf-wsjob-freq.sh` with the following element:
    - `#SBATCH --account=<cluster_account_name>`
    - `#SBATCH --partition=<cluster_partition_name>`
    - `#SBATCH --gpus-per-node=<num_of_gpus_per_node>`
    - `#SBATCH --ntasks-per-node=<num_of_process_for_each_node>` it must be equal to the number of gpus per node
    - `#SBATCH --gres=nvgpufreq` to enable the freqeuncy scaling through the slurm plugin
    - `#SBATCH --mail-user=<user_email>`
    - `#SBATCH --exclusive`
- run `source launch-cloverleaf-freq.sh` script to launch the Cloverleaf application on 1,2,4,8,16 nodes.
