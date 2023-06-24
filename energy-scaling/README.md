# Energy scaling
This folder contains all the source code and scripts needed to generate the data for plotting the multi-node energy scaling results.

## Folder structure
- `passes` contains the source code of the compiler passes used to extract the code features
- `cloverLeaf` contains the CloverLeaf application source code and application-specific scripts for data parsing processing
- `miniWeather `contains the MiniWeather application source code and application-specific scripts for data parsing processing

## MiniWeather 
Note: if you extracted the features and predicted the frequencies for CloverLeaf, then you can skip steps 1 and 2.

> ### Reproduce on a multi-GPU cluster (with NVGPUFREQ SLURM plugin)
Note: if you want to obtain the results using the pre-built data, just run step 3 and 7.

1. Run `source extract_features.sh --cxx_compiler=<DPC++ compiler path>` to extract static code features for each kernel in MiniWeather.
    - the DPC++ compiler path must be the absolute path to the DPC++ compiler
    - the output features will be in the `miniWeather/features-*` subfolders
2. Run `source predict.sh` to generate the predicted frequencies for each kernel and energy metric (`es_50`, `pl_50`, `min_edp`, `min_ed2p`, `default`).
    - the output predictions will be in the `miniWeather/prediction` subfolder.
3. Run `cd miniWeather`.
4. Run `python3 generate.py --cxx_compiler=<DPC++ compiler path> --cuda_arch=<CUDA architecture e.g. sm_70>` to compile the MiniWeather application for different energy metrics.
    - Optional parameters:
      - `--cxx_flags` to pass other options to the DPC++ compiler
      - `--sycl_flags` to pass other compilation options for the YAKL library
    - the binary files for each combination of energy metric (`es_50`, `pl_50`, `min_edp`, `min_ed2p`, `default`) and number of nodes (1, 2, 4, 8, 16), will be in the `miniWeather/executables` folder.
5. Complete the following missing data in the `miniweather-wsjob-freq.sh` script:
    - `#SBATCH --account=<cluster_account_name>`
    - `#SBATCH --partition=<cluster_partition_name>`
    - `#SBATCH --gpus-per-node=<num_of_gpus_per_node>`
    - `#SBATCH --ntasks-per-node=<num_of_processes_per_node>` (it must be equal to the number of gpus per node)
    - `#SBATCH --mail-user=<user_email>`
6. Run `source launch-miniweather-freq.sh` script to launch the MiniWeather application on 1, 2, 4, 8 and 16 nodes.
7. Run `source plot.sh` to parse the logs and generate the plots.
    - Optional parameters:
      - `--provided_data` must be passed as an argument if you want to use the pre-built dataset
    - the plots will be in the `/energy-scaling/plots` folder

## CloverLeaf
Note: if you extracted the features and predicted the frequencies for MiniWeather, then you can skip steps 1 and 2.
> ### Reproduce on a multi-GPU cluster (with NVGPUFREQ SLURM plugin)
Note: if you want to obtain the results using the pre-built data, just run step 3 and 7.

1. Run `source extract_features.sh --cxx_compiler=<DPC++ compiler path>` to extract static code features for each kernel in CloverLeaf.
    - the DPC++ compiler path must be the absolute path to the DPC++ compiler
    - the output features will be in the `cloverLeaf/features-*` subfolders
2. Run `source predict.sh` to generate the predicted frequencies for each kernel and energy metric (`es_50`, `pl_50`, `min_edp`, `min_ed2p`, `default`).
    - the output predictions will be in the `cloverLeaf/prediction` subfolder.
3. Run `cd cloverLeaf`.
4. Run `python3 generate.py --cxx_compiler=<DPC++ compiler path> --cuda_arch=<CUDA architecture e.g. sm_70>` to compile the CloverLeaf application for different energy metrics.
    - Optional parameters:
      - `--cxx_flags` to pass other options to the DPC++ compiler
    - the binary files for each energy metric (`es_50`, `pl_50`, `min_edp`, `min_ed2p`, `default`), will be in the `cloverLeaf/executables` folder.
5. Complete the following missing data in the `cloverleaf-wsjob-freq.sh` script:
    - `#SBATCH --account=<cluster_account_name>`
    - `#SBATCH --partition=<cluster_partition_name>`
    - `#SBATCH --gpus-per-node=<num_of_gpus_per_node>`
    - `#SBATCH --ntasks-per-node=<num_of_processes_per_node>` (it must be equal to the number of gpus per node)
    - `#SBATCH --mail-user=<user_email>`
6. Run `source launch-cloverleaf-freq.sh` script to launch the CloverLeaf application on 1, 2, 4, 8 and 16 nodes.
7. Run `source plot.sh` to parse the logs and generate the plots.
    - Optional parameters:
      - `--provided_data` must be passed as an argument if you want to use the pre-built dataset
    - the plots will be in the `/energy-scaling/plots` folder
