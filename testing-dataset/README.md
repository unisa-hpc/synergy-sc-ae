# Testing dataset and characterization
This folder contains all the source code and scripts needed to generate:
  - the dataset on which the models will be validated
  - the multi-objective characterization plots of the paper

## Folder structure
- `passes` contains the source code of the compiler passes used to extract the code features
- `sycl-bench` contains the source code of the benchmarks used for energy and runtime profiling
- `postprocess` contains the scripts to parse and generate the dataset for validation, and the scripts to generate the multi-objective characterization plots

> ### Reproduce on a single node (with **root access**)
1. Run `source extract_features.sh --cxx_compiler=<DPC++ compiler path>` to extract static code features from the benchmarks.
    - the DPC++ compiler path must be the absolute path to the DPC++ compiler
    - the output features will be in the `features-*` subfolders

2. Run `sudo ./run_syclbench.sh --cxx_compiler=<DPC++ compiler path> --cuda_arch=<CUDA architecture e.g. sm_70>` to run the SYCL-Bench benchmarks and save the logs for parsing.
    - Optional parameters:
      - `--cxx_flags` to pass other options to the DPC++ compiler
      - `--freq_sampling` to reduce the number of tested core frequencies, the script will test one frequency every `freq_sampling`, e.g. passing 2 will halve the number of tested frequencies. **Important: if you use the `--freq_sampling` command-line argument, the same sampling value must be used for generating both the training and testing datasets.**
    - The logs will be in the `logs/` subfolder

3. Run `source process_syclbench.sh` to process the logs and create the data for the validation of the model and the plots of the characterization section.
    - the dataset will be in the `/models-validation/data/testing-data` folder
    - the characterization plots will be in the `/testing-dataset/plots` subfolder

> ### Reproduce on a cluster (with NVGPUFREQ SLURM plugin)
Make sure that your current working directory is the folder containing this README.md file: `testing-dataset`.

1. Run `source extract_features.sh --cxx_compiler=<DPC++ compiler path>` to extract static code features from the benchmarks.
    - the DPC++ compiler path must be the absolute path to the DPC++ compiler
    - the output features will be in the `features-*` subfolders

Before running the `run_syclbench_cluster.sh` script complete the following missing data in the `syclbench_job.sh` script:
  - `#SBATCH --account=<cluster_account_name>`
  - `#SBATCH --partition=<cluster_partition_name>`
  - `#SBATCH --mail-user=<user_email>`

2. Run `source run_syclbench_cluster.sh --cxx_compiler=<DPC++ compiler path> --cuda_arch=<CUDA architecture e.g. sm_70>` to run the SYCL-Bench benchmarks and save the logs for parsing.
    - Optional parameters:
      - `--cxx_flags` to pass other options to the DPC++ compiler
      - `--freq_sampling` to reduce the number of tested core frequencies, the script will test one frequency every `freq_sampling`, e.g. passing 2 will halve the number of tested frequencies. **Important: if you use the `--freq_sampling` command-line argument, the same sampling value must be used for generating both the training and testing datasets**
    - the logs will be in the `logs/` subfolder

3. Run `source process_syclbench.sh` to process the logs and create the data for the validation of the model and the plots of the characterization section.
    - the dataset will be in the `/models-validation/data/testing-data` folder
    - the characterization plots will be in the `/testing-dataset/plots` subfolder

### Next step
Go to the `/models-validation` folder.