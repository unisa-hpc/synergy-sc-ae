# Training dataset
This folder contains all the source code and scripts needed to generate the models' training dataset.

The expected execution time of all micro-benchmarks for a single frequency configuration is around 20 minutes.
The `--freq_sampling` argument can reduce the total execution time by reducing the number of tested core frequencies.

## Folder structure
- `micro` contains the source code of the micro-benchmarks and the generated bitcode and output features
- `passes` contains the source code of the compiler passes used to extract the code features
- `sycl-bench` contains the source code of the micro-benchmarks embedded into the SYCL-Bench infrastructure used for energy and runtime profiling
- `postprocess` contains the scripts to parse and generate the dataset for training

> ### Reproduce on a single node (with **root access**)
1. Run `source extract_features.sh --cxx_compiler=<DPC++ compiler path>` to extract static code features from the micro-benchmarks.
    - the DPC++ compiler path must be the absolute path to the DPC++ compiler
    - the output features will be in the `micro/features-*` subfolders

2. Run `sudo ./run_microbenchmarks.sh --cxx_compiler=<DPC++ compiler path> --cuda_arch=<CUDA architecture e.g. sm_70>` to run the micro-benchmarks and save the logs for parsing.
    - the DPC++ compiler path must be the absolute path to the DPC++ compiler
    - Optional parameters:
      - `--cxx_flags` to pass other options to the DPC++ compiler
      - `--freq_sampling` to reduce the number of tested core frequencies, the script will test one frequency every `freq_sampling`, e.g. passing 2 will halve the number of tested frequencies.
    - The logs will be in the `logs/` subfolder

3. Run `source process_microbenchmarks.sh` to parse the logs and create the data for the training of the model.
    - the dataset will be in the `/models-validation/data/training-data` folder

**Note: if for any reason you stop the micro-benchmarks while running, you should also cleanup the created `logs/` folder.**

> ### Reproduce on a cluster (with NVGPUFREQ SLURM plugin)
Make sure that your current working directory is the folder containing this README.md file: `training-dataset`.

1. Run `source extract_features.sh --cxx_compiler=<DPC++ compiler path>` to extract static code features from the micro-benchmarks.
    - the DPC++ compiler path must be the absolute path to the DPC++ compiler
    - the output features will be in the `micro/features-*` subfolders

Before running the `run_microbenchmarks_cluster.sh` script complete the following missing data in the `microbench_job.sh` script:
  - `#SBATCH --account=<cluster_account_name>`
  - `#SBATCH --partition=<cluster_partition_name>`
  - `#SBATCH --mail-user=<user_email>`

2. Run `source run_microbenchmarks_cluster.sh --cxx_compiler=<DPC++ compiler path> --cuda_arch=<CUDA architecture e.g. sm_70>` to run the micro-benchmarks and save the logs for parsing.
    - Optional parameters:
      - `--cxx_flags` to pass other options to the DPC++ compiler
      - `--freq_sampling` to reduce the number of tested core frequencies, the script will test one frequency every `freq_sampling`, e.g. passing 2 will halve the number of tested frequencies.
    - the logs will be in the `logs/` subfolder

3. Run `source process_microbenchmarks.sh` to parse the logs and create the data for the training of the model.
    - the dataset will be in the `/models-validation/data/training-data` folder

**Note: if for any reason you stop the micro-benchmarks while running, you should also cleanup the created `logs/` folder.**

### Next step
Go to the `/testing-dataset` folder.