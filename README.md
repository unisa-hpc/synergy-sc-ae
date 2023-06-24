# SYnergy SC23 Artifact Evaluation
This repository provides the details for reproducing the results of the SC23 paper *SYnergy: Fine-grained Energy-Efficient Heterogeneous Computing for Scalable Energy Saving*.

## OS Requirements
Our experimental setup was based on Ubuntu 22.04 and 20.04.
We suggest using Ubuntu 20.04 or later.

## Hardware Requirements
- Single-node experiments: at least one NVIDIA GPU is required.
- Multi-node experiments: a cluster with NVIDIA GPUs is required, equipped with the provided NVGPUFREQ SLURM plugin.

## Software Requirements
- DPC++ (Intel/LLVM) [2022-09](https://github.com/intel/llvm/releases/tag/2022-09)
  - Install using the [Getting Started Guide](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md)
- Clang and LLVM 15
  - Install using the [LLVM automatic installation script](https://apt.llvm.org/#llvmsh)
- Python 3
  - Packages: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `paretoset`
- CUDA Toolkit (tested with CUDA 11.8)

Required for multi-node experiments: 
- [NVGPUFREQ SLURM plugin](https://github.com/LigateProject/slurm-nvgpufreq)
  - Follow the instruction in the readme file of the repository
- Application libraries
  - APT package `libpnetcdf-dev` (for MiniWeather)
- MPI Implementation (tested with Spectrum MPI)

## How to use this repository
This repository is divided in four directories:
- `training-dataset`, it contains all the scripts required to generate the data on which the models are trained
- `testing-dataset`, that contains the scripts to run the SYCL-Bench suite to gather validation dataset and characterization plots
- `models-validation`, provides the scripts for models training and inference based on the previous datasets and the scripts to reproduce the validation results
- `energy-scaling`, that provides the scripts to launch the MiniWeather and CloverLeaf applications to reproduce the energy scaling results

Each subdirectory has its own `README.md` file that provides additional information.

## Using pre-generated data
As running the tests may take some time and some specific requirements, we provide our data to obtain the exact same results of the paper.

This workflow does not run any application, but uses the data obtained during our experimental analysis.
To use this workflow visit the `models-validation` and `energy-scaling` folders and follow the readme files.

## Reproduce the results of the paper
In order to reproduce the results without the pre-generated data, make sure that all the requirements are fulfilled.
This workflow requires to visit the folders in the following order:
1. `training-dataset`
2. `testing-dataset`
3. `models-validation`
4. `energy-scaling`

Follow the steps defined in the respective readme files.

## Additional information
This section contains some useful information that you may need during the reproduction of the experiments.
### Specifying the CUDA architecture
Some scripts will require to specify the CUDA architecture (or Compute Capability), this table provides a reference of the format and code to be used to specify the CUDA architecture.

| Fermi | Kepler | Maxwell | Pascal |      Volta     | Turing | Ampere | Ada (Lovelace) |     Hopper    |
|:-----:|:------:|:-------:|:------:|:--------------:|:------:|:------:|:--------------:|:-------------:|
| sm_20 |  sm_30 |  sm_50  |  sm_60 |      sm_70     |  sm_75 |  sm_80 |      sm_89     |     sm_90     |
|       |  sm_35 |  sm_52  |  sm_61 | sm_72 (Xavier) |        |  sm_86 |                | sm_90a (Thor) |

### Obtain the GPU frequencies
When running the benchmarks to generate the training and testing datasets, the scripts will test the available frequencies of the GPU.
You can reduce the number of tested frequencies through the `--freq_sampling` command-line argument, that allows sampling the frequencies.

If you do not know how many core frequencies your GPU has, you can run the following command.
``` 
# All frequencies
nvidia-smi -i 0 --query-supported-clocks=gr --format=csv

# Number of frequencies
nvidia-smi -i 0 --query-supported-clocks=gr --format=csv,noheader | wc -l
```
If your GPU has a lot of frequencies, then it may be a good idea to sample some frequencies to reduce the execution time (this may change the models' accuracy).

**Note: if you use the `--freq_sampling` command-line argument, the same sampling value must be used for generating both the training and testing datasets.**

### `--gres:nvgpufreq` for SLURM batch jobs
In order to run the SLURM jobs with the NVGPUFREQ plugin, the `--gres:nvgpufreq` and `--exclusive` options must be specified in the batch job.

The provided scripts already specify these options.

### Compilation problems with SYCL 
Sometimes when compiling SYCL programs, if more than one gcc version is installed on the system, the SYCL compiler may have troubles finding the correct gcc toolchain that must be used.
In these cases, the `--cxx_flags` command-line argument can be used to give more information to the compiler about the location of the correct gcc toolchain.
Specifying `--gcc-toolchain=<gcc_toolchain_path>` (LLVM < 16) or `--gcc-install-dir=<gcc_install_path>` (LLVM >= 16) in the `--cxx_flags` will allow the compiler to locate the correct toolchain version.