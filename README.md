# SYnergy SC23 Artifact Evaluation
This repository provides the details for reproducing the results of the SC23 paper *SYnergy: Fine-grained Energy-Efficient Heterogeneous Computing for Scalable Energy Saving*.

## OS Requirements
Our experimental setup was based on Ubuntu 22.04 and 20.04.
We suggest using Ubuntu 20.04 or later.

## Hardware Requirements
- Single-node experiments: t least one NVIDIA GPU is required.
- Multi-node experiments: a cluster with NVIDIA GPUs is required, equipped with the provided SLURM plugin.

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
- Application libraries
  - APT package `libpnetcdf-dev` (for MiniWeather)
- MPI Implementation (tested with Spectrum MPI)

## How to use this repository
This repository is divided in four directories:
- `training-dataset`, it contains all the scripts required to generate the data on which the models are trained;
- `testing-dataset`, that contains the scripts to run the SYCL-Bench suite to gather validation dataset and characterization plots;
- `models-validation`, provides the scripts for models training and inference based on the previous datasets and the scripts to reproduce the validation results;
- `energy-scaling`, that provides the scripts to launch the MiniWeather and CloverLeaf applications to reproduce the energy scaling results.

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