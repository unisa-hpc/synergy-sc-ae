# SYnergy SC23 Artifact Evaluation

## Requirements
We suggest using Ubuntu 20.04 or later.
## Hardware Requirements
- For training, validation and characterization, at least an NVIDIA GPU;
- For the scaling results, a cluster with GPUs is needed, with the provided SLURM plugin.
## Software Requirements
- DPC++ (Intel/LLVM) [2022-09](https://github.com/intel/llvm/releases/tag/2022-09)
  - Install using the [Getting Started Guide](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md)
- Clang and LLVM 15
  - Install using the [LLVM automatic installation script](https://apt.llvm.org/#llvmsh)
- Python 3
  - Packages: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `paretoset`
- [SLURM NVGPUFREQ plugin](https://github.com/LigateProject/slurm-nvgpufreq)

## How to use this repository
This repository is divided in four subdirectories:
- `training-dataset`, it contains all the scripts required to generate the data on which the models are trained;
- `testing-dataset`, that contains the scripts to run the SYCL-Bench suite to gather validation dataset and characterization plots;
- `models-validation`, provides the scripts for model training and inference based on the previous datasets and the scripts to reproduce the validation results;
- `energy-scaling`, that provides the scripts to launch the MiniWeather and CloverLeaf applications to reproduce the energy scaling results.

Each subdirectory has its own README.md file that provides additional information.

## Use our own data
As running the tests may take some time and some specific requirements, we provide our data to reproduce the results of the paper.
You can find additional information in each subfolder's readme file.

## Use your own data
The scripts allow you to reproduce our work in a simple way, given that all requirements are fulfilled.
You can find additional information in each subfolder's readme file.