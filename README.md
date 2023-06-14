# SYnergy SC23 Artifact Evaluation

## Requirements
We suggest using Ubuntu 20.04 or later.
## Hardware Requirements
- For training, validation and characterization, at least an NVIDIA GPU;
- For scaling results, a cluster with GPUs is needed, with the provided SLURM plugin.
## Software Requirements
- DPC++ (Intel/LLVM) [2022-09](https://github.com/intel/llvm/releases/tag/2022-09)
  - Install using  use [Getting Started Guide](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md)
- Clang and LLVM 15
  - Install using the [LLVM automatic installation script](https://apt.llvm.org/#llvmsh)
- Python 3
  - Packages: `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `paretoset`

## How to use this repository
This repository is divided in four subdirectories:
- `training`, it contains all the scripts required to build the models;
- `characterization`, that contains the scripts to run the SYCL-Bench suite to gather validation data and characterization;
- `validation`, provides the scripts for inference based on the SYCL-Bench codes and the scripts to reproduce the validation results;
- `scaling`, that provides the scripts to launch the MiniWeather and CloverLeaf applications to reproduce the energy scaling results.

Each subdirectory has its own README.md file that provides additional information.