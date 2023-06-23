# Testing dataset and characterization
This folder contains all the source code and scripts needed to generate the models' test dataset and the characterization plots of the paper.
The models' testing can also be performed on the data provided by the authors, so running SYCL-Bench is optional for functionality assessment.

## Steps to reproduce
1. `extract_features.sh` script will extract static code features from SYCL-Bench 
  - run the script with `source extract_features.sh <DPC++ compiler path>`, providing the absolute path to the DPC++ compiler folder as first argument
  - the output features will be in the `features-*/` folders
2. `run_syclbench.sh` script will run the SYCL-bench benchmarks
  - **this script must be run as root, otherwise frequency scaling cannot be performed**
  - run the script with `sudo ./run_syclbench.sh <DPC++ compiler path> [frequency_sampling]`
    - first argument: absolute path to the DPC++ compiler folder
    - second argument: optional argument to reduce the number of tested core frequencies, the script will test one frequency every `frequency_sampling`
3. `process_syclbench.sh` script will process the logs and create the data for the validation of the model and the plots of the characterization section
  - run the script with `source process_syclbench.sh`
  - the dataset will be in the `/validation/data/testing-data` folder
  - the characterization plots (Figure 5-8) will be in the `plots/` folder
## Steps to reproduce on multi-node
1. `extract_features.sh` script will extract static code features from SYCL-Bench 
  - run the script with `source extract_features.sh <DPC++ compiler path>`, providing the absolute path to the DPC++ compiler folder as first argument
  - the output features will be in the `features-*/` folders
2. `run_syclbench.sh` script will run the SYCL-bench benchmarks
  - **this script must be run as root, otherwise frequency scaling cannot be performed**
  - run the script with `sudo ./run_syclbench.sh <DPC++ compiler path> [frequency_sampling]`
    - first argument: absolute path to the DPC++ compiler folder
    - second argument: optional argument to reduce the number of tested core frequencies, the script will test one frequency every `frequency_sampling`
3. `process_syclbench.sh` script will process the logs and create the data for the validation of the model and the plots of the characterization section
  - run the script with `source process_syclbench.sh`
  - the dataset will be in the `/validation/data/testing-data` folder
  - the characterization plots (Figure 5-8) will be in the `plots/` folder
