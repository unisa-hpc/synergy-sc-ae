# Characterization

Say that we provide logs with our output

## Steps to reproduce
1. `extract_features.sh` script will extract static code features from SYCL-Bench 
  - run the script with `source extract_features.sh <DPC++ compiler path>`, providing the absolute path to the DPC++ compiler as first argument
  - the output features will be in the `features-*/` folders
2. `run_microbenchmarks.sh` script will run the SYCL-bench benchmarks
  - **this script must be run as root, otherwise frequency scaling cannot be performed**
  - run the script with `sudo ./extract_features.sh <DPC++ compiler path>`, providing the absolute path to the DPC++ compiler as first argument
3. `process_syclbench.sh` script will process the logs and create the data for the validation of the model and the plots of the characterization section
  - run the script with `source process_syclbench.sh`
  - or run the script with `source process_syclbench.sh provided` to operate on the logs provided by the authors
