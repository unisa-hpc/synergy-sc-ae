# Training dataset
This folder contains all the source code and scripts needed to generate the models' training dataset.

The models training can also be performed on the data provided by the authors, so running the microbenchmarks is optional for functionality assessment.

## Steps to reproduce
1. `extract_features.sh` script will extract static code features from the microbenchmarks in order to build the model.
  - run the script with `source extract_features.sh <DPC++ compiler path>`, providing the absolute path to the DPC++ compiler folder as first argument
  - the output features will be in the `micro/features-*/` subfolders
2. `run_microbenchmarks.sh` script will run the microbenchmarks and save the logs for parsing
  - **this script must be run as root, otherwise frequency scaling cannot be performed**
  - run the script with `sudo ./run_microbenchmarks.sh <DPC++ compiler path> [frequency_sampling]`
    - first argument: absolute path to the DPC++ compiler folder
    - second argument: optional argument to reduce the number of tested core frequencies, the script will test one frequency every `frequency_sampling`
  - the logs will be in the `logs/` subfolder
3. `process_microbenchmarks.sh` script will parse the logs and create the data for the training of the model
  - run the script with `source process_microbenchmarks.sh`
  - the dataset will be in the `/validation/data/training-data` folder