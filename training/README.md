# Training
## Steps to reproduce
1. `extract_features.sh` script will extract static code features from the microbenchmarks in order to build the model.
  - run the script with `source extract_features.sh <DPC++ compiler path>`, providing the absolute path to the DPC++ compiler as first argument
  - the output features will be in the `micro/features-*/` folders
2. `run_microbenchmarks.sh` script will run the microbenchmarks and parse the results
  - run the script with `source run_microbenchmarks.sh <DPC++ compiler path>`, providing the absolute path to the DPC++ compiler as first argument

TODO:
- parse the results of the logs and merge them with the kernel features in a way that is appropriate for the training script
- train the model
