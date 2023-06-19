# Models validation
This folder contains the scripts to build and validate the machine learning models used for frequency predictions.
The pre-built dataset is located in the `provided-data/` folder, containing the data used in the paper.

## Steps to reproduce
1. `validate.sh` script will train the model and evaluate the accuracy of the model.
  - run the script with `source validate.sh <data directory name>` providing the name of the directory containing the training and testing data
    - running the script with `source validate.sh provided-data` will reproduce the same results of the paper 
  - the output plots for the prediction error (Figure 9) will be placed in the `predictions/benchmarks-errors/` folder
  - the output data used for the error analysis (Table 2) will be in the `/predictions/algorithms.txt` file