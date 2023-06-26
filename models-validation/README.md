# Models validation
This folder contains the scripts to build and validate the machine learning models used for frequency predictions.
The models training and validation can be performed either on the pre-built dataset or on the dataset generated during the previous steps.

The expected execution time is around 30 minutes. The process could take less time if the training dataset is smaller i.e. if you used the `--freq_sampling` argument.

## Folder structure
- `provided-data` contains the pre-built dataset, obtained during our experimental evaluation
- the `data` is generated during the previous steps and contains the training and testing datasets

## Steps to reproduce
1. Run `source validate.sh` to train the models and evaluate the accuracy of the model.
    - Optional parameters:
      - `--provided_data` must be passed as an argument if you want to use the pre-built dataset
    - the output plots for the prediction error will be placed in the `predictions/benchmarks-errors/` folder
    - the output data used for the error analysis will be in the `predictions/algorithms.txt` file

### Next step
Go to the `/energy-scaling` folder.