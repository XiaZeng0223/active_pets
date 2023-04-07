# Active PETs
Code repository for Active PETs. 

Our main contribution is a weighted ensemble of PETs, which is used to actively sample the most beneficial samples from an unlabelled pool.

This readme file mainly describe how to use the code. For more details to reproduce the experiments reported in the paper, please see the readme file under scripts.
# Installation
1. Create virtual environment with Python 3.7+
2. Run following commands:
```
pip install -r requirements.txt
```
# Organization
The repository is organized as the following subfolders:

1. `data`: folder for datasets
2. `src_pet`: source code for simulating active learning
3. `pet`: core code for PETs
4. `scripts`: scripts for running experiments
5. `pets`: saved models from running experiments
6. `results`: results of active learning experiments

# Usage
All commands below should be ran in the top-level directory `activepets`.

## Train a PET model on full training dataset
To simply train a PET model on the full training dataset, run 

`bash scripts/train.sh`  

After training, this model will be saved under a subdirectory called `base` in `pets` directory.  Results on dev set will be saved in `eval_results.txt`.

You may modify the parameters (like model type, task, seed, etc.) in `scripts/train.sh`by configuring the variables at the top of the script.  

## Run active learning simulations
To simulate various active learning without ensemble, run 

`bash scripts/active_train.sh` 

This script will sample data for a fixed number of iterations and then fine-tune the model on the sampled data for each iteration.  The fine-tuned model will be saved under a subdirectory called `{strategy}_{size}` where `strategy` is the active learning strategy used to sample data and `size` is the number of examples used to fine-tune the model.  Results on dev set will be saved in `eval_results.txt`.

To modify parameters in `scripts/active_train.sh`, you can configure the variables at the top of the script.  Please read the instructions below for more information.

## Run active_pets simulations
To simulate various active learning, run

`bash scripts/active_commitee.sh` 

### Naming conventions of strategies
Here are the naming conventions of the strategies from the paper:

1. Random sampling: `rand`
2. BADDGE: `badge`
3. CAL: `cal`
4. ALPS: `alps`
6. Active-PETS: `activepets`

So, whenever you want to use Active-PETs, you would pass in `activepets` as input to the commands presented below.


### Sample size
To set the size of data sampled on each iteration, configure the variable `INCREMENT`.  To set the maximum size of total data sampled, configure the variable `MAX_SIZE`.  The number of iterations would be `MAX_SIZE\INCREMENT`.


