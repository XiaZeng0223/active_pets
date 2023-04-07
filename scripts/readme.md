# Reproduce the results of the paper
## Run various baseline sampling strategy and train a PET  on the data iteratively

Set the model and model type to the model used in the paper, then run the following:


```
bash scripts/run_baselines.sh
```
Available `MODEL_TYPE` are `bert`, `roberta`, and `deberta`.

Available `TASK_NAME` are `cfever`, `scifact` and `scifact_oracle`.

The used `MODEL_NAME` are: `textattack/bert-base-uncased-MNLI` for BERT-base, `yoshitomo-matsubara/bert-large-uncased-mnli` for BERT-large; 
`textattack/roberta-base-MNLI` for RoBERTa-base, `roberta-large-mnli` for RoBERTa-large;
`microsoft/deberta-base-mnli` for DeBERTa-base, `microsoft/deberta-large-mnli` for DeBERTa-large.
Available `TASK_NAME` are `cfever`, `scifact` and `scifact_oracle`.
Available `SAMPLING` baseline strategies are `rand`, `badge`, `cal`, `alps`.

## Reproduce the exact random sampling results of the paper
To run a single experiment with random sampling with the default seed 42, 
use the code from the last section and set the strategy to `rand`.

To run the experiments on 10 different random seeds from 123 to 132, run the following:

```
bash scripts/run_random.sh
```

## Reproduce the activepets results of the paper
```
bash scripts/run_ensemble.sh
```

To run Active PETs with oversampling, simply modify `run_ensemble.sh` by adding `--oversample` to the end of the train function.

## Final results and plots as reported in the paper
To see the final results and plots as reported in the paper, go to folder `final_plots`. 
They are obtained with `plotting/tables.py` and `plotting/plot.py`.