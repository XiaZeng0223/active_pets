# Parse results into tables
Run the following command for baseline methods:
```
python plotting/tables.py --task cfever --results_dir baseline_pets --output_dir baseline_analysis
```
Available tasks include `cfever`, `scifact` and `scifact_oracle`.

Run the following command for activepets, i.e., an ensemble of pet models:
```
python plotting/tables.py --task cfever --ensemble --results_dir ensemble_pets --output_dir ensemble_analysis
```
Available tasks include `cfever`, `scifact` and `scifact_oracle`. Change the results_dir into `ensemble_pets/o` and the output_dir into `ensemble_analysis/o` for active_pets with oversampling.

# Generate plots from tables
Run the following command:
```
python plotting/plot.py --task cfever --dir_tables baseline_analysis ensemble_analysis ensemble_analysis/o rand_analysis --output_dir final_plots
```
Available tasks include `cfever`, `scifact` and `scifact_oracle`.
