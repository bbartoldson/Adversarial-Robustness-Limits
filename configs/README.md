# Illustrative examples of training run launches

Note that these jobs were run on nodes with 4 V100 GPUs each. To reach the desired batch sizes, multiple nodes were used.

### Example run to train model used to fit scaling laws

Note that models used for our scaling laws were trained entirely on synthetic data.

- WRN-28-8 training on DG-20 data
  - `python submit.py --config configs/example_run_for_scaling_laws.json deploy --nnodes 2`

### Compute-optimal model (settings are optimal according to scaling laws from the paper)

CIFAR-10 data is mixed in throughout training.

- compute-optimal training at the prior SOTA's FLOPs (71.59% AA) 
  - `python submit.py --config configs/compute_optimal.json deploy --nnodes 8`

### SOTA

Note that the following configuration is used for the final 4000 epochs of training. The first 6000 epochs use the same config except `unsup_fraction=1` to avoid overfitting to the non-synthetic CIFAR-10 data.

- SOTA training (73.71% AA) 
  - `python submit.py --config configs/SOTA.json deploy --nnodes 32`
