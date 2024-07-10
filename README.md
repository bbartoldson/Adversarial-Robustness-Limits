# Adversarial Robustness Limits

This is the official repository for the ICML 2024 paper [Adversarial Robustness Limits via Scaling-Law and Human-Alignment Studies](https://arxiv.org/abs/2404.09349). This paper revisits the simple, long-studied, yet still unsolved problem of making image classifiers robust to imperceptible perturbations. 

![Figure](https://github.com/bbartoldson/Adversarial-Robustness-Limits/assets/15717529/b83d023f-4a28-473c-ae41-ead42941d5f0)

Focusing on performance of CIFAR-10 classifiers on data attacked by $`\ell_{\infty}`$-norm-constrained perturbations, we derive scaling laws that suggest NN adversarial robustness asymptotes around $90$%. Corroborating this limit, we find that humans quizzed on adversarial images that successfully attack our NN with SOTA robustness fail at a rate that is consistent with their performances also having a peak of about $90$%.


## :memo: Take the quiz

Given images that can be perturbed to successfully attack our NN with SOTA robustness, our paper studied human performance on the clean and perturbed versions of the images. 

See your performance on a 25-image sample of this data via the [Human Perception Robustness Quiz](https://adversarial-robustness-limit-quiz.netlify.app/).

We found humans often agreed with the NN label on the perturbed images, suggesting the ground-truth label can be changed by the perturbation (despite its constrained magnitude), and that NNs should not be considered incorrect on such "invalid" adversarial data. Other times, humans correctly classified the perturbed images, illustrating areas where NNs can still improve. Please see our paper for more information.


## :chart_with_upwards_trend: Learn scaling laws on trained model data

The notebook `Scaling_law_fitting_and_plots.ipynb` illustrates how to fit our three different scaling law approaches and use the fits to reproduce the related plots in our paper.


## :weight_lifting: Train models

The `configs` folder has examples of specific configurations (e.g. that of our SOTA run), and illustrates how to launch a job to train with one of these configurations on Lassen (an LLNL machine with the LSF job manager). Our job launching script `submit.py` is explained below.


### Instructions on how to submit a job allocation on Lassen using the python submission script

Running dependent jobs until the training has finished can be done through the submit.py script

```
python submit.py deploy --help
  -h, --help            show this help message and exit
  --nnodes NNODES, -n NNODES
                        Number of nodes to perform the submission with
  --jname-prefix JNAME_PREFIX
                        the job name prefix, all chained jobs will user prefix_{i}, with i being the current number
  --time TIME, -t TIME  The job allocation time in minutes
  --bank BANK           Which bank to use upon submission
  --count COUNT         Job count, we start from 0
  --expedite            Use the expedite flag
  --out-dir OUT_DIR, -o OUT_DIR
                        The output dir of stdout/stderr of the job
  --max-jobs MAX_JOBS, -m MAX_JOBS
                        the maximum number of jobs we should chain
  --processes-per-node PROCESSES_PER_NODE, -ppn PROCESSES_PER_NODE
                        Number of processes per node
  --default-config DEFAULT_CONFIG
                        Default configuration of options of train-wa-distributed.py.
  --batch-size BATCH_SIZE, -bz BATCH_SIZE
                        The per node batch size. The script will scale it to match training script requirements
  --epochs EPOCHS       Epochs to train with
  --edm-size EDM_SIZE   EMD Size to be used by training script
  --edm_frac EDM_FRAC   set to 7 if 1M and below, else set to 0.8
  --model MODEL         Type of the model to be used
  --data DATA           data to be used for training
```

The script takes as input arguments information regarding the allocation to happen and a subset of the configuration options for the machine learning
script. The rest of the machine learning training script are defined in  configs/defaults.json. For example, the ml parameter "tau" is not exported
in the arguments of the submit.py script. So you can either extend the script to add this option, or you can just modify the configs/defaults.json file
with a new value, or just copy the defaults into another file, edit it and inform submit.py about that file through the ---default-config option.

The script creates a submission script under 'job\_scripts' and submits a job to execute that script. The job\_script first adds a new job in the queue
that depends on the current job and then continues or starts the training process. The process continues until --max-job jobs have been executed and then
it terminates.


## Acknowledgements

This repository began as a fork of the repository for the paper [Better Diffusion Models Further Improve Adversarial Training](https://arxiv.org/pdf/2302.04638.pdf) (Wang et al., 2023). We added capabilities to and adapted their codebase for our needs, ultimately losing compatibility with their code, which can be found here: https://github.com/wzekai99/DM-Improves-AT/.


## Release

The code of this site is released under the MIT License. 

LLNL-CODE-866411
