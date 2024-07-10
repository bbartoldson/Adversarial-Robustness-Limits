"""
Evaluation with AutoAttack.
"""

import json
import time
import argparse
import shutil

import os
import numpy as np
import pandas as pd
from copy import deepcopy

import torch
import torch.nn as nn

from autoattack import AutoAttack

from core.data import get_data_info
from core.data import load_data
from core.models import create_model

from core.utils import Logger
from core.utils import parser_eval
from core.utils import seed

from pathlib import Path

from core.utils import set_config_file_precedence, validate_train_arguments, set_extra_arguments

def getBatchSize(args):
    if not args.config:
        print('Error We do not know how to compute the hash')
        sys.exit()

    hashVal = Path(args.config).stem
    deploy_descr = f'job_scripts/{hashVal}_job_info.json'
    with open(deploy_descr, 'r') as fd:
        data = json.load(fd)
    nnodes = data['nnodes']
    return int(int(args.batch_size) / int(nnodes))

# Setup

parse = parser_eval()
parse = set_extra_arguments(parse)
args = parse.parse_args()
args = set_config_file_precedence(args) # needed to get desc

LOG_DIR = args.log_dir + '/' + args.desc
print(LOG_DIR)
with open(LOG_DIR+'/args.txt', 'r') as f:
    old = json.load(f)
    args.__dict__ = dict(vars(args), **old)

validate_train_arguments(args, parse)

if args.data in ['cifar10', 'cifar10s']:
    da = '/cifar10/'
elif args.data in ['cifar100', 'cifar100s']:
    da = '/cifar100/'
elif args.data in ['svhn', 'svhns']:
    da = '/svhn/'


DATA_DIR = args.data_dir + da
WEIGHTS = LOG_DIR + '/weights-best.pt'

log_path = LOG_DIR + '/log-aa.log'
aa_state_path = LOG_DIR + '/aa_state.aa'
logger = Logger(log_path)

info = get_data_info(DATA_DIR)

BATCH_SIZE = getBatchSize(args)
BATCH_SIZE_VALIDATION = BATCH_SIZE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.log('Using device: {}'.format(device))


# Load data

seed(args.seed)
_, _, train_dataloader, test_dataloader = load_data(DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=False, 
                                                    shuffle_train=False)

if args.train:
    logger.log('Evaluating on training set.')
    l = [x for (x, y) in train_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in train_dataloader]
    y_test = torch.cat(l, 0)
else:
    l = [x for (x, y) in test_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_dataloader]
    y_test = torch.cat(l, 0)



# Model
model = create_model(args.model, args.normalize, info, device)
checkpoint = torch.load(WEIGHTS)
if 'tau' in args and args.tau:
    print ('Using WA model.')
def distributed_safe_load(checkpoint):
    msd = deepcopy(checkpoint['model_state_dict'])
    if 'module'==list(checkpoint['model_state_dict'].keys())[0][:6]:
        for k in checkpoint['model_state_dict']:
            assert k[:7] == 'module.'
            msd[k[7:]] = msd[k]
            del msd[k]
    return msd
model.load_state_dict(distributed_safe_load(checkpoint))
model = torch.nn.DataParallel(model) # adding here because brian removed from create_model to facilitate ddp
model.eval()
del checkpoint


# AA Evaluation

seed(args.seed)
norm = 'Linf' if args.attack in ['fgsm', 'linf-pgd', 'linf-df'] else 'L2'
adversary = AutoAttack(model, norm=norm, eps=args.attack_eps, log_path=log_path, version=args.version, seed=args.seed) 

if args.version == 'custom':
    adversary.attacks_to_run = ['apgd-ce', 'apgd-t']
    adversary.apgd.n_restarts = 1
    adversary.apgd_targeted.n_restarts = 1

with torch.no_grad():
    # Note: adversary.run_standard_evaluation expects Path object for state_path
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=BATCH_SIZE_VALIDATION, state_path=Path(aa_state_path))


csv_row = {'config':args.desc, 'epoch': None, 'auto_attack': None, 'examples_per_epoch': None,
                'fwd_flops_per_example': None, 'seconds_per_example': None, 'parameters': None,
                'CW_train_acc': None, 'CW_train_loss': None, 'CW_test_acc': None, 'CW_test_loss': None,
                'PGD40_train_acc': None, 'PGD40_train_loss': None, 'PGD40_test_acc': None, 'PGD40_test_loss': None,
                'clean_train_acc': None, 'clean_train_loss': None, 'clean_test_acc': None, 'clean_test_loss': None}
def write_result_row(file_name, csv_row):
    row = pd.DataFrame(csv_row, index=[0])
    if os.path.exists(file_name):
        metrics = pd.read_csv(file_name)
        metrics = pd.concat([metrics, row], ignore_index=True)
        metrics.to_csv(file_name, index=False)
    else:
        row.to_csv(file_name, index=False)
def get_aa(fn):
    accs = []
    if os.path.exists(fn):
        with open(fn, 'r') as f:
            lines = f.readlines()
            for l in lines:
                if 'robust accuracy:' in l:
                    accs.append(float(l.split()[-1][:-1])) #take all but the % symbol
    assert len(set(accs)) == 1, accs 
    return accs[0]
csv_row['auto_attack'] = get_aa(log_path)
write_result_row(os.path.join(LOG_DIR, 'results.csv'), csv_row)


print('Eval-aa script completed!')
logger.log('Eval-aa script completed!')
print(csv_row)
logger.log(csv_row)
