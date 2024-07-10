import argparse

from core.attacks import ATTACKS
from core.data import DATASETS
from core.data import SEMISUP_DATASETS
from core.models import MODELS
from .train import SCHEDULERS
import sys
import json

from .utils import str2bool, str2float

def update_parsed_values(cli_args, config):
    """
    Updates the command line arguments using a the key-value
    pairs in the dictionary
    """

    args_dict = vars(cli_args)
    args_dict.update(config)
    args = argparse.Namespace(**args_dict)
    return args


def set_config_file_precedence(args):
    """
    Updates the cli argument values with those in a configuration file
    only if such configuration file was passed as an option
    """

    if args.config:
        with open(args.config, 'r') as fd:
            config_args = json.load(fd)
        args = update_parsed_values(args, config_args)

    return args

def validate_train_arguments(args, parse):
    """
    Validate the arguments are correct to start training
    """

    assert args.data in SEMISUP_DATASETS, f'Only data in {SEMISUP_DATASETS} is supported!'

    if args.desc is None:
        parse.print_help(sys.stderr)
        print('{0} error: the following arguments are required --desc,'
            ' either use CLI or define them in config json file'.format(sys.argv[0]), file=sys.stderr)
        sys.exit()
    return

def set_extra_arguments(parse):
    parse.add_argument('--tau', type=float, default=0.995, help='Weight averaging decay.')
    parse.add_argument('--unfix_N_batches_per_epoch', action='store_true', help='As diffusion data is added, increase number of batches per epoch.')
    parse.add_argument('--better_sampler', action='store_true', help='Sample the data with a more uniform sampler (i.e., without replacement). See the "one_epoch" arg for a better version.')
    parse.add_argument('--EDM_50_amount', type=float, help='In millions of desired training examples, the size of the subset of the 50 million image EDM dataset. Requires the aux_data_filename to be the path to the 50M images.')
    parse.add_argument('--wandb_group', default='')
    parse.add_argument('--wandb_proj', default='')
    parse.add_argument('--pct_start', type=float, default=0.025, help='Warmup fraction for cosinew and cyclic LR schedulers.')
    parse.add_argument('--config', '-json', type=str,
                       help='JSON file containing cli key-value arguments')
    parse.add_argument('--robustblock', action='store_true', help='When using WRN, convert its blocks to RobustResBlocks')
    parse.add_argument('--one_epoch', action='store_true', help='Sample the data with a more uniform sampler (i.e., without replacement). When training resumes, load prior state to continue sampling without replacement.')
    return parse

def parser_train(parser):
    """
    Parse input arguments (train.py).
    """
    # parser.add_argument('--augment', type=str2bool, default=True, help='Augment training set.')
    parser.add_argument('--augment', type=str, default='base', choices=['none', 'base', 'cutout', 'autoaugment', 'randaugment', 'idbh'], help='Augment training set.')

    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for training.')
    parser.add_argument('--batch-size-validation', type=int, default=512, help='Batch size for testing.')

    parser.add_argument('--data-dir', type=str, default='/cluster/home/rarade/data/')
    parser.add_argument('--log-dir', type=str, default='/cluster/scratch/rarade/test/')

    parser.add_argument('-d', '--data', type=str, default='cifar10s', choices=DATASETS, help='Data to use.')
    parser.add_argument('--desc', type=str, default=None,
                        help='Description of experiment. It will be used to name directories.')

    parser.add_argument('-m', '--model', choices=MODELS, default='wrn-28-10-swish', help='Model architecture to be used.')
    parser.add_argument('--normalize', type=str2bool, default=False, help='Normalize input.')
    parser.add_argument('--pretrained-file', type=str, default=None, help='Pretrained weights file name.')

    parser.add_argument('-na', '--num-adv-epochs', type=int, default=400, help='Number of adversarial training epochs.')
    parser.add_argument('--adv-eval-freq', type=int, default=25, help='Adversarial evaluation frequency (in epochs).')

    parser.add_argument('--beta', default=None, type=float, help='Stability regularization, i.e., 1/lambda in TRADES.')

    parser.add_argument('--lr', type=float, default=0.4, help='Learning rate for optimizer (SGD).')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Optimizer (SGD) weight decay.')
    parser.add_argument('--scheduler', choices=SCHEDULERS, default='cosinew', help='Type of scheduler.')
    parser.add_argument('--nesterov', type=str2bool, default=True, help='Use Nesterov momentum.')
    parser.add_argument('--clip-grad', type=float, default=None, help='Gradient norm clipping.')

    parser.add_argument('-a', '--attack', type=str, choices=ATTACKS, default='linf-pgd', help='Type of attack.')
    parser.add_argument('--attack-eps', type=str2float, default=8/255, help='Epsilon for the attack.')
    parser.add_argument('--attack-step', type=str2float, default=2/255, help='Step size for PGD attack.')
    parser.add_argument('--attack-iter', type=int, default=10, help='Max. number of iterations (if any) for the attack.')
    parser.add_argument('--keep-clean', type=str2bool, default=False, help='Use clean samples during adversarial training.')

    parser.add_argument('--debug', action='store_true', default=False,
                        help='Debug code. Run 1 epoch of training and evaluation.')
    parser.add_argument('--mart', action='store_true', default=False, help='MART training.')

    parser.add_argument('--unsup-fraction', type=float, default=0.7, help='Ratio of unlabelled data to labelled data.')
    parser.add_argument('--aux-data-filename', type=str, help='Path to additional Tiny Images data.',
                        default='/cluster/scratch/rarade/cifar10s/ti_500K_pseudo_labeled.pickle')

    parser.add_argument('--seed', type=int, default=1, help='Random seed.')

    ### Consistency
    parser.add_argument('--consistency', action='store_true', default=False, help='use Consistency.')
    parser.add_argument('--cons_lambda', type=float, default=1.0, help='lambda for Consistency.')
    parser.add_argument('--cons_tem', type=float, default=0.5, help='temperature for Consistency.')

    ### Resume
    parser.add_argument('--resume_path', default='', type=str)

    ### Our methods
    parser.add_argument('--LSE', action='store_true', default=False, help='LSE training.')
    parser.add_argument('--ls', type=float, default=0.1, help='label smoothing.')
    parser.add_argument('--clip_value', default=0, type=float)
    parser.add_argument('--CutMix', action='store_true', default=False, help='use CutMix.')
    return parser

def parser_eval():
    """
    Parse input arguments (eval-adv.py, eval-corr.py, eval-aa.py).
    """
    parser = argparse.ArgumentParser(description='Robustness evaluation.')

    parser.add_argument('--data-dir', type=str, default='/cluster/home/rarade/data/')
    parser.add_argument('--log-dir', type=str, default='/cluster/scratch/rarade/test/')

    parser.add_argument('--desc', type=str, default=None, help='Description of model to be evaluated.')
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of test samples.')

    # eval-aa.py
    parser.add_argument('--train', action='store_true', default=False, help='Evaluate on training set.')
    parser.add_argument('-v', '--version', type=str, default='standard', choices=['custom', 'plus', 'standard'],
                        help='Version of AA.')

    # eval-adv.py
    parser.add_argument('--source', type=str, default=None, help='Path to source model for black-box evaluation.')
    parser.add_argument('--wb', action='store_true', default=False, help='Perform white-box PGD evaluation.')

    # eval-rb.py
    parser.add_argument('--threat', type=str, default='corruptions', choices=['corruptions', 'Linf', 'L2'],
                        help='Threat model for RobustBench evaluation.')

    parser.add_argument('--seed', type=int, default=1, help='Random seed.')

    return parser

