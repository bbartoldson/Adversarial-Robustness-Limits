import argparse
import json
from pathlib import Path
from string import Template
import subprocess
import sys, os

from core.utils import parser_train
from core.utils import update_parsed_values
from core.utils import set_extra_arguments
from core.utils import validate_train_arguments
from core.utils import set_config_file_precedence

import hashlib

from train_wa_distributed import distributed_train

def file_contains(fn, string):
    if os.path.exists(fn):
        with open(fn, 'r') as f:
            l = f.readlines()
            if string in l:
                return True
    return False

def get_state(args):
    train_completed = file_contains(os.path.join(args.log_dir, args.desc,
                                                 'log-train.log'),
                                    'Script Completed.\n')
    aa_completed = file_contains(os.path.join(args.log_dir,
                                              args.desc, 'log-aa.log'),
                                 'Eval-aa script completed!\n')
    configs_for_aa = ['scaling_study_num_2_modelwrn-58-12-swish_data175_bs2048_lr0.2_fnvanilla20dg20pfgmpp18-150_unsup0.875',
                      'scaling_study_num_2_modelwrn-82-12-swish_data150_bs2048_lr0.2_fnvanilla20dg20pfgmpp18-150_unsup0.875',
                    'scaling_study_num_2_modelwrn-82-8-swish_data341_bs2048_lr0.1_fnvanilla20dg20pfgmpp18-341_unsup0.875',
                    'scaling_study_num_2_modelwrn-94-16-swish_data500_bs2048_lr0.1_fnvanilla20dg20pfgmpp18_unsup0.875_beta20_withDetachingCleanFromKL_SOTA',
                       'scaling_study_num_2_modelwrn-94-16-swish_data500_bs2048_lr0.1_fnvanilla20dg20pfgmpp18_withCIFARfrom6000to10000' ]
    config_relevant_to_aa = args.desc in configs_for_aa

    print('Training for {0} {1}completed'.format(args.desc,
                                                 '' if train_completed
                                                    else 'not '))
    print('AutoAttack for {0} {1}completed'.format(args.desc,
                                                 '' if aa_completed
                                                    else 'not '))
    state = 'Train'
    if train_completed and (not aa_completed):
        state = 'Done' # to prioritize running experiments
        if config_relevant_to_aa:
            state = 'AutoAttack'
    elif train_completed and aa_completed:
        state = 'Done'
    return state

def display_help(train_parser, extras, args):
    subparser = train_parser.add_subparsers(dest='command')
    deploy = subparser.add_parser('deploy', help='Deploy Help')
    chain = subparser.add_parser('chain', help='Chain Help')
    info = subparser.add_parser('info', help='Info Help')
    args, extras = train_parser.parse_known_args(extras, namespace=args)
    deploy = setup_deploy_args(deploy)
    chain = setup_chain_args(chain)
    print(train_parser.format_help())
    # retrieve subparsers from parser
    subparsers_actions = [
        action for action in train_parser._actions   if isinstance(action, argparse._SubParsersAction)]
    for subparsers_action in subparsers_actions:
    # get all subparsers and print help
        for choice, subparser in subparsers_action.choices.items():
            print("Subparser '{}'".format(choice))
            print(subparser.format_help())

def execute_command(cmd, **kwargs):
    """
    wrapper around subprocess run
    """
    print('Execute', cmd)
    print('kwargs', kwargs)
    try:
        p = subprocess.run( cmd, check=True, **kwargs )
    except subprocess.CalledProcessError as e:
        print('Failed cmd', e.cmd)
        print('ret',e.returncode)
        print('stdout', e.stdout.decode())
        print('stderr', e.stderr.decode(), file=sys.stderr)
        print(e)
        sys.exit()

    if 'capture_output' in kwargs and kwargs['capture_output']:
        print(p.stdout.decode('utf-8'))
        print(p.stderr.decode('utf-8'), file=sys.stderr)


def schedule(bsub_cmd, script, dry=True, script_args=list()):
    """
    Executes the bsub command with the given script and passes the script_args.
    When dry is true it only prints the command
    """
    script_args = ' '.join(script_args)
    cmd = f'{bsub_cmd} sh {script} {script_args}'
    print(cmd)

    if not dry:
        print('Scheduling')
        execute_command(cmd, capture_output=True, shell=True)

def setup_chain_args(chain):
    chain.add_argument('--job-config', type=str, required=True,
                        help='Configuration file containing all information about current submission')
    return chain

def setup_deploy_args(deploy):
      deploy.add_argument('--nnodes', '-n', type=int, required=True,
                          help='Number of nodes to perform the submission with')
      deploy.add_argument('--time', '-t', type=int, default=12*60,
                          help='The job allocation time in minutes')
      deploy.add_argument('--bank', type=str, default='ml4ss',
                          help='Which bank to use upon submission')
      deploy.add_argument('--count', type=int, default=0,
                          help='Job count, we start from 0')
      deploy.add_argument('--expedite', action='store_true', default=False,
                          help='Use the expedite flag')
      deploy.add_argument('--out-dir', '-o', type=str, default='submission_scripts/chainer_output_log',
                          help='The output dir of stdout/stderr of the job')
      deploy.add_argument('--max-jobs', '-m', type=int, default=25,
                          help = 'the maximum number of jobs we should chain')
      deploy.add_argument('--processes-per-node', '-ppn',
                          type=int, default=4,
                          help='Number of processes per node')
      deploy.add_argument('--queue', '-q', default='pbatch',
                          help='Queue to submit the job to')
      return deploy


def create_bsub_command(args, state, count=0):
    """
    Given cli arguments it creates a job submission command
    that depends on the previous command.
    """

    command = ['bsub']
    if args.expedite:
        command.append('-q')
        command.append('expedite')
    else:
        command.append('-q')
        command.append(args.queue)

    command.append('-nnodes')
    if state == 'Train':
        command.append(str(args.nnodes))
    elif state == 'AutoAttack': # Auto attack can not use multiple nodes
        command.append('1')

    command.append('-alloc_flags')
    command.append('ipisolate')
    command.append('-W')
    command.append(str(args.time))
    if args.bank is not None:
        command.append('-G')
        command.append(args.bank)
    command.append('-J')
    command.append(f'{args.jname_prefix}_{count}')
    # I am depending on the previous job
    if count != 0:
        command.append('-w')
        command.append('\'ended({0}_{1})\''.format(args.jname_prefix, count-1))

    command.append('-outdir')
    command.append(args.out_dir)
    command.append('-oo')
    command.append(f'{args.out_dir}/{args.jname_prefix}_{count}_%J.out')
    command.append('-e')
    command.append(f'{args.out_dir}/{args.jname_prefix}_{count}_%J.err')
    return ' '.join(command)

def Chain(config_fn, dry, experiment_args):
    """
    Schedules another job to be dependent to the current running
    job, if we have not reached the maximum number of chaing jobs
    """

    with open(config_fn, 'r') as fd:
        options = json.load(fd)

    args = argparse.Namespace(**options)
    # check if we can submit another job, if so submit it

    with open(args.train_config, 'r') as fd:
        exp_config = json.load(fd)
    state = get_state(argparse.Namespace(**exp_config))

    if state == 'Done':
        return

    if args.count < args.max_jobs:
        command =  create_bsub_command(args, state, args.count+1)
        # Before submitting update config
        # file to record the number
        # of submitted jobs
        options['count'] = args.count + 1
        with open(config_fn, 'w') as fd:
            json.dump(options, fd, indent=6)

        schedule(command, args.job_script, dry, [state])

def validate_deploy_config(deploy_args, args):
  """
  Checks whether the system  has sufficient memory etc,
  to deploy this run.

  Raise an exception and exit if the the config is not valid.
  """

  # TODO Brian add please your logic here
  pass

def Deploy(args, train_config_fn, hashVal, dry, experiment_args):
    """
    configures a configuration file for the upcoming training jobs,
    creates a bsub submission script and submits the first job to
    invoke the train script
    """

    # Convert namespace to dict
    config=vars(args)
    job_script_fn = str(Path('job_scripts') / (Path(hashVal).name + '.sh')).replace('-', '_')
    deploy_descr = f'job_scripts/{hashVal}_job_info.json'

    config.update( {'jname_prefix' : hashVal,
                    'train_config' : train_config_fn,
                    'job_script' : job_script_fn,
                    'deploy_descr' : deploy_descr
                    })

    args = argparse.Namespace(**config)

    with open(deploy_descr, 'w') as fd:
        json.dump(config, fd, indent=6)

    # Create submission script, use as template the 'template.sh' file
    # and replace key values accordingly
    with open('job_scripts/template.sh', 'r') as fd:
        template = Template(fd.read())
        replace = {'CONFIG_FILE' : train_config_fn,
                   'PROC_PER_NODE' : args.processes_per_node,
                   'JOB_CONFIG_FILE' : deploy_descr
                   }
        result = template.substitute(replace)

    # Write submission file to job_scripts
    with open (job_script_fn, 'w') as fd:
        fd.write(result)

    with open(args.train_config, 'r') as fd:
        exp_config = json.load(fd)
    state = get_state(argparse.Namespace(**exp_config))
    if state != 'Done':
        cmd = create_bsub_command(args, state)
        # Deploy always schedules first training setup.
        schedule(cmd, job_script_fn, dry, [state])


def main():
    parser = argparse.ArgumentParser(description='Options for submitting jobs '
                            'using BSUB/JSRUN and add dependencies among them',
                                     add_help=False)
    parser.add_argument('--dry', action='store_true', default=False,
                        help='Run script without submiting job.'
                        ' Just print out the submission command'
                        ' or the config options of the training')

    parser.add_argument('--help', '-h', action='store_true', default=False,
                        help='Print help message')

    args, extras = parser.parse_known_args()
    dry = args.dry
    print_help =  args.help

    train_parser = argparse.ArgumentParser(description='Options to submit a train job in LC LSF')

    train_parser = parser_train(train_parser)
    train_parser = set_extra_arguments(train_parser)

    # Parse only training related arguments
    # leave the rest for the different actions
    args, extras = train_parser.parse_known_args(extras)
    args = set_config_file_precedence(args)
    if (print_help):
        display_help(train_parser, extras, args)
        sys.exit()

    validate_train_arguments(args, train_parser)

    # Gather non default arguments and build
    # a hash out of those and name
    non_default_args = dict()
    for k, v in vars(args).items():
      if v != train_parser.get_default(k):
        non_default_args[k] = v

    if args.config:
        hashVal = Path(args.config).stem # We use the name of the configuration file as the hashvalue
        final_config_fn = args.config
    else:
        # Here I have some logic in creating
        # a unique name for the experiment
        # and allow it to be descriptive you can
        config_descr = '_'.join([f'{k}={non_default_args[k]}' for k in sorted(non_default_args.keys())])
        config_pretty = config_descr[:30]
        hashVal = hashlib.sha256(config_descr.encode('utf-8')).hexdigest();
        hashVal = f'{config_pretty}_{hashVal}'
        final_config_fn = f'configs/{hashVal}.json'
        Path('configs').mkdir(exist_ok=True)

    with open(final_config_fn, 'w') as fd:
        json.dump(non_default_args, fd, indent=6)

    subparser = train_parser.add_subparsers(dest='command')
    subparser.default = 'train'
    train = subparser.add_parser('train')
    deploy = subparser.add_parser('deploy')
    chain = subparser.add_parser('chain')
    info = subparser.add_parser('info')
    args, extras = train_parser.parse_known_args(extras, namespace=args)

    if args.command == 'info':
        print(final_config_fn)
    elif args.command == 'deploy':
        print('deploy')
        deploy = setup_deploy_args(deploy)
        deploy_args = deploy.parse_args(extras)
        validate_deploy_config(deploy_args, args)
        Deploy(deploy_args, final_config_fn, hashVal, dry, args)
    elif args.command == 'chain':
        chain = setup_chain_args(chain)
        chain_args = chain.parse_args(extras)
        Chain(chain_args.job_config, dry, args)
    elif args.command == 'train':
        print('train')
        if not dry:
            distributed_train(args)

if __name__ == '__main__':
  main()
