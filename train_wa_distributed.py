import json
import time
import argparse
import shutil

import os, sys
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from core.data import get_data_info
from core.data import load_data
from core.data import SEMISUP_DATASETS

from core.utils import format_time
from core.utils import Logger
from core.utils import parser_train, set_extra_arguments
from core.utils import set_config_file_precedence
from core.utils import update_parsed_values
from core.utils import validate_train_arguments
from core.utils import Trainer
from core.utils import seed

from gowal21uncovering.utils import WATrainer

# distributed imports:
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from core.utils.distributed import CustomLSFEnvironment
from torch import distributed
import datetime

# weights and biases

from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator


def distributed_train(args):
    """
    Trains the model as defined by args, it expects args to be parsed and verified"
    """

    # Distributed training environment
    env = CustomLSFEnvironment()
    device = torch.device(f'cuda:{env.local_rank}')
    torch.cuda.set_device(env.local_rank)
    torch.distributed.init_process_group("nccl", init_method="env://",
                                            timeout=datetime.timedelta(seconds=3600),
                                            world_size=env.world_size,
                                            rank=env.rank)
    print("ENV:", env.__dict__)
    f = open(os.devnull, "w")
    if env.rank > 0:
        sys.stdout = f
        sys.stderr = f


    DATA_DIR = os.path.join(args.data_dir, args.data)
    LOG_DIR = os.path.join(args.log_dir, args.desc)
    WEIGHTS = os.path.join(LOG_DIR, 'weights-best.pt')
    sampler_path = os.path.join(LOG_DIR, 'sampler-state-last.pt')
    if env.rank==0:
        if os.path.exists(LOG_DIR) and not args.resume_path:
            print('No resume path given but logs exist, deleting prior run/log.')
            assert False, 'No resume path given but logs exist, deleting prior run/log is possible if you remove this assertion'
            shutil.rmtree(LOG_DIR)
        os.makedirs(LOG_DIR, exist_ok=True)
        logger = Logger(os.path.join(LOG_DIR, 'log-train.log'))

        with open(os.path.join(LOG_DIR, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)

    info = get_data_info(DATA_DIR)
    BATCH_SIZE = args.batch_size
    BATCH_SIZE_VALIDATION = args.batch_size_validation
    NUM_ADV_EPOCHS = args.num_adv_epochs

    # To speed up training
    torch.backends.cudnn.benchmark = False #True

    # Adversarial Training
    seed(args.seed)
    if args.tau:
        if env.rank==0:
            print ('Using WA.')
        trainer = WATrainer(info, args, distributed=env)
    else:
        assert False, 'Distributed not yet implemented for wa=False'
        trainer = Trainer(info, args)
    last_lr = args.lr

    if NUM_ADV_EPOCHS > 0:
        metrics = pd.DataFrame()
        old_score = [0.0, 0.0]
        test_adv_acc = None

    trainer.model = nn.SyncBatchNorm.convert_sync_batchnorm(trainer.model)
    trainer.init_optimizer(args.num_adv_epochs, args.pct_start)

    if args.resume_path:
        start_epoch = trainer.load_model_resume(os.path.join(args.resume_path, 'state-last.pt')) + 1
        old_score = trainer.load_current_best(os.path.join(args.resume_path, 'old_score.pt'))
        if env.rank==0 and start_epoch>1:
            logger.log(f'Resuming at epoch {start_epoch}')
    else:
        start_epoch = 1

    if args.pretrained_file and start_epoch==1:
        if env.rank==0:
            logger.log(f'Using pretrained models and optimizer from {args.pretrained_file}, and resetting opt LRs with new scheduler')
        trainer.load_model_resume(args.pretrained_file, no_schedule_update=True)
        trainer.scheduler.last_epoch = -1
        trainer.scheduler.optimizer._step_count = 0
        trainer.scheduler._step_count = 0
        assert not hasattr(trainer.scheduler, '_initial_step') # pytorch 1.12 didn't have this yet, manually setting the above vars then using "step" instead
        trainer.scheduler.step()

    trainer.model = DistributedDataParallel(trainer.model, device_ids=[env.local_rank], gradient_as_bucket_view=True)
    trainer.wa_model = DistributedDataParallel(trainer.wa_model, device_ids=[env.local_rank], gradient_as_bucket_view=True)

    # Load data
    seed(args.seed)
    args.__dict__.update({'start_epoch':start_epoch})
    train_dataset, test_dataset, eval_dataset, train_dataloader, test_dataloader, eval_dataloader = load_data(
        DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=args.augment, use_consistency=args.consistency, shuffle_train=True, 
        aux_data_filename=args.aux_data_filename, unsup_fraction=args.unsup_fraction, validation=True, distributed_env=env, 
        unfix_N_batches_per_epoch=args.unfix_N_batches_per_epoch, better_sampler=args.better_sampler, EDM_50_amount=args.EDM_50_amount,
        args=args
    )
    del train_dataset, test_dataset, eval_dataset

    if args.one_epoch and start_epoch>1:
        sup_start, unsup_start = torch.load(sampler_path)
        print(f'\nResuming training with sampler start indices {sup_start, unsup_start}\n')
        trainer.load_sampler_state(train_dataloader, sampler_path) 

    eval_acc = trainer.eval(test_dataloader)*100

    csv_row = {'config':args.desc, 'epoch': 0, 'auto_attack': None, 'examples_per_epoch': len(train_dataloader)*args.batch_size,
                'fwd_flops_per_example': None, 'seconds_per_example': None, 'parameters': None,
                'CW_train_acc': None, 'CW_train_loss': None, 'CW_test_acc': None, 'CW_test_loss': None,
                'PGD40_train_acc': None, 'PGD40_train_loss': None, 'PGD40_test_acc': None, 'PGD40_test_loss': None,
                'clean_train_acc': None, 'clean_train_loss': None, 'clean_test_acc': None, 'clean_test_loss': None}

    if start_epoch == 1:
        with get_accelerator().device(env.local_rank):
            flops, macs, params = get_model_profile(model=trainer.model,
                                            input_shape=(1, 3, 32, 32), 
                                    args=None, # list of positional arguments to the model.
                                    kwargs=None, # dictionary of keyword arguments to the model.
                                    print_profile=True, # prints the model graph with the measured profile attached to each module
                                    detailed=False, # print the detailed profile
                                    module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                                    top_modules=1, # the number of top modules to print aggregated profile
                                    warm_up=10, # the number of warm-ups before measuring the time of each module
                                    as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                    output_file=None, # path to the output file. If None, the profiler prints to stdout.
                                    ignore_modules=None) # the list of modules to ignore in the profiling

        update_result_row(csv_row, trainer, test_dataloader, train_dataloader, sampler_path if args.one_epoch else None)
        if env.rank==0:
            logger.log(f'FLOPs {flops}, MACs {macs}, Params {params}')
            csv_row['fwd_flops_per_example'], csv_row['parameters'] = flops, params
            write_result_row(os.path.join(LOG_DIR, 'results.csv'), csv_row)
            print(csv_row)

    if env.rank==0:
        logger.log('\n\n')
        logger.log('Standard Accuracy-\tTest: {:2f}%.'.format(eval_acc))
        logger.log('RST Adversarial training for {} epochs'.format(NUM_ADV_EPOCHS))

    for epoch in range(start_epoch, NUM_ADV_EPOCHS+1):
        start = time.time()
        if env.rank==0:
            logger.log('======= Epoch {} ======='.format(epoch))

        if args.scheduler:
            last_lr = trainer.scheduler.get_last_lr()[0]

        res, seconds_per_iter = trainer.train(train_dataloader, epoch=epoch, adversarial=True)
        test_acc = trainer.eval(test_dataloader)
        metric_epoch = epoch in [int(NUM_ADV_EPOCHS*num/40) for num in range(1,41)] or epoch in [1, 4, 8, 16, NUM_ADV_EPOCHS - 16, NUM_ADV_EPOCHS - 8, NUM_ADV_EPOCHS - 4, NUM_ADV_EPOCHS - 1, NUM_ADV_EPOCHS]
        if metric_epoch:
            csv_row['seconds_per_example'] = seconds_per_iter / args.batch_size
            update_result_row(csv_row, trainer, test_dataloader, train_dataloader, sampler_path if args.one_epoch else None, epoch=epoch)

        if env.rank==0:
            if metric_epoch:
                write_result_row(os.path.join(LOG_DIR, 'results.csv'), csv_row)
                print(csv_row)
            logger.log('Loss: {:.4f}.\tLR: {:.4f}'.format(res['loss'], last_lr))
            if 'clean_acc' in res:
                    logger.log('Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(res['clean_acc']*100, test_acc*100))
            else:
                    logger.log('Standard Accuracy-\tTest: {:.2f}%.'.format(test_acc*100))
        epoch_metrics = {'train_'+k: v for k, v in res.items()}
        epoch_metrics.update({'epoch': epoch, 'lr': last_lr, 'test_clean_acc': test_acc, 'test_adversarial_acc': ''})

        if epoch % args.adv_eval_freq == 0 or epoch == NUM_ADV_EPOCHS:
            test_adv_acc = trainer.eval(test_dataloader, adversarial=True)
            if env.rank==0:
                logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(res['adversarial_acc']*100,
                                                                                       test_adv_acc*100))
            epoch_metrics.update({'test_adversarial_acc': test_adv_acc})
        elif env.rank==0:
            logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.'.format(res['adversarial_acc']*100))

        eval_adv_acc = trainer.eval(eval_dataloader, adversarial=True)
        if env.rank==0:
            logger.log('Adversarial Accuracy-\tEval: {:.2f}%.'.format(eval_adv_acc*100))
        epoch_metrics['eval_adversarial_acc'] = eval_adv_acc

        if eval_adv_acc >= old_score[1]:
            old_score[0], old_score[1] = test_acc, eval_adv_acc
            if env.rank==0:
                trainer.save_model(WEIGHTS)
                trainer.save_current_best(old_score, os.path.join(args.resume_path, 'old_score.pt'))
        # trainer.save_model(os.path.join(LOG_DIR, 'weights-last.pt'))
        metrics = metrics.append(pd.DataFrame(epoch_metrics, index=[0]), ignore_index=True)
        if env.rank==0:
            if epoch % 10 == 0:
                trainer.save_model_resume(os.path.join(LOG_DIR, 'state-last.pt'), epoch, train_dataloader)
            if epoch % 400 == 0:
                shutil.copyfile(WEIGHTS, os.path.join(LOG_DIR, f'weights-best-epoch{str(epoch)}.pt'))
            if epoch in [4000, 6000, 7000, 8000]:
                trainer.save_model_resume(os.path.join(LOG_DIR, f'state-epoch{epoch}.pt'), epoch, train_dataloader)
            logger.log('Time taken: {}'.format(format_time(time.time()-start)))
            metrics.to_csv(os.path.join(LOG_DIR, 'stats_adv.csv'), index=False)

        if break_for_data_switch(args, epoch): # time to switch datasets
            distributed.barrier()
            break

    # Record metrics
    if epoch < NUM_ADV_EPOCHS: #we broke out of training early
        return
    train_acc = res['clean_acc'] if 'clean_acc' in res else trainer.eval(train_dataloader)
    if env.rank==0:
        logger.log('\nTraining completed.')
        logger.log('Standard Accuracy-\tTrain: {:.2f}%.\tTest: {:.2f}%.'.format(train_acc*100, old_score[0]*100))
        if NUM_ADV_EPOCHS > 0:
            logger.log('Adversarial Accuracy-\tTrain: {:.2f}%.\tEval: {:.2f}%.'.format(res['adversarial_acc']*100, old_score[1]*100))

        logger.log('Script Completed.')


def break_for_data_switch(args, epoch):
    if not args.one_epoch: return False
    if epoch == 7000: return True 
    if epoch%2000==0: return True
    return False

def update_result_row(csv_row, trainer, test_dataloader, train_dataloader, sampler_path, epoch = None):
    if sampler_path and distributed.get_rank()==0:
        trainer.save_sampler_state(train_dataloader, sampler_path) 
    print('\n**Getting metrics on test data**')
    csv_row['PGD40_test_acc'], csv_row['PGD40_test_loss'], csv_row['clean_test_acc'], csv_row['clean_test_loss'] = trainer.Linf_PGD_40(test_dataloader, 'CE')
    csv_row['CW_test_acc'], csv_row['CW_test_loss'], _, _ = trainer.Linf_PGD_40(test_dataloader, 'CW')
    print('\n**Getting metrics on training data**')
    csv_row['PGD40_train_acc'], csv_row['PGD40_train_loss'], csv_row['clean_train_acc'], csv_row['clean_train_loss'] = trainer.Linf_PGD_40(train_dataloader, 'CE')
    csv_row['CW_train_acc'], csv_row['CW_train_loss'], _, _ = trainer.Linf_PGD_40(train_dataloader, 'CW')
    if epoch:
        csv_row['epoch'] = epoch
    if sampler_path:
        trainer.load_sampler_state(train_dataloader, sampler_path) 

def write_result_row(file_name, csv_row):
    row = pd.DataFrame(csv_row, index=[0])
    if os.path.exists(file_name):
        metrics = pd.read_csv(file_name)
        metrics = pd.concat([metrics, row], ignore_index=True)
        metrics.to_csv(file_name, index=False)
    else:
        row.to_csv(file_name, index=False)

def main():
    parser = argparse.ArgumentParser(description='Standard + Adversarial Training.')
    parser = parser_train(parser)
    parser = set_extra_arguments(parser)
    args = parser.parse_args()
    args = set_config_file_precedence(args)
    validate_train_arguments(args, parser)

    distributed_train(args)

if __name__ == '__main__':
    main()

