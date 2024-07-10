import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm
import time

import os
import copy
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.attacks import create_attack
from core.attacks import CWLoss
from core.metrics import accuracy, num_right
from core.models import create_model

from core.utils import ctx_noparamgrad_and_eval
from core.utils import Trainer
from core.utils import set_bn_momentum
from core.utils import seed

from .trades import trades_loss, trades_loss_LSE
from .cutmix import cutmix


from core.utils.distributed import all_reduce_mean_and_reweight


class WATrainer(object):
    """
    Helper class for training a deep neural network with model weight averaging (identical to Gowal et al, 2020).
    Arguments:
        info (dict): dataset information.
        args (dict): input arguments.
    """
    def __init__(self, info, args, distributed=None):
        if not distributed:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(f'cuda:{distributed.local_rank}')
        self.distributed = distributed

        super().__init__()

        seed(args.seed)

        self.params = args
        self.model = create_model(args.model, args.normalize, info, self.device,
                                  robustblock=args.robustblock)
        self.wa_model = copy.deepcopy(self.model)

        num_samples = 50000 if 'cifar' in self.params.data else 73257
        num_samples = 100000 if 'tiny-imagenet' in self.params.data else num_samples
        if self.params.data in ['cifar10', 'cifar10s', 'svhn', 'svhns']:
            self.num_classes = 10
        elif self.params.data in ['cifar100', 'cifar100s']:
            self.num_classes = 100
        elif self.params.data == 'tiny-imagenet':
            self.num_classes = 200
        self.update_steps = int(np.floor(num_samples/self.params.batch_size) + 1)
        self.warmup_steps = 0.025 * self.params.num_adv_epochs * self.update_steps
    
    
    def init_optimizer(self, num_epochs, pct_start):
        """
        Initialize optimizer and schedulers.
        """
        def group_weight(model):
            group_decay = []
            group_no_decay = []
            for n, p in model.named_parameters():
                if 'batchnorm' in n:
                    group_no_decay.append(p)
                else:
                    group_decay.append(p)
            assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
            groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
            return groups

        # trade off memory for speed with the following
        #from torch.distributed.optim import ZeroRedundancyOptimizer
        #self.optimizer = ZeroRedundancyOptimizer(group_weight(self.model),optimizer_class = torch.optim.SGD,
        #        lr=self.params.lr, weight_decay=self.params.weight_decay,momentum=0.9, nesterov=self.params.nesterov)
        
        self.optimizer = torch.optim.SGD(group_weight(self.model), lr=self.params.lr, weight_decay=self.params.weight_decay, 
                                         momentum=0.9, nesterov=self.params.nesterov)
        if num_epochs <= 0:
            return
        self.init_scheduler(num_epochs, pct_start)
   
 
    def init_scheduler(self, num_epochs, pct_start=0.025):
        """
        Initialize scheduler.
        """
        if self.params.scheduler == 'cyclic':
            num_samples = 50000 if 'cifar10' in self.params.data else None
            num_samples = 100000 if 'tiny-imagenet' in self.params.data else num_samples
            update_steps = int(np.floor(num_samples/self.params.batch_size) + 1)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.params.lr, pct_start=pct_start,
                                                                 steps_per_epoch=update_steps, epochs=int(num_epochs))
        elif self.params.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=0.1, milestones=[100, 105])    
        elif self.params.scheduler == 'cosine':
            self.scheduler = CosineLR(self.optimizer, max_lr=self.params.lr, epochs=int(num_epochs))
        elif self.params.scheduler == 'cosinew':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.params.lr, pct_start=pct_start, 
                                                                 total_steps=int(num_epochs))
        else:
            self.scheduler = None
    
    
    def train(self, dataloader, epoch=0, adversarial=False, verbose=False):
        """
        Run one epoch of training.
        """
        metrics = pd.DataFrame()
        self.model.train()
        
        update_iter = 0
        for data in tqdm(dataloader, desc='Epoch {}: '.format(epoch), disable=not verbose):
            start_time = time.time()
            global_step = (epoch - 1) * self.update_steps + update_iter
            if global_step == 0:
                # make BN running mean and variance init same as Haiku
                set_bn_momentum(self.model, momentum=1.0)
            elif global_step == 1:
                set_bn_momentum(self.model, momentum=0.01)
            update_iter += 1
            
            x, y = data
            if self.params.consistency:
                x_aug1, x_aug2, y = x[0].to(self.device), x[1].to(self.device), y.to(self.device)
                if self.params.beta is not None:
                    loss, batch_metrics = self.trades_loss_consistency(x_aug1, x_aug2, y, beta=self.params.beta)

            else:
                if self.params.CutMix:
                    x_all, y_all = torch.tensor([]), torch.tensor([])
                    for i in range(4): # 128 x 4 = 512 or 256 x 4 = 1024
                        x_tmp, y_tmp = x.detach(), y.detach()
                        x_tmp, y_tmp = cutmix(x_tmp, y_tmp, alpha=1.0, beta=1.0, num_classes=self.num_classes)
                        x_all = torch.cat((x_all, x_tmp), dim=0)
                        y_all = torch.cat((y_all, y_tmp), dim=0)
                    x, y = x_all.to(self.device), y_all.to(self.device)
                else:
                    x, y = x.to(self.device), y.to(self.device)
                
                if adversarial:
                    if self.params.beta is not None and self.params.mart:
                        loss, batch_metrics = self.mart_loss(x, y, beta=self.params.beta)
                    elif self.params.beta is not None and self.params.LSE:
                        loss, batch_metrics = self.trades_loss_LSE(x, y, beta=self.params.beta)
                    elif self.params.beta is not None:
                        loss, batch_metrics = self.trades_loss(x, y, beta=self.params.beta)
                    else:
                        loss, batch_metrics = self.adversarial_loss(x, y)
                else:
                    loss, batch_metrics = self.standard_loss(x, y)
                    
            loss.backward()
            if self.params.clip_grad:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_grad)
            self.optimizer.step()
            if self.params.scheduler in ['cyclic']:
                self.scheduler.step()
            
            global_step = (epoch - 1) * self.update_steps + update_iter
            ema_update(self.wa_model, self.model, global_step, decay_rate=self.params.tau, 
                       warmup_steps=self.warmup_steps, dynamic_decay=True)
            metrics = metrics.append(pd.DataFrame(batch_metrics, index=[0]), ignore_index=True)
            if update_iter == 6: # 6 allows some warmup
                seconds_per_iter = time.time() - start_time
        
        if self.params.scheduler in ['step', 'converge', 'cosine', 'cosinew']:
            self.scheduler.step()
        
        update_bn(self.wa_model, self.model) 

        # `metrics.mean()` is not quite right because batches may have different sizes, but 
        # the important thing to matching the original implementation that runs w/out DDP
        # is using all_reduce to average each GPU's mean
        if self.distributed:
            return {k:v.item() for k,v in zip( metrics.keys(), all_reduce_mean_and_reweight(metrics.mean(),len(metrics),self.device) )}, seconds_per_iter

        return dict(metrics.mean())
    
    
    def trades_loss(self, x, y, beta):
        """
        TRADES training.
        """
        loss, batch_metrics = trades_loss(self.model, x, y, self.optimizer, step_size=self.params.attack_step, 
                                          epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                          beta=beta, attack=self.params.attack, label_smoothing=self.params.ls,
                                          use_cutmix=self.params.CutMix,
                                          device=self.device)
        return loss, batch_metrics

    def trades_loss_consistency(self, x_aug1, x_aug2, y, beta):
        """
        TRADES training with Consistency.
        """
        x = torch.cat([x_aug1, x_aug2], dim=0)
        loss, batch_metrics = trades_loss(self.model, x, y.repeat(2), self.optimizer, step_size=self.params.attack_step, 
                                          epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                          beta=beta, attack=self.params.attack, label_smoothing=self.params.ls,
                                          use_cutmix=self.params.CutMix, use_consistency=True, cons_lambda=self.params.cons_lambda, cons_tem=self.params.cons_tem,
                                          device=self.device)
        return loss, batch_metrics

    def trades_loss_LSE(self, x, y, beta):
        """
        TRADES training with LSE loss.
        """
        loss, batch_metrics = trades_loss_LSE(self.model, x, y, self.optimizer, step_size=self.params.attack_step, 
                                          epsilon=self.params.attack_eps, perturb_steps=self.params.attack_iter, 
                                          beta=beta, attack=self.params.attack, label_smoothing=self.params.ls,
                                          clip_value=self.params.clip_value,
                                          use_cutmix=self.params.CutMix,
                                          num_classes=self.num_classes,
                                          device=self.device)
        return loss, batch_metrics  

    
    def eval(self, dataloader, adversarial=False):
        """
        Evaluate performance of the model.
        """
        if not hasattr(self, 'eval_attack'):
            self.eval_attack = create_attack(self.wa_model, CWLoss, self.params.attack, self.params.attack_eps, 
                                            4*self.params.attack_iter, self.params.attack_step)

        acc = 0
        total = 0
        self.wa_model.eval()
        
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            if adversarial:
                with ctx_noparamgrad_and_eval(self.wa_model):
                    x_adv, _ = self.eval_attack.perturb(x, y)            
                with torch.no_grad():
                    out = self.wa_model(x_adv)
            else:
                out = self.wa_model(x)
            acc += num_right(y, out)
            total += len(x)
        acc /= total 
    
        if self.distributed:
            return all_reduce_mean_and_reweight(acc, total, self.device).item()
        return acc

    
    def Linf_PGD_40(self, dataloader, loss):
        """
        Evaluate performance of the model with the following assumptions

        beta = 5.0, 
        attack, eps, iters, step size = 'linf-pgd', 8/255, 40, 2/255
        adv examples generated with CE-based PGD40 or CW-based PGD40, depending on `loss` 
        """
        if not hasattr(self, 'ce_pgd40_attack'):
            # model, loss, attack, eps, iters, step size
            self.ce_pgd40_attack = create_attack(self.wa_model, nn.CrossEntropyLoss(reduction="sum"), 'linf-pgd', 8/255, 40, 2/255)
        if not hasattr(self, 'cw_pgd40_attack'):
            # model, loss, attack, eps, iters, step size
            self.cw_pgd40_attack = create_attack(self.wa_model, CWLoss, 'linf-pgd', 8/255, 40, 2/255)

        if loss == 'CE':
            attack = self.ce_pgd40_attack
        if loss == 'CW':
            attack = self.cw_pgd40_attack

        adv_acc, adv_loss, clean_acc, clean_loss = 0, 0, 0, 0
        total = 0
        criterion_ce = nn.CrossEntropyLoss(reduction="sum")
        criterion_kl = nn.KLDivLoss(reduction='sum')
        
        self.wa_model.eval()
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            with ctx_noparamgrad_and_eval(self.wa_model):
                x_adv, _ = attack.perturb(x, y)  

            with torch.no_grad():
                logits_clean = self.wa_model(x)
                logits_adv = self.wa_model(x_adv)
            
            clean_acc += num_right(y, logits_clean)
            __clean_loss = criterion_ce(logits_clean, y)
            clean_loss += __clean_loss

            adv_acc += num_right(y, logits_adv)
            __adv_loss = criterion_kl(F.log_softmax(logits_adv, dim=1), 
                                      F.softmax(logits_clean, dim=1))
            adv_loss += __clean_loss + 5.0 * __adv_loss # beta = 5

            total += len(x)

        if self.distributed:
            return (all_reduce_mean_and_reweight(var/total, total, self.device).item() for var in [adv_acc, adv_loss, clean_acc, clean_loss])
        return adv_acc/total , adv_loss/total , clean_acc/total , clean_loss/total 


    def save_model(self, path):
        """
        Save model weights.
        """
        torch.save({
            'model_state_dict': self.wa_model.state_dict(), 
            'unaveraged_model_state_dict': self.model.state_dict()
        }, path)


    def save_model_resume(self, path, epoch, train_dataloader):
        """
        Save model weights and optimizer.
        """
        torch.save({
            'model_state_dict': self.wa_model.state_dict(), 
            'unaveraged_model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(), 
            'scheduler_state_dict': self.scheduler.state_dict(), 
            'epoch': epoch
        }, path)

        self.save_sampler_state(train_dataloader, path.replace('state', 'sampler-state'))

    def save_sampler_state(self, train_dataloader, path):
        sampler = train_dataloader.batch_sampler.train_batch_sampler
        if hasattr(sampler,'sup_batch_start_idx'):
            sup, unsup = sampler.sup_batch_start_idx, sampler.unsup_batch_start_idx
            print(f'Saving sampler state {sup,unsup}, which will be used if one_epoch==True to prevent use of previously seen data across interrupted training periods')
            torch.save((sup, unsup), path)
        else:
            print(f'Not saving sampler state, model may see duplicate data.')

    def load_sampler_state(self, train_dataloader, path):
        sampler = train_dataloader.batch_sampler.train_batch_sampler
        if hasattr(sampler,'sup_batch_start_idx'):
            sampler.sup_batch_start_idx, sampler.unsup_batch_start_idx = torch.load(path)
            print(f'Loading sampler state {sampler.sup_batch_start_idx, sampler.unsup_batch_start_idx}, which will be used to prevent use of previously seen data across interrupted training periods')
        else:
            print(f'Not loading sampler state, model may see duplicate data.')

    def save_current_best(self, old_score, path):
        """
        Save best acc.
        """
        torch.save(old_score, path)
    
    def load_current_best(self, path):
        """
        Load best acc.
        """
        if os.path.exists(path):
            return torch.load(path)
        print('Training job is resuming with no prior old_score information!')
        return [0,0]
    
    def load_model(self, path):
        """
        Load model weights.
        """
        if not os.path.exists(path):
            raise RuntimeError(f'checkpoint not found at path {path}')
        checkpoint = torch.load(path)
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError('Model weights not found at {}.'.format(path))
        msd, umsd = self.distributed_safe_load(checkpoint)
        self.wa_model.load_state_dict(msd)
    

    def load_model_resume(self, path, no_schedule_update=False):
        """
        load model weights and optimizer.
        """
        if not os.path.exists(path):
            print(f'No model found at resume_path, training from scratch!')
            return 0
        print(f'Loading checkpoint from {path}.')
        checkpoint = torch.load(path, map_location='cpu')
        if 'model_state_dict' not in checkpoint:
            raise RuntimeError('Model weights not found at {}.'.format(path))
        msd, umsd = self.distributed_safe_load(checkpoint)
        self.wa_model.load_state_dict(msd)
        self.model.load_state_dict(umsd)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        no_schedule_update or self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']

    def distributed_safe_load(self, checkpoint):
        msd = checkpoint['model_state_dict']
        umsd = checkpoint['unaveraged_model_state_dict']
        if 'module'==list(checkpoint['model_state_dict'].keys())[0][:6]:
            for k in list(checkpoint['model_state_dict'].keys()):
                assert k[:7] == 'module.'
                msd[k[7:]] = msd[k]
                del msd[k]
            for k in list(checkpoint['unaveraged_model_state_dict'].keys()):
                assert k[:7] == 'module.'
                umsd[k[7:]] = umsd[k]
                del umsd[k]
        return msd, umsd

def ema_update(wa_model, model, global_step, decay_rate=0.995, warmup_steps=0, dynamic_decay=True):
    """
    Exponential model weight averaging update.
    """
    factor = int(global_step >= warmup_steps)
    if dynamic_decay:
        delta = global_step - warmup_steps
        decay = min(decay_rate, (1. + delta) / (10. + delta)) if 10. + delta != 0 else decay_rate
    else:
        decay = decay_rate
    decay *= factor
    
    for p_swa, p_model in zip(wa_model.parameters(), model.parameters()):
        p_swa.data *= decay
        p_swa.data += p_model.data * (1 - decay)


@torch.no_grad()
def update_bn(avg_model, model):
    """
    Update batch normalization layers.
    """
    avg_model.eval()
    model.eval()
    for module1, module2 in zip(avg_model.modules(), model.modules()):
        if isinstance(module1, torch.nn.modules.batchnorm._BatchNorm):
            module1.running_mean = module2.running_mean
            module1.running_var = module2.running_var
            module1.num_batches_tracked = module2.num_batches_tracked
