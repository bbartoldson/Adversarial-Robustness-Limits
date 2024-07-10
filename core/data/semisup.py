import os
import h5py
import sys
import pickle
import numpy as np
import torch
import torch.distributed as dist
import math
from typing import TypeVar, Optional, Iterator
from torch.utils.data import Sampler, Dataset
T_co = TypeVar('T_co', covariant=True)


def get_semisup_dataloaders(train_dataset, test_dataset, val_dataset=None, batch_size=256, batch_size_test=256, num_workers=4, 
                            unsup_fraction=0.5, distributed=None, unfix_N_batches_per_epoch=False, better_sampler=False, args=None):
    """
    Return dataloaders with custom sampling of pseudo-labeled data.
    """
    dataset_size = train_dataset.dataset_size
    num_batches = int(np.ceil(dataset_size/batch_size))
    if unfix_N_batches_per_epoch:
        num_batches = None 

    if not distributed:
        train_batch_sampler = SemiSupervisedSampler(train_dataset.sup_indices, train_dataset.unsup_indices, batch_size, 
                                                    unsup_fraction, num_batches=num_batches, args=args)
    else:
        train_batch_sampler = DistributedBatchSemiSupervisedSampler(train_dataset.sup_indices, train_dataset.unsup_indices, batch_size, 
                                                    unsup_fraction, num_batches=num_batches,
                                                    num_replicas=distributed.world_size, rank=distributed.rank, better_sampler=better_sampler, args=args)
        if distributed.rank==0:
            _nb = len(train_batch_sampler)
            print(f'\nTraining with {_nb} batches per epoch, {distributed.world_size} workers, and {batch_size//distributed.world_size} samples per worker batch.')
            print(f'\nTraining with {_nb*distributed.world_size*(batch_size//distributed.world_size)} samples per epoch.')
            if num_batches: 
                assert _nb == num_batches, (_nb, num_batches)
                assert dataset_size<=num_batches*distributed.world_size*(batch_size//distributed.world_size)<dataset_size*1.05
    epoch_size = len(train_batch_sampler) * batch_size

    # kwargs = {'num_workers': num_workers, 'pin_memory': torch.cuda.is_available() }
    kwargs = {'num_workers': num_workers, 'pin_memory': True}    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
    if not distributed:
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False, **kwargs)
    else:
        assert batch_size_test % distributed.world_size == 0, f'requested batch size test {batch_size_test} is not divisible by world size {num_replicas}'
        test_sampler = DistributedTestSampler(test_dataset, num_replicas=distributed.world_size, rank=distributed.rank)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test//distributed.world_size, shuffle=False, sampler=test_sampler, **kwargs)
        from core.utils.distributed import all_reduce_sum
        assert len(test_dataset) == all_reduce_sum(len(test_sampler), torch.device(f'cuda:{distributed.local_rank}'))
        if 0 < len(test_dataset) % batch_size_test < distributed.world_size:
            assert False, f'choose a batch_size_test that does not leave you with different batch counts on different ranks: you have a remainder of {len(test_dataset) % batch_size_test} samples split across {distributed.world_size} ranks'
    if val_dataset:
        if not distributed:
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_test, shuffle=False, **kwargs)
        else:
            val_sampler = DistributedTestSampler(val_dataset, num_replicas=distributed.world_size, rank=distributed.rank)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_test//distributed.world_size, shuffle=False, sampler=val_sampler, **kwargs)
            assert len(val_dataset) == all_reduce_sum(len(val_sampler), torch.device(f'cuda:{distributed.local_rank}'))
            if 0 < len(val_dataset) % batch_size_test < distributed.world_size:
                assert False, f'choose a batch_size_test that does not leave you with different batch counts on different ranks: you have a remainder of {len(val_dataset) % batch_size_test} samples split across {distributed.world_size} ranks'
        return train_dataloader, test_dataloader, val_dataloader
    return train_dataloader, test_dataloader


class SemiSupervisedDataset(torch.utils.data.Dataset):
    """
    A dataset with auxiliary pseudo-labeled data.
    """
    def __init__(self, base_dataset='cifar10', take_amount=None, take_amount_seed=13, aux_data_filename=None, 
                 add_aux_labels=False, aux_take_amount=None, train=False, validation=False, EDM_50_amount=None, env=None, start_epoch=1, **kwargs):

        self.base_dataset = base_dataset
        self.load_base_dataset(train, **kwargs)


        if validation:
            self.dataset.data = self.dataset.data[1024:]
            self.dataset.targets = self.dataset.targets[1024:]
        
        self.train = train

        if self.train:
            if take_amount is not None:
                rng_state = np.random.get_state()
                np.random.seed(take_amount_seed)
                take_inds = np.random.choice(len(self.sup_indices), take_amount, replace=False)
                np.random.set_state(rng_state)

                self.targets = self.targets[take_inds]
                self.data = self.data[take_inds]

            self.sup_indices = list(range(len(self.targets)))
            self.unsup_indices = []

            aux_count = 0
            virtual_file_path_list = None
            if aux_data_filename[0] == '!':
                # WARNING, individual files in path_dict are assumed to have 50M images each. 
                # if you want to add a file with a different count, change `make_virtual_dataset` accordingly 
                aux_data_kind = aux_data_filename[1:] 
                path_dict = {
                    'pfgmpp18': ['/p/vast1/MLdata/virtual/pfgmpp/18-steps/first-batch-50mil-pfgmpp-EDM-cifar10-18-steps.h5',
                                 '/p/vast1/MLdata/virtual/pfgmpp/18-steps/second-batch-50mil-pfgmpp-EDM-cifar10-18-steps.h5'] ,
                    'dg20': ['/p/vast1/MLdata/virtual/discriminator-guidance/20-steps/first-batch-50mil-dg-EDM-cifar10-20-steps.h5',
                             '/p/vast1/MLdata/virtual/discriminator-guidance/20-steps/second-batch-50mil-dg-EDM-cifar10-20-steps.h5'] ,
                    'vanilla20': ['/p/vast1/MLdata/virtual/edm50_shuffled.h5',
                                  '/p/vast1/MLdata/virtual/vanilla/20-steps/first-batch-vanilla-edm-50mil-20-steps.h5'], 
            
                    'vanilla20dg20pfgmpp18': ['/p/vast1/MLdata/virtual/edm50_shuffled.h5',
                           '/p/vast1/MLdata/virtual/discriminator-guidance/20-steps/first-batch-50mil-dg-EDM-cifar10-20-steps.h5',
                           '/p/vast1/MLdata/virtual/pfgmpp/18-steps/first-batch-50mil-pfgmpp-EDM-cifar10-18-steps.h5',
                           '/p/vast1/MLdata/virtual/vanilla/20-steps/first-batch-vanilla-edm-50mil-20-steps.h5',
                           '/p/vast1/MLdata/virtual/discriminator-guidance/20-steps/second-batch-50mil-dg-EDM-cifar10-20-steps.h5',
                           '/p/vast1/MLdata/virtual/pfgmpp/18-steps/second-batch-50mil-pfgmpp-EDM-cifar10-18-steps.h5'],

                    'vanilla20dg20pfgmpp18-150': ['/p/vast1/MLdata/virtual/edm50_shuffled.h5',
                           '/p/vast1/MLdata/virtual/discriminator-guidance/20-steps/first-batch-50mil-dg-EDM-cifar10-20-steps.h5',
                           '/p/vast1/MLdata/virtual/pfgmpp/18-steps/first-batch-50mil-pfgmpp-EDM-cifar10-18-steps.h5'],

                    'vanilla10': ['/p/vast1/MLdata/virtual/vanilla/10-steps/first-batch-vanilla-edm-30mil-10-steps.h5'],
                    'vanilla5': ['/p/vast1/MLdata/virtual/vanilla/5-steps/first-batch-vanilla-edm-30mil-5-steps.h5'],
                    'vanilla6': ['/p/vast1/MLdata/virtual/vanilla/6-steps/first-batch-vanilla-edm-30mil-6-steps.h5'],
                    'vanilla7': ['/p/vast1/MLdata/virtual/vanilla/7-steps/first-batch-vanilla-edm-30mil-7-steps.h5',
                                 '/p/vast1/MLdata/virtual/vanilla/7-steps/second-batch-vanilla-edm-40mil-7-steps.h5',
                                 '/p/vast1/MLdata/virtual/vanilla/7-steps/third-batch-vanilla-edm-30mil-7-steps.h5',
                                ]
                    } 
                count_dict = {'pfgmpp18': int(100e6),
                              'dg20': int(100e6), 
                              'vanilla20': int(100e6), 
                              'vanilla20dg20pfgmpp18': int(300e6),
                              'vanilla20dg20pfgmpp18-150': int(150e6),
                              'vanilla10': int(30e6), 
                              'vanilla5': int(30e6), 
                              'vanilla6': int(30e6), 
                              'vanilla7': int(100e6)
                             }
                assert aux_data_kind.replace('-341','') in path_dict, f'aux data name was {aux_data_filename}, which is not in {path_dict.keys()}'
                assert aux_data_kind.replace('-341','') in count_dict, f'aux data name was {aux_data_filename}, which is not in {count_dict.keys()}'
                aux_count = count_dict[aux_data_kind.replace('-341','')] 
                virtual_file_path_list = path_dict[aux_data_kind.replace('-341','')]
                if aux_data_kind == 'vanilla20dg20pfgmpp18-341':
                    aux_data_kind = ['vanilla20', 'dg20', 'pfgmpp18'][(start_epoch-1)//2000 % 3]
                    if 6000<=start_epoch-1: aux_data_kind = 'pfgmpp18'
                    aux_data_filename = '!'+aux_data_kind
                    print(f'\nOverride: Training on {aux_data_kind} data because start_epoch is {start_epoch}.\n')
                    aux_count = count_dict[aux_data_kind] 
                    virtual_file_path_list = path_dict[aux_data_kind]
                elif aux_data_kind=='vanilla20dg20pfgmpp18':
                    aux_data_kind = ['vanilla20', 'dg20', 'pfgmpp18'][(start_epoch-1)//2000 % 3]
                    # wrap the following three if statements in a check to see if it's a 10K epoch run
                    if 6000<=start_epoch-1<7000: aux_data_kind = 'vanilla20'
                    if 7000<=start_epoch-1<8000: aux_data_kind = 'dg20'
                    if 8000<=start_epoch-1: aux_data_kind = 'pfgmpp18'
                    aux_data_filename = '!'+aux_data_kind
                    print(f'\nOverride: Training on {aux_data_kind} data because start_epoch is {start_epoch}.\n')
                    aux_count = count_dict[aux_data_kind] 
                    virtual_file_path_list = path_dict[aux_data_kind]
            if aux_data_filename is not None:
                aux_path = aux_data_filename
                aux_count = aux_count or int(int(''.join(filter(str.isdigit, aux_path))[3:]) * 1e6)
                print(f'Loading {aux_count:,} EDM samples from {aux_path}')
                if os.path.splitext(aux_path)[1] == '.pickle':
                    # for data from Carmon et al, 2019.
                    with open(aux_path, 'rb') as f:
                        aux = pickle.load(f)
                    aux_data = aux['data']
                    aux_targets = aux['extrapolated_targets']
                elif not virtual_file_path_list and aux_count < 50e6:
                    # for data from Rebuffi et al, 2021.
                    aux = np.load(aux_path)
                    aux_data = aux['image']
                    print(aux_data.shape)
                    aux_targets = aux['label']
                    if aux_path != '/p/vast1/MLdata/CIFAR-10-EDM/discriminator-guidance/5m.npz':
                        assert aux_count == len(aux_targets), f'filename {aux_path} suggests {aux_count} samples, not {len(aux_targets)}'
                    else:
                        assert len(aux_targets) == 5e6+2700
                
                orig_len = len(self.data)

                if aux_take_amount is not None:
                    rng_state = np.random.get_state()
                    np.random.seed(take_amount_seed)
                    take_inds = np.random.choice(len(aux_data), aux_take_amount, replace=False)
                    np.random.set_state(rng_state)

                    aux_data = aux_data[take_inds]
                    aux_targets = aux_targets[take_inds]

                if aux_count < 50e6 and not virtual_file_path_list:
                    assert not EDM_50_amount, 'you must use the 50M image EDM dataset when specifying a value for the EDM_50_amount arg'
                    self.data = np.concatenate((self.data, aux_data), axis=0)
                else: # use virtual data to avoid memory issues
                    data_maker = env is None or env.rank == 0
                    path_to_virtual_data = "/p/vast1/MLdata/virtual/VDS.h5"
                    if virtual_file_path_list:
                        millions = int(aux_count/1e6) 
                        path_to_virtual_data = f"/p/vast1/MLdata/virtual/{aux_data_kind}_{millions}M_VDS.h5"
                    if not os.path.exists(path_to_virtual_data):
                        if data_maker:
                            make_virtual_dataset(path_to_virtual_data, self.data, orig_len, aux_count,
                                                 virtual_file_path_list=virtual_file_path_list)
                    if env is not None:
                        dist.barrier()
                    aux_targets = h5py.File('/p/vast1/MLdata/virtual/edm50_shuffled.h5', 'r')['label'][:].astype(int)
                    if virtual_file_path_list:
                        aux_targets = []
                        for f in virtual_file_path_list:
                            aux_targets.append(h5py.File(f, 'r')['label'][:].astype(int))
                        aux_targets = np.concatenate(aux_targets,axis=0) 
                    if EDM_50_amount:
                        assert EDM_50_amount*1e6 < aux_count, (aux_count, EDM_50_amount)
                        aux_count = int(math.ceil(EDM_50_amount*1e6))
                        aux_targets = aux_targets[:aux_count]
                        path_to_virtual_data = path_to_virtual_data.replace("VDS",f"VDS_{aux_count}")
                        if not os.path.exists(path_to_virtual_data):
                            if data_maker:
                                make_virtual_dataset(path_to_virtual_data, self.data, orig_len, aux_count,
                                                 virtual_file_path_list=virtual_file_path_list)
                        if env is not None:
                            dist.barrier()
                    self.data = h5py.File(path_to_virtual_data, "r")['image']
                    assert len(aux_targets) == aux_count, (len(aux_targets),aux_count)
                    assert len(aux_targets) == len(self.data) - orig_len, (len(aux_targets), len(self.data), orig_len)
                    print(f'\nLoaded virtual data with shape {self.data.shape}.\nOf this, {aux_count:,} images are from DM.')
                    print(f'\nDM data saved at {path_to_virtual_data}\n')

                if not add_aux_labels:
                    self.targets.extend([-1] * len(aux_targets))
                else:
                    self.targets.extend(aux_targets)
                self.unsup_indices.extend(range(orig_len, orig_len+len(aux_targets)))

        else:
            self.sup_indices = list(range(len(self.targets)))
            self.unsup_indices = []
    
    def load_base_dataset(self, **kwargs):
        raise NotImplementedError()
    
    @property
    def data(self):
        return self.dataset.data

    @data.setter
    def data(self, value):
        self.dataset.data = value

    @property
    def targets(self):
        return self.dataset.targets

    @targets.setter
    def targets(self, value):
        self.dataset.targets = value

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        self.dataset.labels = self.targets
        return self.dataset[item]


def make_virtual_dataset(path, self_data, orig_len, aux_count, virtual_file_path_list = None):
    file_lengths = {i:int(50e6) for i in range(1,10)}
    if '/p/vast1/MLdata/virtual/vanilla/7-steps/third-batch-vanilla-edm-30mil-7-steps.h5' in virtual_file_path_list:
        file_lengths = {i:int(x) for i,x in zip(range(1,4), [30e6,40e6,30e6])}
    file_lengths[0] = orig_len
    def concatenate(file_names_to_concatenate):
        entry_key = 'image'  # where the data is inside of the source files.
        layout = h5py.VirtualLayout(shape=(orig_len+aux_count,32,32,3),
                                    dtype=np.uint8)
        with h5py.File(path, 'w', libver='latest') as f:
            for i, filename in enumerate(file_names_to_concatenate):
                vsource = h5py.VirtualSource(filename, entry_key, shape=((file_lengths[i]),32,32,3))
                size = file_lengths[i]
                if i == 0:
                    layout[:size] = vsource
                    examples_added = size 
                else:
                    aux_examples_added = examples_added-orig_len
                    aux_examples_available = aux_examples_added+size
                    if aux_count < aux_examples_available:
                        size = aux_count - aux_examples_added 
                        assert size >= 0, (aux_count)
                    layout[examples_added:examples_added+size] = vsource[:size]
                    examples_added += size 
                print(f'From {filename}, added {size}/{aux_count+orig_len} examples. {examples_added}/{aux_count+orig_len} added so far')

            f.create_virtual_dataset(entry_key, layout, fillvalue=0)
    
    name1 = '/p/vast1/MLdata/virtual/c10.h5'
    if not os.path.exists(name1):
        with h5py.File(name=name1, mode='w') as f:
            d = f.create_dataset('image', self_data.shape, dtype=np.uint8)
            d[:] = self_data
    if not virtual_file_path_list:
        # the following file was created analogously using the shuffled EDM data -- see core/data/edm50_shuffled_save.py.
        name2 = '/p/vast1/MLdata/virtual/edm50_shuffled.h5'  
        concatenate([name1, name2])
    else:
        concatenate([name1]+virtual_file_path_list)

class SemiSupervisedDatasetSVHN(torch.utils.data.Dataset):
    """
    A dataset with auxiliary pseudo-labeled data.
    """
    def __init__(self, base_dataset='svhn', take_amount=None, take_amount_seed=13, aux_data_filename=None, 
                 add_aux_labels=False, aux_take_amount=None, train=False, validation=False, **kwargs):

        self.base_dataset = base_dataset
        self.load_base_dataset(train, **kwargs)
        self.dataset.labels = self.dataset.labels.tolist()


        if validation:
            self.dataset.data = self.dataset.data[1024:]
            self.dataset.labels = self.dataset.labels[1024:]
        
        self.train = train

        if self.train:
            if take_amount is not None:
                rng_state = np.random.get_state()
                np.random.seed(take_amount_seed)
                take_inds = np.random.choice(len(self.sup_indices), take_amount, replace=False)
                np.random.set_state(rng_state)

                self.targets = self.targets[take_inds]
                self.data = self.data[take_inds]

            self.sup_indices = list(range(len(self.targets)))
            self.unsup_indices = []

            if aux_data_filename is not None:
                aux_path = aux_data_filename
                print('Loading data from %s' % aux_path)
                if os.path.splitext(aux_path)[1] == '.pickle':
                    # for data from Carmon et al, 2019.
                    with open(aux_path, 'rb') as f:
                        aux = pickle.load(f)
                    aux_data = aux['data']
                    aux_targets = aux['extrapolated_targets']
                else:
                    # for data from Rebuffi et al, 2021.
                    aux = np.load(aux_path)
                    aux_data = aux['image']
                    print(aux_data.shape)
                    aux_targets = aux['label']

                orig_len = len(self.data)

                if aux_take_amount is not None:
                    rng_state = np.random.get_state()
                    np.random.seed(take_amount_seed)
                    take_inds = np.random.choice(len(aux_data), aux_take_amount, replace=False)
                    np.random.set_state(rng_state)

                    aux_data = aux_data[take_inds]
                    aux_targets = aux_targets[take_inds]

                self.data = np.concatenate((self.data, aux_data.transpose(0,3,1,2)), axis=0)

                if not add_aux_labels:
                    self.targets.extend([-1] * len(aux_data))
                else:
                    self.targets.extend(aux_targets)
                self.unsup_indices.extend(range(orig_len, orig_len+len(aux_data)))

        else:
            self.sup_indices = list(range(len(self.targets)))
            self.unsup_indices = []
    
    def load_base_dataset(self, **kwargs):
        raise NotImplementedError()
    
    @property
    def data(self):
        return self.dataset.data

    @data.setter
    def data(self, value):
        self.dataset.data = value

    @property
    def targets(self):
        return self.dataset.labels

    @targets.setter
    def targets(self, value):
        self.dataset.labels = value

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        self.dataset.labels = self.targets
        return self.dataset[item]
    
    
class SemiSupervisedSampler(torch.utils.data.Sampler):
    """
    Balanced sampling from the labeled and unlabeled data.
    """
    def __init__(self, sup_inds, unsup_inds, batch_size, unsup_fraction=0.5, num_batches=None, one_epoch=None):
        if unsup_fraction is None or unsup_fraction < 0:
            self.sup_inds = sup_inds + unsup_inds
            unsup_fraction = 0.0
        else:
            self.sup_inds = sup_inds
            self.unsup_inds = unsup_inds

        assert not one_epoch, 'not supported'

        self.batch_size = batch_size
        unsup_batch_size = int(batch_size * unsup_fraction)
        self.sup_batch_size = batch_size - unsup_batch_size

        if num_batches is not None:
            self.num_batches = num_batches
        else:
            self.num_batches = int(np.ceil(len(self.sup_inds) / self.sup_batch_size))
        super().__init__(None)

    def __iter__(self):
        batch_counter = 0
        while batch_counter < self.num_batches:
            sup_inds_shuffled = [self.sup_inds[i]
                                 for i in torch.randperm(len(self.sup_inds))]
            for sup_k in range(0, len(self.sup_inds), self.sup_batch_size):
                if batch_counter == self.num_batches:
                    break
                batch = sup_inds_shuffled[sup_k:(sup_k + self.sup_batch_size)]
                if self.sup_batch_size < self.batch_size:
                    batch.extend([self.unsup_inds[i] for i in torch.randint(high=len(self.unsup_inds), 
                                                                            size=(self.batch_size - len(batch),), 
                                                                            dtype=torch.int64)])
                np.random.shuffle(batch)
                yield batch
                batch_counter += 1

    def __len__(self):
        return self.num_batches

class BetterSemiSupervisedSampler(torch.utils.data.Sampler):
    """
    Balanced sampling from the labeled and unlabeled data.

    Better because:
    - goes through full sup dataset k times before seeing any sup example k+1 times
    - goes through full unsup dataset k times before seeing any unsup example k+1 times
    - works with unsup fraction = 1
    """
    def __init__(self, sup_inds, unsup_inds, batch_size, unsup_fraction=0.5, num_batches=None, one_epoch=False):
        if unsup_fraction is None or unsup_fraction < 0:
            self.sup_inds = sup_inds + unsup_inds
            unsup_fraction = 0.0
        else:
            self.sup_inds = sup_inds
            self.unsup_inds = unsup_inds

        self.batch_size = batch_size
        self.unsup_batch_size = int(math.ceil(batch_size * unsup_fraction))
        self.sup_batch_size = batch_size - self.unsup_batch_size

        print(f'-Training with (per GPU) unsup_batch_size {self.unsup_batch_size}')
        print(f'-Training with (per GPU) sup_batch_size {self.sup_batch_size}')

        self.one_epoch = one_epoch

        if num_batches is not None:
            self.num_batches = num_batches
        else:
            if self.sup_batch_size:
                self.num_batches = int(np.ceil(len(self.sup_inds) / self.sup_batch_size))
            else:
                self.num_batches = int(np.ceil(len(self.unsup_inds) / self.unsup_batch_size))

        self.sup_inds = self.shuffle(self.sup_inds)
        self.unsup_inds = self.shuffle(self.unsup_inds)
        self.sup_batch_start_idx = self.unsup_batch_start_idx = 0
        super().__init__(None)

    def shuffle(self, inds):
        if self.one_epoch:
            return inds
        inds = np.array(inds)
        np.random.shuffle(inds)
        return inds.tolist()

    def batch_maker(self, inds, start_idx, bs, batch=None):
        end_idx = start_idx + bs
        if batch is None:
            batch = inds[start_idx : end_idx]
        else:
            batch.extend(inds[start_idx : end_idx])
        start_idx  = end_idx % len(inds)
        if start_idx < end_idx:
            inds = self.shuffle(inds)
            batch.extend(inds[:start_idx])
        return batch, inds, start_idx

    def get_batch(self):
        # sup component
        batch, self.sup_inds, self.sup_batch_start_idx = self.batch_maker(
               self.sup_inds, self.sup_batch_start_idx, self.sup_batch_size, batch=None)
        # unsup component
        batch, self.unsup_inds, self.unsup_batch_start_idx = self.batch_maker(
               self.unsup_inds, self.unsup_batch_start_idx, self.unsup_batch_size, batch)
        assert len(batch) == self.batch_size, (len(batch), self.batch_size)
        return batch

    def __iter__(self):
        batch_counter = 0
        while batch_counter < self.num_batches:
            for _ in range(self.num_batches):
                batch = self.get_batch()
                yield batch
                batch_counter += 1

    def __len__(self):
        return self.num_batches

class DistributedBatchSemiSupervisedSampler(torch.utils.data.DistributedSampler):
    def __init__(self, sup_inds, unsup_inds, batch_size, unsup_fraction=0.5, num_batches=0,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 better_sampler = False, args = None) -> None:
        # shuffling happens inside SemiSupervisedSampler, it is not needed in the distributed sampler
        super().__init__(dataset=sup_inds, num_replicas=num_replicas, rank=rank, shuffle=False, drop_last=False)
        sup_inds = list(super().__iter__())
        unsup_inds = unsup_inds[rank::num_replicas]
        assert batch_size % num_replicas == 0, f'requested batch size {batch_size} is not divisible by world size {num_replicas}'
        batch_size = batch_size // num_replicas
        batch_sampler = SemiSupervisedSampler
        assert not (args.one_epoch and better_sampler), 'You are trying to use two different samplers'
        if better_sampler or args.one_epoch:
            print(f'\nUsing better sampler! State of sampler will only be loaded prior to resuming training if args.one_epoch == 1, it is {args.one_epoch}.')
            batch_sampler = BetterSemiSupervisedSampler
        self.train_batch_sampler = batch_sampler(sup_inds, unsup_inds, batch_size, unsup_fraction, num_batches=num_batches, one_epoch=args.one_epoch)

    def __iter__(self):
        return iter(self.train_batch_sampler)

    def __len__(self) -> int:
        return len(self.train_batch_sampler)


class DistributedTestSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    differs from DistributedSampler by allowing a different number of samples
    per rank on the final batch, which avoids evaluating the same test
    example more than once (the behavior of DistributedSampler)

    """

    def __init__(self, dataset: Dataset, num_replicas: int, rank: int) -> None:
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.num_samples = len(self.dataset) // self.num_replicas + (len(self.dataset) % self.num_replicas > rank)
        self.total_size = len(self.dataset)

    def __iter__(self) -> Iterator[T_co]:
        indices = list(range(self.total_size))  # type: ignore[arg-type]

        # subsample
        indices = indices[self.rank::self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
