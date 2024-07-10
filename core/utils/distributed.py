import os
from torch import distributed, tensor

WORLD_SIZE_KEY = 'OMPI_COMM_WORLD_SIZE'
WORLD_RANK_KEY = 'OMPI_COMM_WORLD_RANK'
LOCAL_RANK_KEY = 'OMPI_COMM_LOCAL_RANK'
MASTER_ADDR_KEY = 'MASTER_ADDR'
MASTER_PORT_KEY = 'MASTER_PORT'
NUM_PROCESSES_PER_NODE_KEY = 'NUM_PROCESS_PER_NODE'

def all_reduce_sum(val, device):
    '''  
    given a scalar, sum it across all ranks 
    '''
    total = tensor(val, device=device) 
    distributed.all_reduce(total) # does sum reduction by default...
    return total  

def all_reduce_mean_and_reweight(mean, weight, device):
    '''  
    given a mean and count used to get it, get mean across all items in all ranks

    multiplies mean by weight to get total per rank,
    all_reduces totals and all_reduces weights to get global total and weight,
    then gets quotient to find mean across items 
    '''
    total = tensor(mean, device=device) * weight
    distributed.all_reduce(total) # does sum reduction by default...
    distributed.barrier()
    total_possible = tensor(weight, device=device)
    distributed.all_reduce(total_possible) # does sum reduction by default...
    return total / total_possible  # ...divide by sum of weights to get mean 

class CustomLSFEnvironment:
    """
    An environment for running on clusters managed by the LSF resource manager.
    """

    def __init__(self):
        self.world_size = int(os.environ[WORLD_SIZE_KEY])
        self.rank = int(os.environ[WORLD_RANK_KEY])
        self.num_processes_per_node = int(os.environ[NUM_PROCESSES_PER_NODE_KEY])
        self.local_rank = self.rank % self.num_processes_per_node
