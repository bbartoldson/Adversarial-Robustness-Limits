import sys, os
robust_residual_network_dir = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/'))
sys.path.append(robust_residual_network_dir)
from robust_residual_network.resnet import PreActResNet

    
def wideresnetwithRobustResBlock(name,
                                dataset='cifar10',
                                num_classes=10,
                                device='cpu',
                                block_type = 'robust_res_block'):
    """
    Returns suitable Wideresnet model with Swish activation function from its name.
    Arguments:
        name (str): name of resnet architecture.
        num_classes (int): number of target classes.
        device (str or torch.device): device to work on.
        dataset (str): dataset to use.
    Returns:
        torch.nn.Module.
    """

    name_parts = name.split('-')
    depth = int(name_parts[1])
    width_mult = [int(name_parts[2])]*3
    activations = [name_parts[3]]*3 # they find swish works better when using "advanced training"
    print (f'WideResNet-{depth}-{width_mult[0]}-{activations[0]} uses normalization, block type {block_type}.')
    # "advanced training" setup (e.g., uses weight averaging)
    stride_config = [1, 2, 2] # this is standard
    scales, base_width, cardinality, se_reduction = 8, 10, 4, 64 # they found these values do well

    def convert_depth_to_num_blocks_per_superblock(depth):
        assert (depth - 4) % 6 == 0
        return (depth - 4) // 6
    #depth, width_mult = [4, 4, 4], [10, 10, 10] # WRN-28-10
    #depth, width_mult = [11, 11, 11], [16, 16, 16] # WRN-70-16
    depth = [convert_depth_to_num_blocks_per_superblock(depth)]*3
    channels = [16, 16 * width_mult[0], 32 * width_mult[1], 64 * width_mult[2]]
    # can be robust_res_block or basic_block. 
    # the latter does not use "SAME" padding in first conv like DM paper, otherwise equal
    block_types =[block_type]*3

    model = PreActResNet(
        num_classes=num_classes,
        channel_configs=channels, depth_configs=depth,
        stride_config=stride_config, stem_stride=1,
        block_types=block_types,
        activations=activations,
        normalizations=('BatchNorm', 'BatchNorm', 'BatchNorm'), # use BatchNorm
        use_init=True,
        cardinality=cardinality,
        base_width=base_width,
        scales=scales,
        se_reduction=se_reduction,
        pre_process=True # Default is False, but they use True for "advanced training" ()
    )

    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6} million parameters.")

    return model