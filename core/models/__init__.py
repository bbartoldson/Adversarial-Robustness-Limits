import torch

from .resnet import Normalization
from .preact_resnet import preact_resnet
from .resnet import resnet
from .wideresnet import wideresnet

from .preact_resnetwithswish import preact_resnetwithswish
from .wideresnetwithswish import wideresnetwithswish
from .wideresnetwithRobustResBlock import wideresnetwithRobustResBlock

from .convnext_large import convnext_large

from core.data import DATASETS


MODELS = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 
          'preact-resnet18', 'preact-resnet34', 'preact-resnet50', 'preact-resnet101', 
          'wrn-28-10', 'wrn-32-10', 'wrn-34-10', 'wrn-34-20', 
          'preact-resnet18-swish', 'preact-resnet34-swish',
          'wrn-28-1-swish', 
          'wrn-28-10-swish', 'wrn-34-20-swish', 'wrn-70-16-swish']
#there are too many combos to list for the convnexts, but they look like this: {interpolation scheme, required}_{pretrained_, optional}convnext


def create_model(name, normalize, info, device, robustblock=False):
    """
    Returns suitable model from its name.
    Arguments:
        name (str): name of resnet architecture.
        normalize (bool): normalize input.
        info (dict): dataset information.
        device (str or torch.device): device to work on.
    Returns:
        torch.nn.Module.
    """
    if robustblock:
        if 'wrn' in name:
            assert 'relu' in name or 'swish' in name
            backbone = wideresnetwithRobustResBlock(name, num_classes=info['num_classes'], device=device)
        else:
            raise ValueError('Invalid model name {}!'.format(name))

    elif info['data'] in ['tiny-imagenet']:
        assert 'preact-resnet' in name, 'Only preact-resnets are supported for this dataset!'
        from .ti_preact_resnet import ti_preact_resnet
        backbone = ti_preact_resnet(name, num_classes=info['num_classes'], device=device)
    
    elif info['data'] in DATASETS and info['data'] not in ['tiny-imagenet']:
        if 'preact-resnet' in name and 'swish' not in name:
            backbone = preact_resnet(name, num_classes=info['num_classes'], pretrained=False, device=device)
        elif 'preact-resnet' in name and 'swish' in name:
            backbone = preact_resnetwithswish(name, dataset=info['data'], num_classes=info['num_classes'])
        elif 'resnet' in name and 'preact' not in name:
            backbone = resnet(name, num_classes=info['num_classes'], pretrained=False, device=device)
        elif 'wrn' in name and 'swish' not in name:
            backbone = wideresnet(name, num_classes=info['num_classes'], device=device)
        elif 'wrn' in name and 'swish' in name:
            backbone = wideresnetwithswish(name, dataset=info['data'], num_classes=info['num_classes'], device=device)
        elif 'convnext' in name:
            upsampler = name.split('_')[0]
            pretrained = 'pretrained' in name
            assert upsampler in ['identity', 'nearest', 'bilinear'], f'{upsampler} upsampler selected but must be identity, nearest or bilinear'
            backbone = convnext_large(pretrained=pretrained, dataset=info['data'], num_classes=info['num_classes'], upsampler=upsampler)
        else:
            raise ValueError('Invalid model name {}!'.format(name))
    
    else:
        raise ValueError('Models for {} not yet supported!'.format(info['data']))
        
    if normalize:
        model = torch.nn.Sequential(Normalization(info['mean'], info['std']), backbone)
    else:
        model = torch.nn.Sequential(backbone)
    
    #model = torch.nn.DataParallel(model) # brian commented out on 3/30/23 to avoid device mismatch issues
    model = model.to(device)
    return model
