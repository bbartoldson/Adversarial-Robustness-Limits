# Code borrowed from https://github.com/deepmind/deepmind-research/blob/master/adversarial_robustness/pytorch/model_zoo.py
# (Gowal et al 2020)

from typing import Tuple, Union

import math
import torch
import torch.nn as nn
from torchvision.models import convnext_large as pytorch_convnext_large


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)
SVHN_MEAN = (0.5, 0.5, 0.5)
SVHN_STD = (0.5, 0.5, 0.5)

#_ACTIVATION = {
#    'relu': nn.ReLU,
#    'swish': nn.SiLU,
#}

class ConvNextLarge(nn.Module):
    """
    ConvNextLarge model
    Arguments:
        num_classes (int): number of output classes.
        mean (tuple): mean of dataset.
        std (tuple): standard deviation of dataset.
        pretrained (bool): Set `True` to use pretrained ImageNet-1K weights
    """
    def __init__(self,
                 num_classes: int = 10,
                 mean: Union[Tuple[float, ...], float] = CIFAR10_MEAN,
                 std: Union[Tuple[float, ...], float] = CIFAR10_STD,
                 pretrained: bool = False,
                 num_input_channels: int = 3,
                 upsampler = None):
        super().__init__()
        self.num_classes = num_classes
        self.mean = torch.tensor(mean).view(num_input_channels, 1, 1)
        self.std = torch.tensor(std).view(num_input_channels, 1, 1)
        self.mean_cuda = None
        self.std_cuda = None
        self.pretrained = pretrained

        if self.pretrained: # Use pretrained imagenet weights
            torch.hub.set_dir('/p/vast1/MLdata/cached_models')
            self.net = pytorch_convnext_large(weights='IMAGENET1K_V1')
        else:
            self.net = pytorch_convnext_large()

        # Update number of output features to num_classes
        if self.num_classes != 1000:
            self.net.classifier[2] = torch.nn.Linear(self.net.classifier[2].in_features, num_classes, bias = self.net.classifier[2].bias is not None)

        if upsampler == 'bilinear':
            self.upsample = torch.nn.Upsample((224,224), mode='bilinear', align_corners=True)
        elif upsampler == 'nearest': 
            self.upsample = torch.nn.Upsample((224,224), mode='nearest') # this looks more like cifar-10 and less smooth than bilinear
        elif upsampler == 'identity':
            self.upsample = torch.nn.Identity() # this does nothing
        else:
            assert False, f'upsampler must be bilinear, nearest, or identity, you gave {upsampler}'
        print(f'\nUsing upsampling with {upsampler} strategy\n')
        
    
    def forward(self, x):
        if x.is_cuda:
            if self.mean_cuda is None:
                self.mean_cuda = self.mean.cuda()
                self.std_cuda = self.std.cuda()
            out = (x - self.mean_cuda) / self.std_cuda
        else:
            out = (x - self.mean) / self.std

        out = self.upsample(out)
        
        out = self.net(out)
        return out
    
    
def convnext_large(pretrained=False, dataset='cifar10', num_classes=10, upsampler=None):
    """
    Returns suitable ConvNext-Large with random or pretrained ImageNet-1K weights.
    Arguments:
        pretrained (bool): Set `True` to use pretrained ImageNet-1K weights
        num_classes (int): number of target classes.
        dataset (str): dataset to use.
    Returns:
        torch.nn.Module.
    """

    if pretrained:
        print (f'Pretrained ConvNext-Large uses normalization.')
    else:
        print (f'Randomly initialized ConvNext-Large uses normalization.')
    if 'cifar100' in dataset:
        return ConvNextLarge(num_classes=num_classes, mean=CIFAR100_MEAN, std=CIFAR100_STD, pretrained=pretrained, upsampler=upsampler)
    elif 'svhn' in dataset:
        return ConvNextLarge(num_classes=num_classes, mean=SVHN_MEAN, std=SVHN_STD, pretrained=pretrained, upsampler=upsampler)
    return ConvNextLarge(num_classes=num_classes, pretrained=pretrained, upsampler=upsampler)
