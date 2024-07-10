import torch

def accuracy(true, preds):
    """
    Computes multi-class accuracy.
    Arguments:
        true (torch.Tensor): true labels.
        preds (torch.Tensor): predicted labels.
    Returns:
        Multi-class accuracy.
    """
    accuracy = (torch.softmax(preds, dim=1).argmax(dim=1) == true).sum().float()/float(true.size(0))
    return accuracy.item()

def num_right(true, preds):
    """
    Computes multi-class num_right.
    Arguments:
        true (torch.Tensor): true labels.
        preds (torch.Tensor): predicted labels.
    Returns:
        Multi-class num_right.
    """
    num_right = (torch.softmax(preds, dim=1).argmax(dim=1) == true).sum().float()
    return num_right.item()
