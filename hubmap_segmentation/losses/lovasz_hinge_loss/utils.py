# Copyright (c) OpenMMLab. All rights reserved.
"""Modified from https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytor
ch/lovasz_losses.py Lovasz-Softmax and Jaccard hinge loss in PyTorch Maxim
Berman 2018 ESAT-PSI KU Leuven (MIT License)"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def lovasz_grad(gt_sorted):
    """Computes gradient of the Lovasz extension w.r.t sorted errors.
    See Alg. 1 in paper.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def flatten_probs(probs, labels, ignore_index=None):
    """Flattens predictions in the batch."""
    if probs.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probs.size()
        probs = probs.view(B, 1, H, W)
    B, C, H, W = probs.size()
    probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B*H*W, C=P,C
    labels = labels.view(-1)
    if ignore_index is None:
        return probs, labels
    valid = (labels != ignore_index)
    vprobs = probs[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobs, vlabels


def lovasz_softmax_flat(probs, labels, classes='present', class_weight=None):
    """Multi-class Lovasz-Softmax loss.
    Args:
        probs (torch.Tensor): [P, C], class probabilities at each prediction
            (between 0 and 1).
        labels (torch.Tensor): [P], ground truth labels (between 0 and C - 1).
        classes (str | list[int], optional): Classes chosen to calculate loss.
            'all' for all classes, 'present' for classes present in labels, or
            a list of classes to average. Default: 'present'.
        class_weight (list[float], optional): The weight for each class.
            Default: None.
    Returns:
        torch.Tensor: The calculated loss.
    """
    if probs.numel() == 0:
        # only void pixels, the gradients should be 0
        return probs * 0.
    C = probs.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        class_pred = probs[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        loss = torch.dot(errors_sorted, lovasz_grad(fg_sorted))
        if class_weight is not None:
            loss *= class_weight[c]
        losses.append(loss)
    return torch.stack(losses).mean()


def lovasz_softmax(probs,
                   labels,
                   classes='present',
                   class_weight=None,
                   ignore_index=255):
    """Multi-class Lovasz-Softmax loss.
    Args:
        probs (torch.Tensor): [B, C, H, W], class probabilities at each
            prediction (between 0 and 1).
        labels (torch.Tensor): [B, H, W], ground truth labels (between 0 and
            C - 1).
        classes (str | list[int], optional): Classes chosen to calculate loss.
            'all' for all classes, 'present' for classes present in labels, or
            a list of classes to average. Default: 'present'.
        per_image (bool, optional): If per_image is True, compute the loss per
            image instead of per batch. Default: False.
        class_weight (list[float], optional): The weight for each class.
            Default: None.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. This parameter only works when per_image is True.
            Default: None.
        ignore_index (int | None): The label index to be ignored. Default: 255.
    Returns:
        torch.Tensor: The calculated loss.
    """

    loss = lovasz_softmax_flat(
            *flatten_probs(probs, labels, ignore_index),
            classes=classes,
            class_weight=class_weight)
    return loss
