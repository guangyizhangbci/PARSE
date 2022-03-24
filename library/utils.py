#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu June 18 10:16:37 2021

@author: patrick
"""


import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import math
import umap
import matplotlib.pyplot as plt


class WeightClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-1,1)

class WeightInit(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = nn.init.normal_(W, 0.0, 0.02)


def Average(lst):
    return sum(lst) / len(lst)

def to_categorical(y):
    """ 1-hot encodes a tensor """
    num_classes = len(np.unique(y))
    return np.eye(num_classes, dtype='uint8')[y.astype(int)]


def load_dataset_to_device(data, label, batch_size, class_flag=False, shuffle_flag=True):


    if class_flag==True:
        label = np.ravel(label)
        label = to_categorical(label)

    data, label  = torch.Tensor(data), torch.Tensor(label)

    dataset = torch.utils.data.TensorDataset(data, label)
    dataset = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                shuffle=shuffle_flag, num_workers=2, drop_last=True,  pin_memory=True)

    return dataset


def train_val_split(labels, n_labeled_per_class, random_seed, class_num):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    np.random.seed(random_seed)

    for i in range(class_num):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-50])
        val_idxs.extend(idxs[-50:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


def train_split(labels, n_labeled_per_class, random_seed, class_num):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []

    np.random.seed(random_seed)
    for i in range(class_num):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class: ])

    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)

    return train_labeled_idxs, train_unlabeled_idxs



def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]

    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def interleave_list(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave_list(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])
