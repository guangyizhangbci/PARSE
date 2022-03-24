#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import copy
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import os,sys,inspect
import yaml
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)




def load_config(name):
    with open(os.path.join(sys.path[0], name)) as file:
        config = yaml.safe_load(file)

    return config

config = load_config('dataset_params.yaml')



class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)




class Conv_EEG(nn.Module):
    def __init__(self, dataset_name, method_name):
        super(Conv_EEG, self).__init__()
        self.dataset   = dataset_name
        self.method    = method_name
        self.features  = config[self.dataset]['Feature_No']
        self.embed_dim = 10*(self.features-4)
        self.class_num = config[self.dataset]['Class_No']
        self.fn        = nn.Flatten() #  Flatten learnt embedding from convolutional layers


        self.encoder = nn.Sequential(
            nn.Conv1d(1, 5, 3, stride=1),
            nn.BatchNorm1d(5),
            nn.LeakyReLU(0.3),
            nn.Conv1d(5, 10, 3, stride=1),
            nn.BatchNorm1d(10),
            nn.LeakyReLU(0.3)
            )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(10, 5, 3, stride=1),
            nn.BatchNorm1d(5),
            nn.ReLU(0.3),
            nn.ConvTranspose1d(5, 1, 3, stride=1),
            nn.BatchNorm1d(1),
            nn.ReLU(0.3)
            )

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim,64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, self.class_num)
            )

        self.discriminator = nn.Sequential(
            nn.Linear(self.embed_dim,64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, self.class_num)
            )


    def network_flag(self):
        ae_flag  = False
        grl_flag = False

        if self.method=='PARSE':
            grl_flag = True
        elif self.method=='AutoEncoder':
            ae_flag = True
        else:
            pass

        return ae_flag, grl_flag

    def forward(self, input):

        ae_flag, grl_flag = self.network_flag()
        encoded_embed = self.encoder(input)
        flatten_embed = self.fn(encoded_embed)


        if grl_flag==True:
            output_c = self.classifier(flatten_embed)
            '''Add GRL before discriminator so that the gradients in the encoder will be reversed'''
            embed    = grad_reverse(flatten_embed)
            output_d = self.discriminator(embed)
            output= (output_c, output_d)

        elif ae_flag==True:
            reconstructed_input = self.decoder(flatten_embed)
            output = self.classifier(flatten_embed)
            output = (flatten_embed, reconstructed_input, output)

        else:
            output = self.classifier(flatten_embed)


        return output




#
