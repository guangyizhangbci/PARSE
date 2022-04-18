from __future__ import print_function
import argparse
import os
import shutil
import time
import random
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchsummary import summary
import matplotlib.animation as animation
from torch.autograd import Variable
import os,sys,inspect
from library.train_loop import TrainLoop
from library.optmization import Optmization
from library.optmization import ema_model, WeightEMA
from library.model import Conv_EEG
import math
from library.utils import *
import yaml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import f1_score, accuracy_score
from copy import deepcopy

parser = argparse.ArgumentParser(description='PyTorch SSL Training')
# Optimization options
parser.add_argument('--dataset', default='SEED-V', type=str, metavar='N',
                    help='dataset name')
parser.add_argument('--method', default='PARSE', type=str, metavar='N',
                    help='method name')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=8, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--n-labeled', type=int, default=25,
                        help='Number of labeled data')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--T', default=1, type=float,
                    help='pseudo label temperature')
parser.add_argument('--w-da', default=1.0, type=float,
                    help='data distribution weight')
parser.add_argument('--weak-aug', default=0.2, type=float,
                    help='weak aumentation')
parser.add_argument('--strong-aug', default=0.8, type=float,
                    help='strong aumentation')
parser.add_argument('--threshold', default=0.95, type=float,
                    help='pseudo label threshold')
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--init-weight', default=0, type=float)
parser.add_argument('--end-weight',  default=30, type=float)


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


def load_config(name):
    with open(os.path.join(sys.path[0], name)) as file:
        config = yaml.safe_load(file)

    return config

config = load_config('dataset_params.yaml')

checkpoint_save_path_subject = './PARSE/SEED_V/ckpt/labeled_{}/'.format(args.n_labeled)

if not os.path.exists(checkpoint_save_path_subject):
    try:
        os.makedirs(checkpoint_save_path_subject)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        pass


class Model(nn.Module):
    def __init__(self, Conv_EEG):
        super(Model, self).__init__()
        self.model = Conv_EEG(args.dataset, args.method)


    def augmentation(self, input, std):

        input_shape =input.size()
        noise = torch.normal(mean=0.5, std=std, size =input_shape)
        noise = noise.to(device)

        return input + noise

    def forward(self, input, compute_model=True):

        if compute_model==False:
            input_s  = self.augmentation(input, args.strong_aug)
            input_w  = self.augmentation(input, args.weak_aug)

            output = (input_s, input_w)

        else:
            if args.method=='PARSE':
                output_c, output_d = self.model(input)
                output = (output_c, output_d)
            else:
                output = self.model(input)
        return output



def eval_process(Net, X_test, Y_test, n_labeled_per_class, subject_num, fold_num):

    data_test, label_test  = np.expand_dims(X_test, axis=1), Y_test
    batch_size = args.batch_size
    test_dataset = load_dataset_to_device(data_test, label_test, batch_size=batch_size, class_flag=True, shuffle_flag=False)

    result = eval(Net, test_dataset, subject_num, fold_num)

    return result




def eval(Net, test_dataset, subject_num, fold_num):

    training_params = {'method': args.method, 'batch_size': args.batch_size}

    '''load_pre-trained_model'''
    optimizer = optim.Adam(Net.parameters(), args.lr)
    ckpt_path = os.path.join(checkpoint_save_path_subject, 'subject_{}_fold_{}.ckpt'.format(subject_num, fold_num+1))

    checkpoint = torch.load(ckpt_path)
    Net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    Net.eval()

    with torch.no_grad():
        test_ytrue_batch = []
        test_ypred_batch = []

        for image_batch, label_batch in test_dataset:
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)

            loss, y_true, y_pred  = TrainLoop(training_params).eval_step(image_batch,label_batch, Net)


            test_ytrue_batch.append(y_true)
            test_ypred_batch.append(y_pred)


        test_ypred_epoch = np.array(test_ypred_batch).flatten()
        test_ytrue_epoch = np.array(test_ytrue_batch).flatten()


    metric = accuracy_score(test_ytrue_epoch, test_ypred_epoch)

    return metric


def net_init(model):
    '''load and initialize the model'''
    Net = Model(model).to(device)
    Net.apply(WeightInit)
    Net.apply(WeightClipper)

    return Net



if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    '''set random seeds for torch and numpy libraies'''
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)


    ''' data and label address loader for SEED-V dataset '''

    data_addr  = './PARSE/DATA/SEED_V/EEG/de/{}_{}.npy'      # Subject_No, Fold_No
    label_addr = './PARSE/DATA/SEED_V/EEG/label/{}_{}.npy'   # Subject_No, Fold_No


    n_labeled_per_class = args.n_labeled # number of labeled samples need to be chosen for each emotion class

    '''create result directory'''
    directory = './PARSE/{}_result/ssl_method_{}/eval/'.format(args.dataset, args.method)

    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            pass

    dataset_dict = config[args.dataset]
    acc_array = np.zeros((dataset_dict['Subject_No'], dataset_dict['Fold_No']))

    for subject_num in range(1, dataset_dict['Subject_No']+1):

        X1 = np.load(data_addr.format(subject_num, 1))
        X2 = np.load(data_addr.format(subject_num, 2))
        X3 = np.load(data_addr.format(subject_num, 3))

        X  = np.vstack((X1, X2, X3))

        Y1 = np.load(label_addr.format(subject_num, 1))
        Y2 = np.load(label_addr.format(subject_num, 2))
        Y3 = np.load(label_addr.format(subject_num, 3))

        Y  = np.vstack((Y1, Y2, Y3))

        scaler=MinMaxScaler()
        X = scaler.fit_transform(X)

        for fold_num in range(dataset_dict['Fold_No']):

            Net = net_init(Conv_EEG)

            optimizer = optim.Adam(Net.parameters(), lr=args.lr)
            # ema_optimizer= WeightEMA(Net, ema_Net, alpha=args.ema_decay)

            fold_1_index = [i for i in range(0, len(X1))]
            fold_2_index = [i for i in range(len(X1), len(X1)+len(X2))]
            fold_3_index = [i for i in range(len(X1)+len(X2), len(X1)+len(X2)+len(X3))]

            if fold_num ==0:
                train_index, test_index = fold_1_index + fold_2_index, fold_3_index
            elif fold_num ==1:
                train_index, test_index = fold_2_index + fold_3_index, fold_1_index
            else:
                train_index, test_index = fold_3_index + fold_1_index, fold_2_index


            X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]

            acc_array[subject_num-1, fold_num] = eval_process(Net, X_test, Y_test, n_labeled_per_class, subject_num, fold_num)

    np.savetxt(os.path.join(directory, 'acc_labeled_{}.csv').format(n_labeled_per_class), acc_array , delimiter=",")






#
