'''
Training Loops
'''
import numpy as np
import torch
import math

class Optmization():
    def __init__(self, optmization_params):
        super(Optmization, self).__init__()
        self.params   = optmization_params
        self.lr       = self.params['lr']
        self.current_epoch = self.params['current_epoch']
        self.total_epochs  = self.params['total_epochs']
        self.current_batch = self.params ['current_batch']
        self.max_iters     = self.params ['max_iterations']


    def ada_weight(self):
        '''Warm-up function'''
        '''Smoothly raises from 0 to 1 for the first half of the training and remains at 1 for the second half'''
        total_steps  = self.total_epochs * self.max_iters
        current_step = self.current_epoch * self.max_iters + self.current_batch
        pi = np.float32(np.pi)

        return 0.5 - np.cos(np.minimum(pi, np.float32((2*pi*current_step)/total_steps)))/2


    def decayed_learning_rate(self):
        '''Slowly decreases learning rate'''
        alpha = 0.25
        step = self.current_epoch*self.max_iters + self.current_batch
        decay_steps  = self.total_epochs*self.max_iters
        step = min(step, decay_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * step / decay_steps))
        decayed = (1 - alpha) * cosine_decay + alpha
        return self.lr * decayed


    def linear_rampup(self):
        ''''Slowly increase the weight applied on unsupervised loss'''
        current = self.current_epoch + self.current_batch/self.max_iters
        rampup_length = self.total_epochs
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current / rampup_length, 0.0, 1.0)
            return float(current)


    def cal_consistency_weight(self):
        '''Sets the weights for the consistency loss'''
        epoch    = self.current_epoch
        init_ep  = 0
        end_ep   = self.params['total_epochs']
        init_w   = self.params['init_w']
        end_w    = self.params['end_w']

        if epoch > end_ep:
            weight_cl = end_w
        elif epoch < init_ep:
            weight_cl = init_w
        else:
            T = float(epoch - init_ep)/float(end_ep - init_ep)
            #weight_mse = T * (end_w - init_w) + init_w #linear
            weight_cl = (math.exp(-5.0 * (1.0 - T) * (1.0 - T))) * (end_w - init_w) + init_w #exp
        #print('Consistency weight: %f'%weight_cl)
        return weight_cl



def update_ema_variables(model, model_teacher, global_step):
    # Use the true average until the exponential average is more correct
    ema_constant = 0.95
    gamma = min(1.0 - 1.0 / float(global_step + 1), ema_constant)
    for param_t, param in zip(model_teacher.parameters(), model.parameters()):
        param_t.data.mul_(gamma).add_(1 - gamma, param.data)



def ema_model(model):

    for param in model.parameters():
        param.detach_()

    return model



class WeightEMA(object):
    def __init__(self, model, ema_model, alpha, lr):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)








#
