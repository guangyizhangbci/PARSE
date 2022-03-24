'''
###Train Loops###
'''

import torch
import numpy as np
from library.optmization import Optmization
from library.utils import *
from library.optmization import update_ema_variables


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TrainLoop():
    def __init__(self, training_params):
        super(TrainLoop, self).__init__()

        self.params  = training_params


    '''
    ***MixMatch***
    '''

    def train_step_mix(self, inputs_x, targets_x, inputs_u, model, optimizer, unsupervised_weight):


        with torch.no_grad():
            # compute guessed labels of unlabel samples

            inputs_x_s, inputs_x_w = model(inputs_x, compute_model=False)
            inputs_u_s, inputs_u_w = model(inputs_u, compute_model=False)
            logits_u_s= model(inputs_u_s, compute_model=True)
            logits_u_w= model(inputs_u_w, compute_model=True)


            '''generate psudo-labels based on averaging model predictions of weakly and strongly augmented unlabeled data'''
            p = (torch.softmax(logits_u_s, dim=1) + torch.softmax(logits_u_w, dim=1)) / 2
            pt = p**(1/self.params['T'])

            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()


        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u_s, inputs_u_w], dim=0)
        all_targets = torch.cat([targets_x, targets_u,  targets_u],  dim=0)


        lam = np.random.beta(self.params['alpha'], self.params['alpha'])

        lam = max(lam, 1-lam)

        '''shuffle the index, interpolation between random pair of labeled and unlabeled samples'''
        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]


        mixed_input  = lam * input_a  + (1 - lam) * input_b
        mixed_target = lam * target_a + (1 - lam) * target_b


        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, self.params['batch_size']))
        mixed_input = interleave(mixed_input, self.params['batch_size'])


        logits = [model(mixed_input[0], compute_model=True)] # only use the output of model
        for input in mixed_input[1:]:
            logits.append(model(input, compute_model=True))


        # put interleaved samples back

        logits = interleave(logits, self.params['batch_size'])
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)


        probs_u = torch.softmax(logits_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * mixed_target[:self.params['batch_size']], dim=1))
        Lu = torch.mean((probs_u - mixed_target[self.params['batch_size']:])**2)

        unsupervised_weight = self.params['lambda_u'] * unsupervised_weight


        loss = Lx + unsupervised_weight * Lu

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ema_optimizer.step()

        return loss.item()


    '''
    ***FixMatch***
    '''

    def train_step_fix(self, inputs_x, targets_x, inputs_u, model, optimizer, ema_optimizer):


        with torch.no_grad():
            # compute guessed labels of unlabel samples
            inputs_u_s, inputs_u_w = model(inputs_u, compute_model=False)


        inputs = interleave_list(torch.cat((inputs_x, inputs_u_w, inputs_u_s)), self.params['batch_size'])

        logits = model(inputs, compute_model=True)
        logits = de_interleave_list(logits, self.params['batch_size'])
        logits_x = logits[:self.params['batch_size']]
        logits_u_w, logits_u_s = logits[self.params['batch_size']:].chunk(2)


        pseudo_label = torch.softmax(logits_u_w.detach()/self.params['T'], dim=-1)


        max_probs, targets_u_digital = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.params['threshold']).float() # mask if and only if greater or equal than self.params['threshold']


        targets_x_digital = torch.max(targets_x, 1)[1]

        Lx = F.cross_entropy(logits_x, targets_x_digital, reduction='mean')
        Lu = (F.cross_entropy(logits_u_s, targets_u_digital, reduction='none') * mask).mean()

        loss = Lx + Lu

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        return loss.item()



    '''
    ***AdaMatch***
    '''

    def train_step_ada(self, inputs_x, targets_x, inputs_u, model, optimizer, unsupervised_weight):

        with torch.no_grad():

            inputs_x_s, inputs_x_w = model(inputs_x, compute_model=False)
            inputs_u_s, inputs_u_w = model(inputs_u, compute_model=False)

            inputs_x_total  = torch.cat([inputs_x, inputs_x_w], 0)
            inputs_xu_total = torch.cat([inputs_x, inputs_x_w, inputs_u_s, inputs_u_w], 0)



            logits_xu_total = model(inputs_xu_total, compute_model=True)

        model.eval()
        z_d_prime_x = model(inputs_x_total, compute_model=True)

        model.train()
        z_prime_x = logits_xu_total[:inputs_x_total.shape[0]]


        '''Random Logits Interporlation'''

        lambd = torch.rand(inputs_x_total.shape[0], 3).to(device)
        # lambd = 1.0

        logits_x_total_final = lambd* z_prime_x + (1-lambd)* z_d_prime_x


        '''Distribution Alignment'''

        target_x_w = torch.softmax(logits_x_total_final[-inputs_x_w.shape[0]:].detach(), dim=-1)

        logits_u = logits_xu_total[inputs_x_total.shape[0]:]
        logits_u_s, logits_u_w = logits_u.chunk(2)
        target_u_w = torch.softmax(logits_u_w.detach(), dim=-1)


        ratio_expectation = torch.mean(target_x_w, 0)/torch.mean(target_u_w, 0)
        ratio_expectation = ratio_expectation.unsqueeze(0).repeat(inputs_x.shape[0],1)


        target_u_w_tiled = F.normalize(target_u_w*ratio_expectation, dim=1, p=1)




        # mixup
        all_inputs  = torch.cat([inputs_x, inputs_x_w, inputs_x_s], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_x], dim=0)



        lam = np.random.beta(self.params['alpha'], self.params['alpha'])

        lam = max(lam, 1-lam)


        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]


        mixed_input  = lam * input_a  + (1 - lam) * input_b
        mixed_target = lam * target_a + (1 - lam) * target_b


        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, self.params['batch_size']))
        mixed_input = interleave(mixed_input, self.params['batch_size'])


        logits = [model(mixed_input[0])] # only use the output of model
        for input in mixed_input[1:]:
            logits.append(model(input))


        # put interleaved samples back

        logits = interleave(logits, self.params['batch_size'])
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)



        '''Relative confidence thresholding'''

        row_wise_max = torch.max(target_x_w, dim=-1)[0]

        final_sum = torch.mean(row_wise_max, dim=0)
        c_tau = self.params['threshold'] * final_sum

        max_probs = torch.max(target_u_w_tiled, dim=-1)[0]
        # mask if and only if greater or equal than threshold

        mask = max_probs.ge(c_tau).float()



        targets_x_digital = torch.max(targets_x, 1)[1]

        mixed_x_digital = torch.max(mixed_target[:self.params['batch_size']], 1)[1]
        mixed_u_digital = torch.max(mixed_target[self.params['batch_size']:], 1)[1]




        target_u_w_tiled = Variable(target_u_w_tiled, requires_grad=False)
        target_u_w_tiled_digital = torch.max(target_u_w_tiled, 1)[1]


        '''Loss Function'''
        Lx =F.cross_entropy(logits_x_total_final[inputs_x_s.shape[0]:], targets_x_digital, reduction='mean')
        Lu = (F.cross_entropy(logits_u_s, target_u_w_tiled_digital, reduction='none') * mask)
        loss = Lx + unsupervised_weight* torch.mean(Lu, 0) + lam*F.cross_entropy(logits_x, mixed_x_digital, reduction='mean')


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        return loss.item()



    '''
    ***PARSE***
    '''

    def train_step_parse(self, inputs_x, targets_x, inputs_u, model, optimizer, unsupervised_weight):

        with torch.no_grad():
            '''no gradients are propgated during following steps'''

            '''apply strong and weak augmentation on labeled and unlabeled data'''
            inputs_x_s, inputs_x_w = model(inputs_x, compute_model=False)
            inputs_u_s, inputs_u_w = model(inputs_u, compute_model=False)

            inputs_total  = torch.cat([inputs_x_w, inputs_u, inputs_u_w, inputs_u_s], 0)

            logits_total =  model(inputs_total, compute_model=True)[0]

            logits_x_w, logits_u, logits_u_w, logits_u_s =  logits_total.chunk(4)

            '''generate emotion psudo-labels based on averaging model predictions of original, weakly and strongly augmented unlabeled data'''
            p = (torch.softmax(logits_u, dim=1) + torch.softmax(logits_u_w, dim=1) + torch.softmax(logits_u_s, dim=1)) / 3

            '''sharpen the averaged predictions'''
            pt = p**(1/self.params['T'])

            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()
            target_u_digital = torch.max(targets_u, 1)[1]


            target_u_w = torch.softmax(logits_u_w.detach()/self.params['T'], dim=-1)
            targets_u_w_digital = torch.max(target_u_w, dim=-1)[1]

            target_x_w = torch.softmax(logits_x_w.detach()/self.params['T'], dim=-1)
            targets_x_w_digital = torch.max(target_x_w, dim=-1)[1]


        '''Mixup'''

        inputs_x_total  = torch.cat([inputs_x, inputs_x_w, inputs_x_s], dim=0)
        target_x_total  = torch.cat([targets_x, targets_x, targets_x],  dim=0)

        inputs_u_total  = torch.cat([inputs_u, inputs_u_w, inputs_u_s], dim=0)
        target_u_total  = torch.cat([targets_u, targets_u, targets_u],  dim=0)


        all_inputs  = torch.cat([inputs_x_total, inputs_u_total], dim=0)
        all_targets = torch.cat([target_x_total, target_u_total], dim=0)


        '''lambda value is randomly generated within the range of [0,1] in each training batch. The random values obey Beta distribution'''
        lam = np.random.beta(self.params['alpha'],self.params['alpha'])


        input_a,  input_b   = inputs_x_total, inputs_u_total
        target_a, target_b  = target_x_total, target_u_total


        '''interpolated sets of EEG data and emotion labels '''
        mixed_input  = lam * input_a  + (1 - lam) * input_b
        mixed_target = lam * target_a + (1 - lam) * target_b


        '''interpolated set of domain labels (labeled Vs. unlabeled)'''
        label_dm_init  = torch.ones((mixed_input.shape[0], 1))
        mixed_label_dm = torch.cat([label_dm_init*lam, label_dm_init*(1-lam)], 1).to(device)
        mixed_label_dm_digital = torch.max(mixed_label_dm, 1)[1]

        all_inputs_final = torch.cat([inputs_x, inputs_u_s, mixed_input], dim=0)



        '''interleave labeled and unlabed samples between batches to get correct batchnorm calculation'''
        all_inputs_final = list(torch.split(all_inputs_final, self.params['batch_size']))
        all_inputs_final = interleave(all_inputs_final, self.params['batch_size'])


        total_logits_c, total_logits_d = [model(all_inputs_final[0])[0]], [model(all_inputs_final[0])[1]] # only use the output of model
        for input in all_inputs_final[1:]:
            total_logits_c.append(model(input)[0])
            total_logits_d.append(model(input)[1])


        '''put interleaved samples back'''
        total_logits_c = interleave(total_logits_c, self.params['batch_size'])
        total_logits_d = interleave(total_logits_d, self.params['batch_size'])

        '''classifier and discriminator outputs of interpolated data (EEG, emotion and domain labels)'''
        mixed_logits_c = torch.cat(total_logits_c[2:], dim=0)
        mixed_logits_d = torch.cat(total_logits_d[2:], dim=0)


        logits_c_x, logits_c_u_s = total_logits_c[0], total_logits_c[1]

        mixed_prob_c = torch.softmax(mixed_logits_c, dim=1)

        targets_x_digital = torch.max(targets_x, 1)[1]


        '''Loss Function'''
        Lx = F.cross_entropy(logits_c_x, targets_x_digital, reduction='mean')
        Lu = F.cross_entropy(logits_c_u_s, targets_u_w_digital,reduction='mean')
        L_dm = F.cross_entropy(mixed_logits_d, mixed_label_dm_digital, reduction='mean')
        L_cm =torch.mean((mixed_prob_c - mixed_target)**2)

        weight_1 = self.params['w_da']
        weight_2 = unsupervised_weight

        loss = Lx + weight_1*(lam*L_cm + L_dm)+ weight_2*torch.mean(Lu, 0)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        return loss.item()


    def eval_step(self, inputs, labels, model):

        if self.params['method'] == 'PARSE':
            outputs_classification = model(inputs, compute_model=True)[0]
        else:
            outputs_classification = model(inputs, compute_model=True)

        classification_pred = torch.max(outputs_classification, 1)[1]


        batch_size = labels.shape[0]
        digital_labels = torch.max(labels, 1)[1]
        # print(digital_labels.shape)

        err = nn.CrossEntropyLoss()(outputs_classification, digital_labels)

        y_true = digital_labels.detach().cpu().clone().numpy()
        y_pred = classification_pred.detach().cpu().clone().numpy()


        return err.item(), y_true, y_pred











    #
