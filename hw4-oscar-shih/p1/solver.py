import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.optim.lr_scheduler import StepLR
from model import Protonet
from util import pairwise_distances


class Solver(object):
    def __init__(self, config, train_loader, val_loader):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.N_way_train = config.N_way_train
        self.N_shot_train = config.N_shot_train
        self.N_query_train = config.N_query_train
        self.N_way_val = config.N_way_val
        self.N_shot_val = config.N_shot_val
        self.N_query_val = config.N_query_val
        self.matching_fn = config.matching_fn

        self.num_epochs = config.num_epochs
        self.resume_iter = config.resume_iter
        self.lr = config.lr
        self.num_steps_decay = config.num_steps_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.weight_decay = config.weight_decay
        self.exp_name = config.name
        os.makedirs(config.ckp_dir, exist_ok=True)
        self.ckp_dir = os.path.join(config.ckp_dir, self.exp_name)
        os.makedirs(self.ckp_dir, exist_ok=True)
        self.log_interval = config.log_interval
        self.ckp_interval = config.ckp_interval

        self.build_model()

    def build_model(self):
        self.model = Protonet().to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr,  betas=[self.beta1, self.beta2], weight_decay=self.weight_decay)
        if self.matching_fn == 'parametric':
            self.parametric = nn.Sequential(
                nn.Linear(800, 400),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(400, 1)
            ).to(self.device)
            self.optimizer = torch.optim.AdamW(list(self.model.parameters()) + list(self.parametric.parameters()), lr=self.lr,  betas=[self.beta1, self.beta2], weight_decay=self.weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=self.num_steps_decay, gamma=0.9)
        print(self.model)

    def save_checkpoint(self, step):
        state = {'state_dict': self.model.state_dict(),
                 'optimizer' : self.optimizer.state_dict()}
        if self.matching_fn == 'parametric':
            state = {'state_dict': self.model.state_dict(),
                     'optimizer' : self.optimizer.state_dict(),
                     'parametric': self.parametric.state_dict()}

        new_checkpoint_path = os.path.join(self.ckp_dir, '{}-protonet.pth'.format(step + 1))
        torch.save(state, new_checkpoint_path)
        print('model saved to %s' % new_checkpoint_path)

    def load_checkpoint(self, resume_iter):
        print('Loading the trained models from step {}...'.format(resume_iter))
        new_checkpoint_path = os.path.join(self.ckp_dir, '{}-protonet.pth'.format(resume_iter))
        state = torch.load(new_checkpoint_path)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        if self.matching_fn == 'parametric':
            self.parametric.load_state_dict(state['parametric'])
        print('model loaded from %s' % new_checkpoint_path)
    
    def train(self):
        criterion = nn.CrossEntropyLoss()

        best_mean = 0
        iteration = 0

        if self.resume_iter:
            print("resuming step %d ..."% self.resume_iter)
            iteration = self.resume_iter
            self.load_checkpoint(self.resume_iter)
            loss, mean, std = self.eval()
            if mean > best_mean:
                best_mean = mean

        episodic_acc = []

        for ep in range(self.num_epochs):
            self.model.train()

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                
                support_input = data[:self.N_way_train * self.N_shot_train,:,:,:] 
                query_input   = data[self.N_way_train * self.N_shot_train:,:,:,:]

                label_encoder = {target[i * self.N_shot_train] : i for i in range(self.N_way_train)}
                query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[self.N_way_train * self.N_shot_train:]])

                support = self.model(support_input)
                queries = self.model(query_input)
                prototypes = support.reshape(self.N_way_train, self.N_shot_train, -1).mean(dim=1)

                if self.matching_fn == 'parametric':
                    distances = pairwise_distances(queries, prototypes, self.matching_fn, self.parametric)

                else:
                    distances = pairwise_distances(queries, prototypes, self.matching_fn)

                loss = criterion(-distances, query_label)
                loss.backward()
                self.optimizer.step()

                y_pred = (-distances).softmax(dim=1).max(1, keepdim=True)[1]
                episodic_acc.append(1. * y_pred.eq(query_label.view_as(y_pred)).sum().item() / len(query_label))


                if (iteration + 1) % self.log_interval == 0:
                    episodic_acc = np.array(episodic_acc)
                    mean = episodic_acc.mean()
                    std = episodic_acc.std()

                    print('Epoch: {:3d} [{:d}/{:d}]\tIteration: {:5d}\tLoss: {:.6f}\tAccuracy: {:.2f} +- {:.2f} %'.format(
                        ep, (batch_idx + 1), len(self.train_loader), iteration + 1, loss.item(), 
                        mean * 100, 1.96 * std / (self.log_interval)**(1/2) * 100))
                    episodic_acc = []


                if (iteration + 1) % self.ckp_interval == 0:
                    loss, mean, std = self.eval()
                    if mean > best_mean:
                        best_mean = mean
                        self.save_checkpoint(iteration)

                iteration += 1
            self.scheduler.step()


    def eval(self):
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        episodic_acc = []
        loss = []
        
        with torch.no_grad():
            for b_idx, (data, target) in enumerate(self.val_loader):
                data = data.to(self.device)
                support_input = data[:self.N_way_val * self.N_shot_val,:,:,:] 
                query_input = data[self.N_way_val * self.N_shot_val:,:,:,:]

                label_encoder = {target[i * self.N_shot_val] : i for i in range(self.N_way_val)}
                query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[self.N_way_val * self.N_shot_val:]])

                support = self.model(support_input)
                queries = self.model(query_input)
                prototypes = support.reshape(self.N_way_val, self.N_shot_val, -1).mean(dim=1)

                if self.matching_fn == 'parametric':
                    distances = pairwise_distances(queries, prototypes, self.matching_fn, self.parametric)
                else:
                    distances = pairwise_distances(queries, prototypes, self.matching_fn)
                    
                loss.append(criterion(-distances, query_label).item())
                y_pred = (-distances).softmax(dim=1).max(1, keepdim=True)[1]
                episodic_acc.append(1. * y_pred.eq(query_label.view_as(y_pred)).sum().item() / len(query_label))

        loss = np.array(loss)
        episodic_acc = np.array(episodic_acc)
        loss = loss.mean()
        mean = episodic_acc.mean()
        std = episodic_acc.std()

        print('\nLoss: {:.6f}\tAccuracy: {:.2f} +- {:.2f} %\n'.format(loss,mean * 100, 1.96 * std / (600)**(1/2) * 100))

        return loss, mean, std
