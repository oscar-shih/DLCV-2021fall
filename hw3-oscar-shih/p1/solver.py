import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import Model
from transformers import get_cosine_schedule_with_warmup

class Solver(object):
    def __init__(self, trainset_loader, valset_loader, config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.trainset_loader = trainset_loader
        self.valset_loader = valset_loader
        self.epoch = config.epoch
        self.lr = config.lr
        self.iteration = config.resume_iter
        self.exp_name = config.name
        self.log_interval = config.log_interval
        self.save_interval = config.save_interval
        os.makedirs(config.ckp_dir, exist_ok=True)
        self.ckp_dir = os.path.join(config.ckp_dir, self.exp_name)
        self.build_model()

    def build_model(self):
        self.model = Model().cuda()
        # print(self.model)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, 5, 20)

        if self.iteration != 0:
            checkpoint_path = os.path.join(self.ckp_dir, 'model.ckpt')
            self.load_checkpoint(checkpoint_path)
        

    def save_checkpoint(self, checkpoint_path):
        state = {'state_dict': self.model.state_dict(),
                 'optimizer' : self.optimizer.state_dict()}
        torch.save(state, checkpoint_path, _use_new_zipfile_serialization = False)
        print('model saved to %s' % checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        print('model loaded from %s' % checkpoint_path)
    
    def training(self):
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        iteration = 0
        for ep in range(self.epoch):
            self.scheduler.step()
            self.model.train()
            checkpoint_path = os.path.join(self.ckp_dir, 'model%i.ckpt' % ep)
            # TODO: finish training function
            for batch_idx, (data, target) in enumerate(self.trainset_loader):
                data, target = data.cuda(), target.cuda()
                self.optimizer.zero_grad()
                # print(data.shape)
                output = self.model(data).squeeze(0)
                loss = criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                if iteration % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Lr: {}'.format(
                        ep, batch_idx * len(data), len(self.trainset_loader.dataset),
                        100. * batch_idx / len(self.trainset_loader), loss.item(), self.optimizer.param_groups[0]['lr']))
                if iteration % self.save_interval == 0 and iteration > 0:
                    self.valid()
                    self.save_checkpoint(checkpoint_path)
                iteration += 1
        self.save_checkpoint(checkpoint_path='ckpt/final.ckpt')

    def valid(self):
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        val_loss = 0
        val_acc  = 0
        with torch.no_grad():
            for data, target in self.valset_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += criterion(output, target).item()
                pred = output.max(1, keepdim=True)[1]
                val_acc += pred.eq(target.view_as(pred)).sum().item()
        val_loss /= len(self.valset_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss, val_acc, len(self.valset_loader.dataset),
            100. * val_acc / len(self.valset_loader.dataset)))
