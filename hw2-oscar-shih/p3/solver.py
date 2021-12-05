import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from model import DANN


class Solver(object):
    def __init__(self, src_trainset_loader, src_valset_loader, tgt_trainset_loader=None, config=None):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.src_trainset_loader = src_trainset_loader
        self.src_valset_loader = src_valset_loader
        self.tgt_trainset_loader = tgt_trainset_loader
        self.num_iters = config.num_iters
        self.resume_iters = config.resume_iters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.src_only = config.src_only
        self.exp_name = config.name
        os.makedirs(config.ckp_dir, exist_ok=True)
        self.ckp_dir = os.path.join(config.ckp_dir, self.exp_name)
        os.makedirs(self.ckp_dir, exist_ok=True)
        self.log_interval = config.log_interval
        self.save_interval = config.save_interval
        self.use_wandb = config.use_wandb
        
        self.build_model()

    def build_model(self):
        self.model = DANN().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,  betas=[self.beta1, self.beta2])

    def save_checkpoint(self, step):
        state = {'state_dict': self.model.state_dict(),
                 'optimizer' : self.optimizer.state_dict()}
        new_checkpoint_path = os.path.join(self.ckp_dir, '{}-dann.pth'.format(step + 1))
        torch.save(state, new_checkpoint_path)
        print('model saved to %s' % new_checkpoint_path)

    def load_checkpoint(self, resume_iters):
        print('Loading the trained models from step {}...'.format(resume_iters))
        new_checkpoint_path = os.path.join(self.ckp_dir, '{}-dann.pth'.format(resume_iters))
        state = torch.load(new_checkpoint_path)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        print('model loaded from %s' % new_checkpoint_path)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optimizer.zero_grad()
    
    def train(self):
        criterion = nn.NLLLoss()

        best_acc = 0
        best_loss = 1e15
        iteration = 0
        if self.resume_iters:
            print("resuming step %d ..."% self.resume_iters)
            iteration = self.resume_iters
            self.load_checkpoint(self.resume_iters)
            best_loss, best_acc = self.eval()

        while iteration < self.num_iters:
            self.model.train()  
            self.optimizer.zero_grad()

            try:
                data, label = next(src_data_iter)
            except:
                src_data_iter = iter(self.src_trainset_loader)
                data, label = next(src_data_iter)

            data, label = data.to(self.device), label.to(self.device)
            src_batch_size = data.size(0)

            
            p = float(iteration) / (self.num_iters)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            src_domain = torch.zeros((src_batch_size,), dtype=torch.long, device=self.device)

            class_output, domain_output = self.model(data, alpha)

            src_c_loss = criterion(class_output, label)
            src_d_loss = criterion(domain_output, src_domain)

            loss = src_c_loss + src_d_loss

            if self.src_only:
                tgt_d_loss = torch.zeros(1)

            else:
                try:
                    tgt_data, _ = next(tgt_data_iter)
                except:
                    tgt_data_iter = iter(self.tgt_trainset_loader)
                    tgt_data, _ = next(tgt_data_iter)

                tgt_data = tgt_data.to(self.device)
                tgt_batch_size = tgt_data.size(0)
                tgt_domain = torch.ones((tgt_batch_size,), dtype=torch.long, device=self.device)

                _, domain_output = self.model(tgt_data, alpha)
                tgt_d_loss = criterion(domain_output, tgt_domain)

                loss += tgt_d_loss

            
            loss.backward()
            self.optimizer.step()

            # Output training stats
            if (iteration+1) % self.log_interval == 0:
                print('Iteration: {:5d}\tloss: {:.6f}\tloss_src_class: {:.6f}\tloss_src_domain: {:.6f}\tloss_tgt_domain: {:.6f}'.format(
                    iteration + 1, 
                    loss.item(), src_c_loss.item(), src_d_loss.item(), tgt_d_loss.item()))

                if self.use_wandb:
                    import wandb
                    wandb.log({"loss": loss.item(),
                               "loss_src_class": src_c_loss.item(),
                               "loss_src_domain": src_d_loss.item(),
                               "loss_tgt_domain": tgt_d_loss.item()}, 
                               step=iteration+1)

            # Save model checkpoints
            if (iteration+1) % self.save_interval == 0 and iteration > 0:
                val_loss, val_acc = self.eval()
                if self.use_wandb:
                    import wandb
                    wandb.log({"val_loss": val_loss,
                               "val_acc": val_acc}, step=iteration+1, commit=False)

                self.save_checkpoint(iteration)

                if (val_acc > best_acc):
                    print('val acc: %.2f > %.2f' % (val_acc, best_acc))
                    best_acc = val_acc
                if (val_loss < best_loss):
                    print('val loss: %.4f < %.4f' % (val_loss, best_loss))
                    best_loss = val_loss

            iteration += 1


    def eval(self):
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        val_loss = 0.0
        correct = 0.0
        with torch.no_grad():
            for data, label in self.src_valset_loader:
                data, label = data.to(self.device), label.to(self.device)
                output, _ = self.model(data)
                val_loss += criterion(output, label).item()
                pred = torch.exp(output).max(1, keepdim=True)[1]
                correct += pred.eq(label.view_as(pred)).sum().item()

        val_loss /= len(self.src_valset_loader)
        val_acc = 100. * correct / len(self.src_valset_loader.dataset)
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.  format(
            val_loss, correct, len(self.src_valset_loader.dataset), val_acc))

        return val_loss, val_acc
