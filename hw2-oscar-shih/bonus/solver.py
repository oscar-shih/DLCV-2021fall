import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from model import DSN
from utils import SIMSE, DiffLoss, exp_lr_scheduler, xavier_weights_init


class Solver(object):
    def __init__(self, src_trainset_loader, src_valset_loader, tgt_trainset_loader=None, config=None):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.src_trainset_loader = src_trainset_loader
        self.src_valset_loader = src_valset_loader
        self.tgt_trainset_loader = tgt_trainset_loader
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.step_decay_weight = config.step_decay_weight
        self.active_domain_loss_step = config.active_domain_loss_step
        self.resume_iters = config.resume_iters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.weight_decay = config.weight_decay
        self.alpha_weight = config.alpha_weight
        self.beta_weight = config.beta_weight
        self.gamma_weight = config.gamma_weight
        self.src_only = config.src_only
        self.exp_name = config.name
        os.makedirs(config.ckp_dir, exist_ok=True)
        self.ckp_dir = os.path.join(config.ckp_dir, self.exp_name)
        os.makedirs(self.ckp_dir, exist_ok=True)
        self.example_dir = os.path.join(self.ckp_dir, "output")
        os.makedirs(self.example_dir, exist_ok=True)
        self.log_interval = config.log_interval
        self.save_interval = config.save_interval
        self.use_wandb = config.use_wandb
        
        self.build_model()

    def build_model(self):
        self.model = DSN().to(self.device)
        self.model.apply(xavier_weights_init)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,  betas=[self.beta1, self.beta2], weight_decay=self.weight_decay)

    def save_checkpoint(self, step):
        state = {'state_dict': self.model.state_dict(),
                 'optimizer' : self.optimizer.state_dict()}
        new_checkpoint_path = os.path.join(self.ckp_dir, '{}-dsn.pth'.format(step + 1))
        torch.save(state, new_checkpoint_path)
        print('model saved to %s' % new_checkpoint_path)

    def load_checkpoint(self, resume_iters):
        print('Loading the trained models from step {}...'.format(resume_iters))
        new_checkpoint_path = os.path.join(self.ckp_dir, '{}-dsn.pth'.format(resume_iters))
        state = torch.load(new_checkpoint_path)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        print('model loaded from %s' % new_checkpoint_path)

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optimizer.zero_grad()
    
    def train(self):
        task_criterion = nn.CrossEntropyLoss()
        recon_criterion = SIMSE()
        diff_criterion = DiffLoss()
        sim_criterion = nn.CrossEntropyLoss()
        fix_src_data, _ = next(iter(self.src_valset_loader))
        fix_src_data = fix_src_data.to(self.device)
        if not self.src_only:
            fix_tgt_data, _ = next(iter(self.tgt_trainset_loader))
            fix_tgt_data = fix_tgt_data.to(self.device)

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
            loss = 0.0

            if self.src_only:
                tgt_domain_loss = torch.zeros(1)
                tgt_recon_loss = torch.zeros(1)
                tgt_diff_loss = torch.zeros(1)

            else:
                try:
                    tgt_data, _ = next(tgt_data_iter)
                except:
                    tgt_data_iter = iter(self.tgt_trainset_loader)
                    tgt_data, _ = next(tgt_data_iter)

                tgt_data = tgt_data.to(self.device)
                tgt_batch_size = len(tgt_data)

                if iteration > self.active_domain_loss_step:
                    p = float(iteration - self.active_domain_loss_step) / (self.num_iters - self.active_domain_loss_step)
                    p = 2. / (1. + np.exp(-10 * p)) - 1

                    _, tgt_domain_output, tgt_private_code, tgt_shared_code, tgt_recon = self.model(tgt_data, mode='target', p=p)
                    tgt_domain_label = torch.ones((tgt_batch_size,), dtype=torch.long, device=self.device)
                    tgt_domain_loss = sim_criterion(tgt_domain_output, tgt_domain_label)
                    loss += self.gamma_weight * tgt_domain_loss

                else:
                    _, tgt_domain_output, tgt_private_code, tgt_shared_code, tgt_recon = self.model(tgt_data, mode='target')
                    tgt_domain_loss = torch.zeros(1)

                tgt_recon_loss = recon_criterion(tgt_recon, tgt_data)
                tgt_diff_loss = diff_criterion(tgt_private_code, tgt_shared_code)

                loss += (self.alpha_weight * tgt_recon_loss + self.beta_weight * tgt_diff_loss)
            

            try:
                src_data, src_class_label = next(src_data_iter)
            except:
                src_data_iter = iter(self.src_trainset_loader)
                src_data, src_class_label = next(src_data_iter)

            src_data, src_class_label = src_data.to(self.device), src_class_label.to(self.device)
            src_batch_size = src_data.size(0)


            if iteration > self.active_domain_loss_step:
                p = float(iteration - self.active_domain_loss_step) / (self.num_iters - self.active_domain_loss_step)
                p = 2. / (1. + np.exp(-10 * p)) - 1

                src_class_output, src_domain_output, src_private_code, src_shared_code, src_recon = self.model(src_data, mode='source', p=p)
                src_domain_label = torch.zeros((src_batch_size,), dtype=torch.long, device=self.device)
                src_domain_loss = sim_criterion(src_domain_output, src_domain_label)
                loss += self.gamma_weight * src_domain_loss

            else:
                src_class_output, src_domain_output, src_private_code, src_shared_code, src_recon = self.model(src_data, mode='source')
                src_domain_loss = torch.zeros(1)

            src_class_loss = task_criterion(src_class_output, src_class_label)
            src_recon_loss = recon_criterion(src_recon, src_data)
            src_diff_loss = diff_criterion(src_private_code, src_shared_code)

            loss += (src_class_loss + self.alpha_weight * src_recon_loss + self.beta_weight * src_diff_loss)

            loss.backward()
            self.optimizer = exp_lr_scheduler(optimizer=self.optimizer, step=iteration, init_lr=self.lr, lr_decay_step=self.num_iters_decay, step_decay_weight=self.step_decay_weight)
            self.optimizer.step()

            
            # Output training stats
            if (iteration+1) % self.log_interval == 0:
                print('Iteration: {:5d} / {:d} loss: {:.6f} loss_src_class: {:.6f} loss_src_domain: {:.6f} loss_src_recon: {:.6f} loss_src_diff: {:.6f} loss_tgt_domain: {:.6f} loss_tgt_recon: {:.6f} loss_tgt_diff: {:.6f}'.format(
                    iteration + 1, self.num_iters, 
                    loss.item(), src_class_loss.item(), src_domain_loss.item(), src_recon_loss.item(), src_diff_loss.item(), 
                    tgt_domain_loss.item(), tgt_recon_loss.item(), tgt_diff_loss.item()))

                if self.use_wandb:
                    import wandb
                    wandb.log({"loss": loss.item(),
                               "loss_src_class": src_class_loss.item(),
                               "loss_src_domain": src_domain_loss.item(),
                               "loss_src_recon": src_recon_loss.item(),
                               "loss_src_diff": src_diff_loss.item(),
                               "loss_tgt_domain": tgt_domain_loss.item(),
                               "loss_tgt_recon": tgt_recon_loss.item(),
                               "loss_tgt_diff": tgt_diff_loss.item()}, 
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

                _, _, _, _, rec_all = self.model(fix_src_data, mode='source', rec_scheme='all')
                _, _, _, _, rec_share = self.model(fix_src_data, mode='source', rec_scheme='share')
                _, _, _, _, rec_private = self.model(fix_src_data, mode='source', rec_scheme='private')
                vutils.save_image(torch.cat((fix_src_data, rec_all, rec_share, rec_private)), os.path.join(self.example_dir, '%d_src.png' % (iteration+1)), nrow=16, normalize=True)
                
                if not self.src_only:
                    _, _, _, _, rec_all = self.model(fix_tgt_data, mode='target', rec_scheme='all')
                    _, _, _, _, rec_share = self.model(fix_tgt_data, mode='target', rec_scheme='share')
                    _, _, _, _, rec_private = self.model(fix_tgt_data, mode='target', rec_scheme='private')
                    vutils.save_image(torch.cat((fix_tgt_data, rec_all, rec_share, rec_private)), os.path.join(self.example_dir, '%d_tgt.png' % (iteration+1)), nrow=16, normalize=True)

            iteration += 1


    def eval(self):
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        val_loss = 0.0
        correct = 0.0
        with torch.no_grad():
            for b_idx, (data, label) in enumerate(self.src_valset_loader):
                data, label = data.to(self.device), label.to(self.device)
                output, _, _, _, _ = self.model(data, mode='source', rec_scheme='all')
                val_loss += criterion(output, label).item()
                pred = torch.exp(output).max(1, keepdim=True)[1]
                correct += pred.eq(label.view_as(pred)).sum().item()

        val_loss /= len(self.src_valset_loader)
        val_acc = 100. * correct / len(self.src_valset_loader.dataset)
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.  format(
            val_loss, correct, len(self.src_valset_loader.dataset), val_acc))

        return val_loss, val_acc
