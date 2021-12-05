import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import FCN32s, DeepLabv3_ResNet50, FCN8s


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
        self.model_type = config.model_type

        os.makedirs(config.ckp_dir, exist_ok=True)
        self.ckp_dir = os.path.join(config.ckp_dir, self.exp_name)
        os.makedirs(self.ckp_dir, exist_ok=True)

        self.build_model()

    def build_model(self):
        if self.model_type == 'DeepLabv3_Resnet50':
            self.model = DeepLabv3_ResNet50().to(self.device)
        elif self.model_type == 'FCN8s':
            self.model = FCN8s().to(self.device)
        else:
            self.model = FCN32s().to(self.device)

        print(self.model)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if self.iteration != 0:
            checkpoint_path = os.path.join(self.ckp_dir, 'model.ckpt')
            self.load_checkpoint(checkpoint_path, self.iteration)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', verbose=True, min_lr=1e-5)

    def save_checkpoint(self, checkpoint_path, step):
        state = {'state_dict': self.model.state_dict(),
                 'optimizer' : self.optimizer.state_dict()}
        new_checkpoint_path = '{}-{}'.format(checkpoint_path, step+1)
        torch.save(state, new_checkpoint_path, _use_new_zipfile_serialization = False)
        print('model saved to %s' % new_checkpoint_path)

    def load_checkpoint(self, checkpoint_path, step):
        new_checkpoint_path = '{}-{}'.format(checkpoint_path, step)
        state = torch.load(new_checkpoint_path)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        print('model loaded from %s' % new_checkpoint_path)
    
    def train(self):
        checkpoint_path = os.path.join(self.ckp_dir, 'model.ckpt')
        criterion = nn.CrossEntropyLoss()

        best_mean_iou = 0
        if self.iteration > 0:
            _, best_mean_iou = self.eval()

        for ep in range(self.epoch):
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.trainset_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                self.optimizer.step()

                if self.iteration % self.log_interval == 0 and self.iteration > 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tIteration: {}\tLoss: {:.6f}'.format(
                        ep, batch_idx * len(data), len(self.trainset_loader.dataset),
                        100. * batch_idx / len(self.trainset_loader), self.iteration, loss.item()))

                if self.iteration % self.save_interval == 0 and self.iteration > 0:
                    val_loss, mean_iou = self.eval()
                    self.scheduler.step(mean_iou)
                    if mean_iou > best_mean_iou:
                        best_mean_iou = mean_iou
                        self.save_checkpoint(checkpoint_path, self.iteration)

                self.iteration += 1

        # save the final model
        val_loss, mean_iou = self.eval()
        self.scheduler.step(mean_iou)
        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            self.save_checkpoint(checkpoint_path, self.iteration)


    def eval(self):
        criterion = nn.CrossEntropyLoss()
        self.model.eval()

        val_loss = 0
        mean_iou = 0

        tp_fp = torch.zeros(6, device=self.device)
        tp_fn = torch.zeros(6, device=self.device)
        tp = torch.zeros(6, device=self.device)

        with torch.no_grad():
            for data, target in self.valset_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += criterion(output, target).item()
                pred = output.max(1, keepdim=True)[1]
                pred = pred.view_as(target)
                for i in range(6):
                    tp_fp[i] += torch.sum(pred == i)
                    tp_fn[i] += torch.sum(target == i)
                    tp[i] += torch.sum((pred == i) * (target == i))

        val_loss /= len(self.valset_loader.dataset)
        print('Val set: Average loss: {:.4f}'.  format(val_loss))

        for i in range(6):
            iou = tp[i] / (tp_fp[i] + tp_fn[i] - tp[i])
            mean_iou += iou / 6
            print('class #%d : %1.5f' % (i, iou))

        print('mean_iou: %f' % mean_iou)

        return val_loss, mean_iou