from torch.autograd import Function
import torch.nn as nn
import torch


class SIMSE(nn.Module):
    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)

        return torch.sum(diffs.pow(2)) / n - torch.sum(diffs).pow(2) / (n ** 2)


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, shared_input, private_input):

        batch_size = shared_input.size(0)
        shared_input = shared_input.view(batch_size, -1)
        private_input = private_input.view(batch_size, -1)

        shared_input_l2_norm = torch.norm(shared_input, p=2, dim=1, keepdim=True).detach()
        shared_input_l2 = shared_input.div(shared_input_l2_norm.expand_as(shared_input) + 1e-20)

        private_input_l2_norm = torch.norm(private_input, p=2, dim=1, keepdim=True).detach()
        private_input_l2 = private_input.div(private_input_l2_norm.expand_as(private_input) + 1e-20)

        diff_loss = torch.mean((shared_input_l2.t().mm(private_input_l2)).pow(2))

        return diff_loss


def exp_lr_scheduler(optimizer, step, init_lr=1e-2, lr_decay_step=20000, step_decay_weight=0.9):

    # Decay learning rate by a factor of step_decay_weight every lr_decay_step
    current_lr = init_lr * (step_decay_weight ** (step / lr_decay_step))

    if step % lr_decay_step == 0:
        print('learning rate is set to %f' % current_lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    return optimizer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def xavier_weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_uniform_(m.weight.data)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def kaiming_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_uniform_(m.weight.data, mode='fan_out', nonlinearity='relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.normal(m.weight.data, 1.0, 0.02)
        nn.init.constant(m.bias.data, 0.0) 