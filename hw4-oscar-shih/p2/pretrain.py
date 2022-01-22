import torch
from torchvision import models
from byol_pytorch import BYOL
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import myDataset

epoch = 150
batch_size = 64
lr = 3e-4

train_tfm = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
trainset_dir = '../hw4_data/mini/train'
trainset = myDataset(root=trainset_dir, transform=train_tfm)
trainset_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)

resnet = models.resnet50(pretrained=False).to('cuda')
resnet.load_state_dict(torch.load('./improved-net.pt'))
learner = BYOL(
    resnet,
    image_size = 128,
    hidden_layer = 'avgpool'
)

opt = torch.optim.Adam(learner.parameters(), lr=lr)

for ep in range(epoch):
    for batch_idx, (images, target) in enumerate(trainset_loader):
        images = images.to('cuda')
        loss = learner(images)
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() # update moving average of target encoder
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Lr: {}'.format(
                ep, batch_idx * len(images), len(trainset_loader.dataset),
                100. * batch_idx / len(trainset_loader), loss.item(), opt.param_groups[0]['lr']))
# save your improved network
torch.save(resnet.state_dict(), './improved-net.pt', _use_new_zipfile_serialization=False)
