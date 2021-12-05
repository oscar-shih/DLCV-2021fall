import os
import argparse
import glob

import torch
import torchvision.transforms as transforms
from PIL import Image

from model import Model

def main(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    state = torch.load(config.ckpt_path)
    model = Model().to(device)
    model.load_state_dict(state['state_dict'])
    

    filenames = glob.glob(os.path.join(config.img_dir, '*.png'))
    filenames = sorted(filenames)
    out_filename = config.save_dir
    
    model.eval()
    with open(out_filename, 'w') as out_file:
        out_file.write('image_id,label\n')
        with torch.no_grad():
            for fn in filenames:
                data = Image.open(fn)
                data = transform(data)
                data = torch.unsqueeze(data, 0)
                data = data.to(device)
                output = model(data)
                pred = output.max(1, keepdim=True)[1]
                out_file.write(fn.split('/')[-1] + ',' + str(pred.item()) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Training configuration.
    parser.add_argument('--img_dir', type=str, default='hw1_data/p1_data/val_50')
    parser.add_argument('--save_dir', type=str, default='ckpt')
    parser.add_argument('--ckpt_path', default='/content/drive/MyDrive/DLCV/HW1/p1_model.ckpt', type=str, help='Checkpoint path.')
    
    config = parser.parse_args()
    main(config)
