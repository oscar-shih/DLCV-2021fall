import glob
import os
import argparse
import random
import numpy as np 
import pandas as pd
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from model import Net
from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

def randomseed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
        
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

randomseed(1126)

def main(config):
    transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    data_df = pd.read_csv(config.img_csv).set_index("id")
    lab = preprocessing.LabelEncoder()
    lab.classes = np.array(['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle',
                'Bucket', 'Calculator', 'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 
                'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet'
                'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer', 'Helmet', 'Kettle',
                'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse',
                'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer',
                'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers',
                'Soda', 'Speaker', 'Spoon', 'TV', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'Webcam'], dtype=object)
    labels = lab.fit_transform(data_df['label'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    state = torch.load(config.ckp_path)
    model = Net().to(device)
    model.load_state_dict(state['state_dict'])
    
    # filenames = glob.glob(os.path.join(config.img_dir, '*.jpg'))
    out_filename = config.output_path

    correct = 0
    total = 0
    model.eval()
    with open(out_filename, 'w') as out_file:
        out_file.write('id,filename,label\n')
        for i in range(len(data_df)):
            path = data_df.loc[i, "filename"]
            image = transform(os.path.join(config.img_dir, path))
            image = image.unsqueeze(0).to(device)
            label = labels[i]
            output = model(image).detach().cpu()
            pred = output.argmax(axis=1).item()
            correct += (label == pred)
            pred = lab.inverse_transform([pred])[0]
            out_file.write(str(i) + ',' + path + ',' + pred + '\n')
            total += 1
    print("acc on val {:.2f}".format(100. * correct / total))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_csv', type=str, default='../hw4_data/office/val.csv')
    parser.add_argument('--img_dir', type=str, default='../hw4_data/office/val')
    parser.add_argument('--output_path', type=str, default='ckpt')
    parser.add_argument('--ckp_path', default='ckpt/model3.ckpt', type=str, help='Checkpoint path.')
    
    config = parser.parse_args()
    main(config)
 