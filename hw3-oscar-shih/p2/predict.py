import torch

from transformers import BertTokenizer
from PIL import Image
import argparse

from models import caption
from datasets import coco, utils
from configuration import Config
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform

parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--input_path', type=str, help='path to input image', required=True)
parser.add_argument('--output_path', type=str, help='path to output image', required=True)
args = parser.parse_args()

os.makedirs(args.output_path, exist_ok = True)

config = Config()

version = 'v3'
model = torch.hub.load('saahiluppal/catr', version, pretrained=True)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)


def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template

def show_mask_on_image(img, mask):
    mask = skimage.transform.pyramid_expand(mask, upscale=min(img.size[0] // mask.shape[0], img.size[1] // mask.shape[1]), sigma=4)
    mask = skimage.transform.resize(mask, (img.size[1], img.size[0]))
    mask = mask / np.amax(mask)
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * (1 - mask)), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

@torch.no_grad()
def evaluate(path, config, dir_path):
    image = Image.open(path)
    image = coco.val_transform(image)
    # print('original shape:', image.shape)
    image = image.unsqueeze(0)
    caption, cap_mask = create_caption_and_mask(start_token, config.max_position_embeddings)
    
    attn = []
    mask_size = 0
    
    model.eval()
    for i in range(config.max_position_embeddings - 1):
        predictions, attn_map, mask_size = model(image, caption, cap_mask)
        attn.append(attn_map)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)
        if predicted_id[0] == 102: break
        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False
    result = tokenizer.decode(caption[0].tolist(), skip_special_tokens=True).capitalize()
    text = ['<start>'] + result.capitalize().split() + ['<end>']
    plt.figure(figsize = (50, 50))
    img = Image.open(path).convert('RGB')
    for i in range(len(text)):
        plt.subplot(len(text) // 5 + 1, 5, i + 1)
        plt.text(0, 1, '%s' % (text[i]), color='black', backgroundcolor='white', fontsize=40)
        if i == 0:
            plt.imshow(img)
            continue
        mask = np.mean(np.array([sa.cpu().numpy()[:,i - 1,:].reshape(mask_size) for sa in attn]), axis = 0)
        # plt.imshow(a)
        plt.imshow(show_mask_on_image(img, mask))
        plt.axis('off')

    plt.tight_layout()
    print(os.path.join(dir_path, path.split('/')[-1].split('.')[0] + '.png'))
    plt.savefig(os.path.join(dir_path, path.split('/')[-1].split('.')[0] + '.png'), bbox_inches = 'tight', pad_inches = 0)

for pic in os.listdir(args.input_path):
    evaluate(os.path.join(args.input_path, pic), config, args.output_path)
