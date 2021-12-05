import os
import glob
import shutil
ema_filename = glob.glob('generate_result/default/*-ema.jpg')
mr_filename = glob.glob('generate_result/default/*-mr.jpg')
tgt_dir = 'image_mr'
os.mkdir(tgt_dir)
for file in mr_filename:
    try:
        shutil.move(file, tgt_dir)
    except OSError as e:
        print(f"Error:{ e.strerror}")
tgt_dir = 'image_ema'
os.mkdir(tgt_dir)
for file in ema_filename:
    try:
        shutil.move(file, tgt_dir)
    except OSError as e:
        print(f"Error:{ e.strerror}")
        
filename = glob.glob('image_mr/*.jpg')
print('mr_filename length: ', len(filename))
filename = glob.glob('image_ema/*.jpg')
print('ema_filename length: ', len(filename))
