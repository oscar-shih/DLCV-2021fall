# TODO: create shell script for running your ViT testing code

# Example
# python3 hw3_1.py $1 $2
gdown --id 1t8KzgcwcbHiusHqkkbY2SbS3KWgl0Npw -O model3.zip
unzip model3.zip
python3 ./p1/inference.py --ckpt_path model3.ckpt --img_dir $1 --save_dir $2