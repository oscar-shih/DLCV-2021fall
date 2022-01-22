# TODO: create shell script for running your data hallucination model

# Example
wget https://www.dropbox.com/s/bb9tw92i118m7zt/model_1550.pth?dl=0 -O model_p2.pth
python3 ./p2/inference.py --img_csv $1 --img_dir $2 --output_path $3 --ckp_path model_p2.pth
