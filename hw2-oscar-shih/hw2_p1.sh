pip3 install stylegan2_pytorch
if [ $# -lt 1 ]; then
    echo usage: $0 target_directory
    exit 1
fi
target_directory=$1
wget https://www.dropbox.com/s/5mkarmydj6ybax6/models.zip?dl=0  -O models.zip
unzip models.zip
python3 ./p1/p1_generate.py --save_dir $1 --load_from 50
