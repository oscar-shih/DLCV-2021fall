filename='p2_model.ckpt'

if [ ! -f $filename ]; then
        wget https://www.dropbox.com/s/tc5ta9y9rtlm2aa/p2_model.ckpt?dl=0 -O $filename
fi

python3 ./p2/test.py --img_dir $1 --save_dir $2 --ckp_path $filename
