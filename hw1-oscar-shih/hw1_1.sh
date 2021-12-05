filename='p1_model.ckpt'
if [ ! -f p1_model.ckpt ]; then
        wget https://www.dropbox.com/s/6obgpyea85uooej/p1_model.ckpt?dl=0 -O $filename
fi
python3 ./p1/test.py --img_dir $1 --save_dir $2 --ckpt_path $filename

