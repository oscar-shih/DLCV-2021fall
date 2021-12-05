# TODO: create shell script for running your improved UDA model

# Example
wget https://www.dropbox.com/s/a9k69dpil8iepkx/bonus_model.zip?dl=0 -O bonus_model.zip
unzip bonus_model.zip


if [ "$2" = "mnistm" ]
then
    python3 ./bonus/test.py --img_dir $1 --save_path $3 --ckp_path ./bonus_model/svhn-mnistm.pth
elif [ "$2" = "usps" ]
then
    python3 ./bonus/test.py --img_dir $1 --save_path $3 --ckp_path ./bonus_model/mnistm-usps.pth
else
    python3 ./bonus/test.py --img_dir $1 --save_path $3 --ckp_path ./bonus_model/usps-svhn.pth
fi
