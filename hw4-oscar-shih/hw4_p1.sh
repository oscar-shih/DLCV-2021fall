# TODO: create shell script for running your prototypical network

# Example
wget https://www.dropbox.com/s/efg8rp75jyysvw3/best.pth?dl=0 -O best.pth
python3 test_testcase.py --test_csv $1 --test_data_dir $2 --testcase_csv $3 --output_csv $4 --load best.pth  --matching_fn l2
