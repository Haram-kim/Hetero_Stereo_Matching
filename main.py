import sys
import yaml
from HSM import HSM
from dataloader import DataLoaderDSEC

try:
    dir_path = sys.argv[1]
    data_seq = sys.argv[2]
except:
    print("ERROR: Please enter data set path $directory path$ $data sequence name$")
    print("Example) python main.py /c/DSEC/ interlaken_00_c")
    exit()

loader = DataLoaderDSEC(dir_path, data_seq)
params = loader.load_params()

if __name__ == '__main__':
    
    hsm = HSM(params)
    for idx in range(0, loader.image_num-1):
        image, image_ts, event = loader.get_data(idx)
        hsm.process(image, image_ts, event)
        hsm.evaluate(idx)
        print('processing: {0} / {1}'.format(idx+1, loader.image_num))

    hsm.metric_print('init')
    hsm.metric_print('proposed')
    hsm.time_print()
