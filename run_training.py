import sys
from lib.training.execute import get_configs_from_args, execute
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

if __name__ == '__main__':
    config = get_configs_from_args(sys.argv)
    execute('train', config)
