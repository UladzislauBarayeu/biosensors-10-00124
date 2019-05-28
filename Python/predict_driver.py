
import subprocess as sp
import sys
import argparse
from configurations import *
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if not args.arg:
        for i in list_of_subjects:
            print('Running script with {}'.format(i))
            sp.check_call([sys.executable, 'train_driver.py', '--arg', str(i)])
    else:
        from predict_data import *
        predict_two_tasks(nn, int(args.arg))
