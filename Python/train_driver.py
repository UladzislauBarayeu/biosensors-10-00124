#predict_all_false(2, 4)
# from export_for_matlab import  *
# export_nn_for_svm_two_tasks(21,15)

import subprocess as sp
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if not args.arg:
        for i in [2]:
            print('Running script with {}'.format(i))
            sp.check_call([sys.executable, 'train_driver.py', '--arg', str(i)])
    else:
        from train_network import *
        train_both_tasks(2, int(args.arg), two_times=False, batch_size=36, lr=0.001, epoch=2)
