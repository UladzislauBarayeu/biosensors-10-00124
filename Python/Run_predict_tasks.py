import subprocess as sp
import sys
import argparse
from configurations import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg1')
    parser.add_argument('--arg2')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if not args.arg2:
        for i in range(1, 3):
            # print('Running script with {}'.format(i))
            sp.check_call([sys.executable, 'Run_predict_tasks.py', '--arg1', args.arg1, '--arg2', str(i)])

    else:


        if int(args.arg2) == 1:
            from predict_data import *
            predict_allFalse_two_tasks(nn, int(args.arg1), global_task=global_task)

        if int(args.arg2) == 2:
            from export_for_matlab import *

            export_allFalse_for_svm_two_tasks(nn, int(args.arg1), global_task)

