
import subprocess as sp
import sys
import argparse
from configurations import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg1')
    parser.add_argument('--arg2')
    parser.add_argument('--arg3')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if not args.arg1:
        for i in list_of_subjects:
            print('Running script with {}'.format(i))
            sp.check_call([sys.executable, 'Run_subjects.py', '--arg1', str(i), '--arg2', args.arg2, '--arg3', args.arg3])

    else:
        sp.check_call([sys.executable, 'Run_tasks.py', '--arg1', args.arg1, '--arg2', args.arg2, '--arg3', args.arg3])

