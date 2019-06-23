
import subprocess as sp
import sys
import argparse
from configurations import *
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg1')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if not args.arg1:
        for i in list_of_subjects:
            print('Running script with {}'.format(i))
            sp.check_call([sys.executable, 'Run_subjects.py', '--arg1', str(i)])

    else:
        sp.check_call([sys.executable, 'Run_tasks.py', '--arg1', args.arg1])

