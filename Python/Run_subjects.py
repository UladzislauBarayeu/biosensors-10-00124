# ================================================
# Author: Nastassya Horlava
# Github: @HorlavaNastassya
# Email: g.nasta.work@gmail.com
# ===============================================

import subprocess as sp
import sys
import argparse
from configurations import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg1')
    parser.add_argument('--arg2')
    parser.add_argument('--arg3')
    parser.add_argument('--arg5')
    parser.add_argument('--arg6')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if not args.arg1:
        list_of_subjects_gen = [i for i in range(int(args.arg5), int(args.arg6)+1, 1)]
        for i in list_of_subjects_gen:
            print('Running script with {}'.format(i))
            sp.check_call([sys.executable, 'Run_subjects.py', '--arg1', str(i), '--arg2', args.arg2, '--arg3', args.arg3])

    else:
        sp.check_call([sys.executable, 'Run_tasks.py', '--arg1', args.arg1, '--arg2', args.arg2, '--arg3', args.arg3])

