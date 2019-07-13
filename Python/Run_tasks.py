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
        for i in range(1, 6):
            # print('Running script with {}'.format(i))
            sp.check_call([sys.executable, 'Run_tasks.py', '--arg1', args.arg1, '--arg2', str(i)])

    else:
        if int(args.arg2) == 1:
            from train_network import *
            train_both_tasks(nn, int(args.arg1), data_len=data_len, two_times=two_times, batch_size=batch_size, lr=lr, epoch=epoch,
                             global_task=global_task, earlystop=earlystop)

        if int(args.arg2) == 2:
            from predict_data import *
            predict_two_tasks(nn, int(args.arg1))

        if int(args.arg2) == 3:
            from predict_data import *
            predict_allFalse_two_tasks(nn, int(args.arg1), global_task=global_task)

        if int(args.arg2) == 4:
            from export_for_matlab import *
            export_nn_for_svm_two_tasks(nn, int(args.arg1), global_task=global_task)
            create_json_for_ROC(nn, s=int(args.arg1))

        if int(args.arg2) == 5:
            from export_for_matlab import *
            export_allFalse_for_svm_two_tasks(nn, int(args.arg1), global_task)

