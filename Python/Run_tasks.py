import subprocess as sp
import sys
import argparse
from configurations import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arg1')
    parser.add_argument('--arg2')
    parser.add_argument('--arg3')
    parser.add_argument('--arg4')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if not args.arg4:
        for i in range(1, 6):
            # print('Running script with {}'.format(i))
            sp.check_call([sys.executable, 'Run_tasks.py', '--arg1', args.arg1, '--arg2', args.arg2, '--arg3', args.arg3, '--arg4', str(i), ])

    else:
        if int(args.arg4) == 1:
            from train_network import *
            train_both_tasks(args.arg2, int(args.arg1), data_len=data_len, two_times=two_times, batch_size=batch_size, lr=lr, epoch=epoch,
                             global_task=global_task, earlystop=earlystop, channels=args.arg3)

        if int(args.arg4) == 2:
            from predict_data import *
            predict_two_tasks(args.arg2, int(args.arg1), channels=args.arg3)

        if int(args.arg4) == 3:
            from predict_data import *
            predict_allFalse_two_tasks(args.arg2, int(args.arg1), global_task=global_task, channels=args.arg3)

        if int(args.arg4) == 4:
            from export_for_matlab import *
            export_nn_for_svm_two_tasks(args.arg2, int(args.arg1),channels=args.arg3, global_task=global_task)
            create_json_for_ROC(args.arg2, s=int(args.arg1), channels=args.arg3)

        if int(args.arg4) == 5:
            from export_for_matlab import *
            export_allFalse_for_svm_two_tasks(args.arg2, int(args.arg1), global_task, channels=args.arg3)

