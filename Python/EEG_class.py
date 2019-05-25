import json
import numpy as np
import h5py
import os
import random
from sklearn.utils import shuffle

class EEGdata:

    def __init__(self):
        self.true_data=[]
        self.false_data=[]
        self.path=''

    def load_labels(self, h5file, directory=None):
        self.path = directory + h5file
        json_data = open(self.path)
        d = json.load(json_data)
        json_data.close()
        labels=np.array(d['Subject_old']['result_label'][:])
        return labels

    def load_raw_data(self, h5file, directory=None, task='T1', load_false_data_from_files=True, other=None,  data_len=0):

        self.path=directory+h5file
        json_data = open(self.path)
        d = json.load(json_data)
        json_data.close()

        true_data=np.array(d[task][:110])
        false_data=[]
        self.task=task

        if load_false_data_from_files:
            all_subjects=shuffle(os.listdir(directory))
            all_subjects.remove(h5file)
            t_size=data_len-len(true_data)
            self.id_for_false=random.sample(all_subjects, t_size-104)

            self.id_for_false.extend(all_subjects)
            self.internal_id=[]
            for el in self.id_for_false:
                json_data = open(directory+el)
                d = json.load(json_data)
                json_data.close()
                temp=d[task]
                i=random.randint(0, 109)
                self.internal_id.append(i)
                false_data.append(temp[i])

            false_data=np.array(false_data)
            false_data = np.transpose(false_data, (0, 2, 1))


        else:

            t_size = data_len - len(true_data)
            self.id_for_false = other.id_for_false
            self.internal_id=other.internal_id
            j=0
            for el in self.id_for_false:
                json_data = open(directory + el)
                d = json.load(json_data)
                json_data.close()
                temp = d[task]
                false_data.append(temp[self.internal_id[j]])
                j += 1

            false_data = np.array(false_data)
            false_data = np.transpose(false_data, (0, 2, 1))
            #self.false_data=np.copy(other.false_data)

        true_data = np.transpose(true_data, (0, 2, 1))
        self.all_data = np.concatenate((true_data, false_data), axis=0)
        self.all_data = np.expand_dims(self.all_data, axis=3)
        self.true_labels, self.false_labels=np.array([[1., 0.] for i in range(len(true_data))]),np.array([[0., 1.] for i in range(len(false_data))])
        self.all_labels=np.concatenate((self.true_labels, self.false_labels), axis=0)


    def load_data(self, h5file, directory, expand=False):
        f = h5py.File(directory+h5file, 'r')
        self.true_data = f['true_data'][:]
        self.false_data = f['false_data'][:]
        self.task = f.attrs['task']
        self.path=f.attrs['path']
        self.all_data = np.concatenate((self.true_data, self.false_data), axis=0)
        self.true_labels, self.false_labels=np.array([[1., 0.] for i in range(len(self.true_data))]),np.array([[0., 1.] for i in range(len(self.false_data))])
        self.all_labels=np.concatenate((self.true_labels, self.false_labels), axis=0)
        if expand:
            self.all_data = np.expand_dims(self.all_data, axis=3)


    def save_data_for_training(self, h5file, dir, comp=5):
        with h5py.File(dir+h5file, 'w') as f:
            d = f.create_dataset("true_data", data=self.true_data, compression="gzip", compression_opts=comp)
            d = f.create_dataset("false_data", data=self.false_data, compression="gzip", compression_opts=comp)
            f.attrs["path"] = self.path
            f.attrs['task'] = self.task

    def expand(self):
        self.all_data = np.expand_dims(self.all_data, axis=3)















