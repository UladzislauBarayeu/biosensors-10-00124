import json
import os
import random
from sklearn.utils import shuffle
from configurations import *

class EEGdata:

    def __init__(self):
        self.true_data = []
        self.false_data = []
        self.file_path = ''
        self.dir = ''

    def load_labels(self, h5file, global_task='Task1'):
        self.file_path = repo_with_raw_data + global_task + '/' + h5file
        json_data = open(self.file_path)
        d = json.load(json_data)
        json_data.close()
        labels = np.array(d['result_label'][:])
        return labels

    def load_raw_data(self, subject, global_task='Task1', task='T1', load_false_data_from_files=True, other=None,
                      data_len=0):
        file=str(subject) + '.json'
        self.dir = repo_with_raw_data + global_task + '/'
        self.file_path = self.dir + file
        json_data = open(self.file_path)
        d = json.load(json_data)
        json_data.close()

        true_data = np.array(d[task][:number_of_trials])
        false_data = []
        self.task = task

        if load_false_data_from_files:
            all_subjects = shuffle(os.listdir(self.dir))
            all_subjects.remove(file)
            t_size = data_len - len(true_data)
            self.id_for_false = [random.choice(all_subjects) for i in range(t_size)]
            self.internal_id = []
            for el in self.id_for_false:
                json_data = open(self.dir + el)
                d = json.load(json_data)
                json_data.close()
                temp = d[task]
                i = random.randint(0, number_of_trials-1)
                self.internal_id.append(i)
                false_data.append(temp[i])

            false_data = np.array(false_data)
            false_data = np.transpose(false_data, (0, 2, 1))


        else:
            self.id_for_false = other.id_for_false
            self.internal_id = other.internal_id
            j = 0
            for el in self.id_for_false:
                json_data = open(self.dir + el)
                d = json.load(json_data)
                json_data.close()
                temp = d[task]
                false_data.append(temp[self.internal_id[j]])
                j += 1

            false_data = np.array(false_data)
            false_data = np.transpose(false_data, (0, 2, 1))

        true_data = np.transpose(true_data, (0, 2, 1))
        self.all_data = np.concatenate((true_data, false_data), axis=0)
        self.all_data = np.expand_dims(self.all_data, axis=3)
        self.true_labels, self.false_labels = np.array([[1., 0.] for i in range(len(true_data))]), np.array(
            [[0., 1.] for i in range(len(false_data))])
        self.all_labels = np.concatenate((self.true_labels, self.false_labels), axis=0)
