import h5py
import numpy as np
from progress.bar import Bar
from os import listdir
from os.path import isfile, join
import csv


class Data:
    def __init__(self):
        self.training_set_dir = './train/'
        self.training_set = [f.replace('.hdf5', '') for f in listdir(
            self.training_set_dir) if isfile(join(self.training_set_dir, f)) and f.endswith(".hdf5")]

        self.labels = {}
        with open('./train/_labels.csv') as fp:
            reader = csv.reader(fp, delimiter=",", quotechar='"')
            next(reader, None)
            self.labels = {key: int(val) for key, val in reader}

        self.testing_set_dir = './test/'
        self.testing_set = [f.replace('.hdf5', '') for f in listdir(
            self.testing_set_dir) if isfile(join(self.testing_set_dir, f)) and f.endswith(".hdf5")]

    def generate_sfts_data(self, for_dataset='TRAIN'):
        data = []
        classification_labels = []

        if for_dataset == 'TEST':
            fileset = self.testing_set
            read_dir = self.testing_set_dir
        else:
            fileset = self.training_set
            read_dir = self.training_set_dir

        bar = Bar(max=len(fileset))
        for filename in fileset:
            if read_dir == self.training_set_dir:
                classification_labels.append(self.labels[filename])

            np.set_printoptions(threshold=np.inf)
            f1 = h5py.File(read_dir + filename + '.hdf5', 'r+')
            a_group_key1 = list(f1.keys())[0]

            address_h1_sfts = a_group_key1 + '/H1/SFTs/'
            address_l1_sfts = a_group_key1 + '/L1/SFTs/'
            values_h1_sfts = np.array(f1[address_h1_sfts])
            values_l1_sfts = np.array(f1[address_l1_sfts])

            sfts = np.concatenate((values_h1_sfts, values_l1_sfts), axis=None)
            sfts_real = np.zeros(len(sfts))
            for x, sft in enumerate(sfts):
                sfts_real[x] = sft.real

            data.append(sfts_real)
            bar.next()
        bar.finish()

        if read_dir == self.training_set_dir:
            return data, classification_labels

        return data
