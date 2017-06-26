import pandas as pd
import numpy as np


class Dataset(object):
    def __init__(self,  train_path, test_path, oneHot = False):
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)
        self.y_ = self.train['final_status']
        self.index_in_epoch = 0
        self.num_examples = self.train.shape[0]
        if oneHot == True:
            ohm =  np.zeros((self.y_.shape[0], 2))
            ohm[np.arange(self.y_.shape[0]), self.y_] = 1
            self.y_ = ohm
        return self.train, self.y_, self.test


    def next_batch(self, batch_size):
        start = self.index_in_epoch
        self.start += batch_size
        if self.index_in_epoch > self.num_examples:
            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.train = self.train[perm]
            self.y_ = self.y_[perm]

            start = 0
            self.index_in_epoch = batch_size
        end = self.index_in_epoch
        return self.train[start:end], self.y_[start:end]
