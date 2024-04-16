import numpy as np


train1 = np.loadtxt("train1.txt")
train2 = np.loadtxt("train2.txt")
test1 = np.loadtxt("test1.txt")
test2 = np.loadtxt("test2.txt")

def split_data(data):
    n_data = data.shape[0]
    X = data[:, 0].reshape(n_data, 1)
    y = data[:, 1].reshape(n_data, 1)
    return X, y


class Data_Prep():
    def __init__(self,Normalize = False):
        self.train1 = np.loadtxt("train1.txt")
        self.train2 = np.loadtxt("train2.txt")
        self.test1 = np.loadtxt("test1.txt")
        self.test2 = np.loadtxt("test2.txt")
        self.Normalize = Normalize

    def spdata(self):
        X_train1, y_train1 = split_data(self.train1)
        X_test1, y_test1 = split_data(self.test1)
        X_train2, y_train2 = split_data(self.train2)
        X_test2, y_test2 = split_data(self.test2)

        if self.Normalize:
            return map(lambda x: self.norm_data(x),
                [X_train1, y_train1, X_test1, y_test1, X_train2, y_train2, X_test2, y_test2])
        else:
            return X_train1 , y_train1, X_test1, y_test1, X_train2, y_train2,X_test2, y_test2

    def norm_data(self,X):
        return (X - np.mean(X)) / np.std(X)