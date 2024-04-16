import numpy as np
import matplotlib.pyplot as plt


def plot_function(x, y, x_label, y_label, text):
    plt.plot(x, y, label="MSE")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    txt = text
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()

class NN_LinearRegression(object):

    def __init__(self, learning_rate=0.01, beta=0.9, momentum=False,n_epochs = 2500):
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.beta = beta
        self.weight = (((np.random.rand(1, 1)) * 2) - 1)
        self.bias = (((np.random.rand(1, 1)) * 2) - 1)
        self.momentum_w = 0
        self.momentum_b = 0
        self.n_epochs = n_epochs
        self.Loss_List = []

    def forward(self, X):
        return np.dot(X, self.weight) + self.bias

    def back_prop(self, X, y, predicted_values):

        if self.momentum:
            error = (2 * (predicted_values - y)) / X.shape[0]
            delta_weight = (np.dot(X.T, error) * self.learning_rate) + (
                    self.momentum_w * self.beta)
            self.momentum_w = delta_weight
            delta_bias = np.sum(error * self.learning_rate) + (self.momentum_b * self.beta)
            self.momentum_b = delta_bias

            self.weight -= delta_weight
            self.bias -= delta_bias

        else:
            error = (2 * (predicted_values - y)) / X.shape[0]
            delta_weight = np.dot(X.T, error) * self.learning_rate
            delta_bias = np.sum(error * self.learning_rate)

            self.weight -= delta_weight
            self.bias -= delta_bias

    def train(self, X, y):
        for _ in range(self.n_epochs):
            output = self.forward(X)
            self.back_prop(X, y, output)
            Loss = self.loss(X, y)
            print("Loss Train at epoch  "+str(_)+"  :  " + str(Loss))
            self.Loss_List.append(Loss)

    def predict(self, X):
        return self.forward(X)

    def loss(self, X, y):
        return np.mean(np.square(y - self.predict(X)))

    def plot_loss(self):
        plot_function(list(range(1, len(self.Loss_List) + 1)), self.Loss_List, 'Epochs', 'MSE',
                      "Epoch VS MSE - Batch Mode")
    def plot_scatter(self,X,y,text):
        prediction = self.weight[0][0] * X + self.bias[0][0]
        plt.scatter(X, y)
        plt.plot(X, prediction, color='red', label="Predicted",)
        plt.figtext(0.5, 0.01, text, wrap=True, horizontalalignment='center', fontsize=12)
        plt.show()
