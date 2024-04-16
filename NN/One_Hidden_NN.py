import numpy as np
import matplotlib.pyplot as plt


def plot_function(x, y, x_label, y_label, text):
    plt.plot(x, y, label="MSE")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    txt = text
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()

class ANN(object):

    def __init__(self, number_of_nodes_in_hidden_layer=4, learning_rate=0.1, beta=0.88, momentum=True,
                 activation_function="sigmoid",Loss_threshold = 0,n_epochs = 10000):
        self.hidden_layer_derivative = None
        self.result = None
        self.hidden_layer = None
        #self.product = None
        self.error = None
        #self.product = None
        self.number_of_hidden_layer = number_of_nodes_in_hidden_layer
        self.learning_rate = learning_rate
        self.beta = beta
        self.momentum = momentum
        self.activation_function = activation_function
        self.weight_1 = (((np.random.rand(1, self.number_of_hidden_layer)) * 2) - 1)
        self.bias_1 = (((np.random.rand(1, self.number_of_hidden_layer)) * 2) - 1)
        self.weight_2 = (((np.random.rand(self.number_of_hidden_layer, 1)) * 2) - 1)
        self.bias_2 = (((np.random.rand(1, 1)) * 2) - 1)
        self.n_epochs = n_epochs
        self.Loss_List = []
        self.Loss_threshold = Loss_threshold

        self.momentum_w_1 = 0
        self.momentum_w_2 = 0
        self.momentum_b_1 = 0
        self.momentum_b_2 = 0

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        # return self.sigmoid(x) * (1 - self.sigmoid(x))
        return x * (1 - x)

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def relu_derivative(x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def forward(self, X):
        if self.activation_function == "sigmoid":
            product = np.dot(X, self.weight_1) + self.bias_1
            self.hidden_layer = self.sigmoid(product)
            self.result = np.dot(self.hidden_layer, self.weight_2) + self.bias_2
            return self.result
        else:
            product = np.dot(X, self.weight_1) + self.bias_1
            self.hidden_layer = self.relu(product)
            self.result = np.dot(self.hidden_layer, self.weight_2) + self.bias_2
            return self.result

    def back_prop(self, X, y, predictions):
        if self.momentum:

            if self.activation_function == "sigmoid":

                self.error = (2 * (predictions - y)) / X.shape[0]

                self.hidden_layer_derivative = (self.error.dot(self.weight_2.T)) * self.sigmoid_derivative(
                    self.hidden_layer)

                delta_weight_1 = ((X.T.dot(
                    self.hidden_layer_derivative)) * self.learning_rate) + self.beta * self.momentum_w_1
                delta_weight_2 = ((self.hidden_layer.T.dot(
                    self.error)) * self.learning_rate) + self.beta * self.momentum_w_2

                delta_bias_1 = np.sum(self.hidden_layer_derivative * self.learning_rate, axis=0,
                                      keepdims=True) + self.beta * self.momentum_b_1
                delta_bias_2 = np.sum(self.error * self.learning_rate) + self.beta * self.momentum_b_2

                self.momentum_w_1 = delta_weight_1
                self.momentum_w_2 = delta_weight_2
                self.momentum_b_1 = delta_bias_1
                self.momentum_b_2 = delta_bias_2

                self.weight_1 -= delta_weight_1
                self.weight_2 -= delta_weight_2
                self.bias_1 -= delta_bias_1
                self.bias_2 -= delta_bias_2

            else:
                self.error = (2 * (predictions - y)) / X.shape[0]

                self.hidden_layer_derivative = (self.error.dot(self.weight_2.T)) * self.relu_derivative(
                    self.hidden_layer)

                delta_weight_1 = ((X.T.dot(
                    self.hidden_layer_derivative)) * self.learning_rate) + self.beta * self.momentum_w_1
                delta_weight_2 = ((self.hidden_layer.T.dot(
                    self.error)) * self.learning_rate) + self.beta * self.momentum_w_2

                delta_bias_1 = np.sum(self.hidden_layer_derivative * self.learning_rate, axis=0,
                                      keepdims=True) + self.beta * self.momentum_b_1
                delta_bias_2 = np.sum(self.error * self.learning_rate) + self.beta * self.momentum_b_2

                self.momentum_w_1 = delta_weight_1
                self.momentum_w_2 = delta_weight_2
                self.momentum_b_1 = delta_bias_1
                self.momentum_b_2 = delta_bias_2

                self.weight_1 -= delta_weight_1
                self.weight_2 -= delta_weight_2
                self.bias_1 -= delta_bias_1
                self.bias_2 -= delta_bias_2
        else:
            if self.activation_function == "sigmoid":
                self.error = (2 * (predictions - y))

                self.hidden_layer_derivative = (self.error.dot(self.weight_2.T)) * self.sigmoid_derivative(
                    self.hidden_layer)

                delta_weight_1 = ((X.T.dot(self.hidden_layer_derivative)) * self.learning_rate) / \
                                 X.shape[0]
                delta_weight_2 = ((self.hidden_layer.T.dot(self.error)) * self.learning_rate) / \
                                 X.shape[0]

                delta_bias_1 = np.sum(self.hidden_layer_derivative * self.learning_rate, axis=0, keepdims=True) / \
                               X.shape[0]
                delta_bias_2 = np.sum(self.error * self.learning_rate) / X.shape[0]

                self.weight_1 -= delta_weight_1
                self.weight_2 -= delta_weight_2
                self.bias_1 -= delta_bias_1
                self.bias_2 -= delta_bias_2
            else:
                self.error = (2 * (predictions - y))

                self.hidden_layer_derivative = (self.error.dot(self.weight_2.T)) * self.relu_derivative(
                    self.hidden_layer)

                delta_weight_1 = ((X.T.dot(self.hidden_layer_derivative)) * self.learning_rate) / \
                                 X.shape[0]
                delta_weight_2 = ((self.hidden_layer.T.dot(self.error)) * self.learning_rate) / \
                                 X.shape[0]

                delta_bias_1 = np.sum(self.hidden_layer_derivative * self.learning_rate, axis=0, keepdims=True) / \
                               X.shape[0]
                delta_bias_2 = np.sum(self.error * self.learning_rate) / X.shape[0]

                self.weight_1 -= delta_weight_1
                self.weight_2 -= delta_weight_2
                self.bias_1 -= delta_bias_1
                self.bias_2 -= delta_bias_2

    def train(self, X, y):
        for _ in range(self.n_epochs):
            output = self.forward(X)
            self.back_prop(X, y, output)
            Loss = self.loss(X, y)
            print("Loss Train at epoch  "+str(_)+"  :  " + str(Loss))
            self.Loss_List.append(Loss)
            if (Loss < self.Loss_threshold):
                break

    def predict(self, x):
        return self.forward(x)

    def loss(self, x, y):
        return np.mean(np.square(y - self.predict(x)))

    def plot_loss(self):
        plot_function(list(range(1, len(self.Loss_List) + 1)), self.Loss_List, 'Epochs', 'MSE',
                      "Epoch VS MSE - Batch Mode")

    def plot_scatter(self,X,y,text):
        prediction = self.predict(X)
        zX, zP = zip(*sorted(zip(X,prediction )))
        plt.scatter(X, y)
        #plt.scatter(X,prediction,color='red')
        plt.plot(zX, zP, color='red', label="Predicted",)
        plt.figtext(0.5, 0.01, text, wrap=True, horizontalalignment='center', fontsize=12)
        plt.show()


    def train_stochastic(self,X,y):
        for i in range(self.n_epochs):
            # shuffle data
            permutation = np.random.permutation(X.shape[0])
            X = X[permutation]
            y = y[permutation]

            # loop through each sample in the dataset
            for j in range(X.shape[0]):
                # forward pass
                output = self.forward(X[j])

                # backward pass
                self.back_prop(X[j].reshape(1, -1), y[j].reshape(1, -1), output.reshape(1, -1))

            Loss = self.loss(X, y)
            print("Loss Train at epoch  "+str(i)+"  :  " + str(Loss))
            self.Loss_List.append(Loss)
            if (Loss < self.Loss_threshold):
                break