import numpy as np


class NN2L:
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.where(x < 0, 0, x)

    def cost(self, Y, Yh):
        loss_sum = np.sum(-(Y * np.log(Yh) + (1 - Y) * np.log(1 - Yh)))
        return 1 / 209 * loss_sum

    def __init__(self, input_features_size, l_rate=0.01):
        self.W1 = np.random.normal(0, 1, (10, input_features_size))
        self.b1 = np.zeros((10, 1))
        self.W2 = np.random.normal(0, 1, (1, 10))
        self.b2 = np.zeros((1, 1))
        self.sigmoid_v = np.vectorize(self.sigmoid)
        self.dW1 = np.zeros(self.W1.shape)
        self.dW2 = np.zeros(self.W2.shape)
        self.db1 = np.zeros((10, 1))
        self.db2 = np.zeros((1, 1))
        self.l_rate = l_rate

    def forward_pass(self, X):
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        self.A2 = self.sigmoid_v(self.Z2)
        return self.A2

    def backprop(self):
        dz2 = (self.A2 - self.Y.T)
        self.dW2 = 1 / 209 * dz2.dot(self.A1.T)
        self.db2 = 1 / 209 * np.sum(dz2, axis=1, keepdims=True)
        dz1 = self.W2.T.dot(dz2) * np.where(self.Z1 < 0, 0, 1)
        self.dW1 = 1 / 209 * dz1.dot(self.X.T)
        self.db1 = 1 / 209 * np.sum(dz1, axis=1, keepdims=True)

    def update_params(self):
        self.W2 -= (self.dW2 * self.l_rate)
        self.b2 -= (self.db2 * self.l_rate)
        self.W1 -= (self.dW1 * self.l_rate)
        self.b1 -= (self.db1 * self.l_rate)

    def train(self, X, Y, num_iter=50000):
        self.X = X
        self.Y = Y.reshape(209, 1)

        for iter in range(num_iter):
            self.forward_pass(X)
            error = self.cost(self.Y, self.A2.T)
            predicted = np.where(self.A2 >= 0.5, 1, 0).T
            if iter % 100 == 0:
                acc = np.sum(self.Y == predicted)/209*100
                print('Error: {}, Accuracy = {}'.format(error, acc))
            self.backprop()
            self.update_params()
