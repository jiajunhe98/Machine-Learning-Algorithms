import numpy as np


def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


def sigmoid_derivative(x):
    y = sigmoid(x)
    dx = (1 - y) * y
    return dx


def relu(x):
    mask = x < 0
    y = x.copy()
    y[mask] = 0
    return y


def relu_derivative(x):
    mask = x >= 0
    dx = np.zeros(x.shape)
    dx[mask] = 1
    return dx


def tanh(x):
    y = np.tanh(x)
    return y


def tanh_derivative(x):
    y = tanh(x)
    dx = 1 - y ** 2
    return dx


def softmax(x):
    y = np.exp(x)
    sum_y = np.sum(y, axis=0,keepdims=True)
    return y / sum_y

def linear(x):
    return x


def cross_entropy(predicts, labels, m):
    return -np.sum(labels * np.log(predicts)) / m


def mse(predicts, labels, m):
    return np.sum((predicts - labels) ** 2) / (2 * m)


class Layer(object):

    def __init__(self, input_dim, output_dim, activation="relu", drop_out=0.0, lambd=0.0, learning_rate=0.01):
        self.W = np.random.randn(output_dim, input_dim) / np.sqrt(input_dim)
        self.b = np.zeros((output_dim, 1))
        self.keep_prob = 1 - drop_out
        self.lambd = lambd
        self.learning_rate = learning_rate
        self.A = None
        self.Z = None
        self.dW = None
        self.db = None
        self.mask = None
        if activation == "relu":
            self.activation = relu
            self.activation_derivation = relu_derivative
        if activation == "sigmoid":
            self.activation = sigmoid
            self.activation_derivation = sigmoid_derivative
        if activation == "tanh":
            self.activation = tanh
            self.activation_derivation = tanh_derivative
        if activation == "softmax":
            self.activation = softmax
            self.activation_derivation = None
        if activation == "linear":
            self.activation = linear
            self.activation_derivation = None

    def forward(self, A, drop=True):
        self.A = A.copy()
        self.Z = np.dot(self.W, A) + self.b
        A_next = self.activation(self.Z) / self.keep_prob
        self.mask = np.random.rand(*A_next.shape) > self.keep_prob
        if drop:
            A_next[self.mask] = 0
        return A_next

    def backward(self, dA, m, dZ=None, drop=True):
        if dZ is None:
            if drop:
                dA[self.mask] = 0
                dA /= self.keep_prob
            dZ = dA * self.activation_derivation(self.Z)
        self.dW = np.dot(dZ, self.A.T) / m + self.W * self.lambd / m
        self.db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_last = np.dot(self.W.T, dZ)
        return dA_last

    def update(self):
        self.W -= self.learning_rate * self.dW
        self.b -= self.learning_rate * self.db


class MLP(object):

    def __init__(self, x, y, hidden_layer, activation, drop_out, loss="cross_entropy", lambd=0.0, learning_rate=0.01):
        assert x.ndim == 2
        assert y.ndim == 2
        self.x = x.T
        self.y = y.T
        self.hidden_layer = hidden_layer
        self.activation = activation
        self.drop_out = drop_out
        self.lambd = lambd
        self.learning_rate = learning_rate
        self.loss = loss
        self.m = x.shape[1]
        self.layers = []
        hidden_layer.insert(0, self.x.shape[0])
        hidden_layer.append(self.y.shape[0])
        if loss == "cross_entropy":
            activation.append("softmax")
        elif loss == "mse":
            activation.append("linear")
        drop_out.append(0.0)
        for i in range(1, len(self.hidden_layer)):
            layer = Layer(self.hidden_layer[i-1], self.hidden_layer[i],
                          self.activation[i-1], self.drop_out[i-1], self.lambd, self.learning_rate)
            self.layers.append(layer)

    def train(self, max_iteration=100):
        cost_after_each_iteration = []
        for i in range(max_iteration):
            output = self.__forward(drop_out=True)
            if self.loss == "cross_entropy":
                cost = cross_entropy(output, self.y, self.m)
            if self.loss == "mse":
                cost = mse(output, self.y, self.m)
            cost_after_each_iteration.append(cost)
            self.__backward(output, drop_out=True)
            self.__update()
        return cost_after_each_iteration

    def predict(self, x):
        prob = self.__forward(x=x.T, drop_out=False)
        return np.argmax(prob, axis=0)

    def __forward(self, x=None, drop_out=True): # true when training, false when predicting
        A = self.x.copy() if x is None else x
        for i in self.layers:
            A = i.forward(A, drop_out)
        return A

    def __backward(self, A, drop_out=True): # true when training, false when checking
        layer = self.layers[-1]
        dZ = A - self.y
        dA = layer.backward(dA=None, m=self.m, dZ=dZ, drop=drop_out)
        for i in self.layers[-2:: -1]:
            dA = i.backward(dA=dA, m=self.m, dZ=None, drop=drop_out)

    def __update(self):
        for i in self.layers:
            i.update()





from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
data = load_digits()
mask = np.random.rand(data.data.shape[0]) < 0.7
x = data.data[mask]
norm = Normalizer().fit(x)
x = norm.transform(x)

y = OneHotEncoder().fit_transform(data.target[mask].reshape(-1, 1)).toarray()

mlp = MLP(x, y, hidden_layer=[50,10], activation=["relu", "relu"], drop_out=[0.1, 0.], loss="mse", lambd=0.3)
cost = mlp.train(5000)
#print(mlp.layers[0].b)
plt.plot(np.arange(0, len(cost)), cost)
plt.show()

mask2 = np.random.rand(data.data.shape[0]) >= 0.7
a = data.data[mask2]
a = norm.transform(a)
b = data.target[mask2]
print(mlp.predict(a))
print(b)
print(np.sum(mlp.predict(a) == b) / b.shape[0])
print(np.sum(mlp.predict(x) == data.target[mask]) / data.target[mask].shape[0])