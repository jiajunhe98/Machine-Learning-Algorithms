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
    """
    This class implements layers in MLP.

    Args:
        input_dim(int): input dimension.
        output_dim(int): output dimension.
        activation(str): activation, {"relu", sigmoid", "tanh", "softmax", "linear"}, default = "relu".
        drop_out(float): drop out rate.
        lambd(float): L2 regularization rate.
        optimize(str): method of optimization, {"gradient_descent", "Adam"}, default = "gradient_descent".

    Methods:
        forward: Fordward prop.
        backward: Backward prop.
        update: Update params.
    """

    def __init__(self, input_dim, output_dim, activation="relu", drop_out=0.0, lambd=0.0, optimize="gradient_descent"):
        """
        Constructor method.

        """
        xavier = (6 / (input_dim + output_dim)) ** 0.5
        self.W = np.random.uniform(-xavier, xavier, size=(output_dim, input_dim))
        self.b = np.zeros((output_dim, 1))
        self.keep_prob = 1 - drop_out
        self.lambd = lambd
        self.optimize = optimize
        self.A = None
        self.Z = None
        self.dW = None
        self.db = None
        self.VdW = np.zeros(self.W.shape)
        self.Vdb = np.zeros(self.b.shape)
        self.beta1 = 0.9    # rate for Adam
        self.SdW = np.zeros(self.W.shape)
        self.Sdb = np.zeros(self.b.shape)
        self.beta2 = 0.999  # rate for Adam
        self.mask = None
        self.learning_times = 0
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
        """
        Forward prop.

        Args:
            A: input (also the output from last layer), shape = (input, n_samples)
            drop(bool): if drop out, True when training, False when predicting.

        Return:
            Activations. Shape = (output, n_samples)
        """
        self.A = A.copy()
        self.Z = np.dot(self.W, A) + self.b
        A_next = self.activation(self.Z)
        if drop:
            A_next /= self.keep_prob
            self.mask = np.random.rand(*A_next.shape) > self.keep_prob
            A_next[self.mask] = 0
        return A_next

    def backward(self, dA, m, dZ=None, drop=True):
        """
        Backward prop.

        Args:
            dA: derivative of A.
            m(int): n_samples
            dZ: derivative of Z.(A = g(Z)). If this layer is the last layer, dZ is not None, else dZ is None.
            drop(bool): if drop out, True when training, False when predicting.

        Return:
            derivative of A of last layer.

        """
        if dZ is None:
            if drop:
                dA[self.mask] = 0
                dA /= self.keep_prob
            dZ = dA * self.activation_derivation(self.Z)
        self.dW = np.dot(dZ, self.A.T) / m + self.W * self.lambd / m
        self.db = np.sum(dZ, axis=1, keepdims=True) / m
        dA_last = np.dot(self.W.T, dZ)
        return dA_last

    def update(self, learning_rate):
        """
        update the parameters by Adam or gradient descent

        Args:
            learning_rate(float).
        """
        self.learning_times += 1
        if self.optimize == "gradient_descent":
            self.W -= learning_rate * self.dW
            self.b -= learning_rate * self.db
        if self.optimize == "Adam":
            self.VdW = self.beta1 * self.VdW + (1 - self.beta1) * self.dW
            self.Vdb = self.beta1 * self.Vdb + (1 - self.beta1) * self.db
            self.SdW = self.beta2 * self.SdW + (1 - self.beta2) * (self.dW ** 2)
            self.Sdb = self.beta2 * self.Sdb + (1 - self.beta2) * (self.db ** 2)
            vdw_correct = self.VdW / (1 - self.beta1 ** self.learning_times)
            vdb_correct = self.Vdb / (1 - self.beta1 ** self.learning_times)
            sdw_correct = self.SdW / (1 - self.beta2 ** self.learning_times)
            sdb_correct = self.Sdb / (1 - self.beta2 ** self.learning_times)
            self.W -= learning_rate * vdw_correct / (1e-8 + np.sqrt(sdw_correct))
            self.b -= learning_rate * vdb_correct / (1e-8 + np.sqrt(sdb_correct))


class MLP(object):
    """
    This class implements the neuron net(MLP).

    Args:
        x_dim(int): input dim.
        y_dim(int): output dim.
        hidden_layer: list of ints, each represents the dim of each hidden layer.
        activation: list of strings, each represents the activation of each hidden layer. The activation of output
                    layers should not be included, it will be chosen depending on loss.
        drop_out: list of drop out rate of each hidden layer.
        loss: loss function, {"cross_entropy", "mse"}. And the activation of output layer is also chosen depending on
            this arg. If loss == "cross_entropy", the activation of output is softmax, if loss == "mse", the activation
            of output is linear.
        lambd(float): L2 regularization rate.
        optimize(str): Method od optimize. {"gradient_descent", "Adam"}. Default = "gradient_descent".

    Methods:
        train:
        predict:


    """

    def __init__(self, x_dim, y_dim, hidden_layer, activation, drop_out, loss="cross_entropy", lambd=0.0, optimize="gradient_descent"):
        """
        Constructor method
        """
        self.lambd = lambd
        if loss == "cross_entropy":
            self.loss = cross_entropy
        if loss == "mse":
            self.loss = mse
        self.layers = []
        hidden_layer.insert(0, x_dim)
        hidden_layer.append(y_dim)
        if loss == "cross_entropy":
            activation.append("softmax")
        if loss == "mse":
            activation.append("linear")
        drop_out.append(0.0)
        for i in range(1, len(hidden_layer)):
            layer = Layer(hidden_layer[i-1], hidden_layer[i], activation[i-1], drop_out[i-1],
                          self.lambd, optimize)
            self.layers.append(layer)

    def train(self, x, y, max_iteration=200, learning_rate=0.05, batch_size=None, learning_rate_decay=0.):
        """
        Train the model.

        Args:
             x: batch of data input. shape (n_samples, input_dimension)
             y: batch of labels. shape (n_samples, output_dimension)
             max_iteration: maximum num of interations.
             learning_rate: learning rate.
             batch_size: int or None. If None, all x will be used as one batch.
             learning_rate_decay: learning rate decaying rate. default 0.

        Return:
            a list of cost function values after each iteration.

        """
        x = x.T
        y = y.T
        batch_size = batch_size if batch_size is not None else x.shape[1]
        batch_num = int(np.ceil(x.shape[1] / batch_size))
        x_batch = [x[:, i * batch_size: (i + 1) * batch_size] for i in range(batch_num)]
        y_batch = [y[:, i * batch_size: (i + 1) * batch_size] for i in range(batch_num)]
        cost_after_each_epoch = []
        for epoch in range(max_iteration):
            current_learning_rate = learning_rate * (1 / (1 + epoch * learning_rate_decay))
            for i in range(batch_num):
                self.__mini_batch_train(x_batch[i], y_batch[i], current_learning_rate)
            output = self.__forward(A=x, drop_out=False)
            cost = self.loss(output, y, x.shape[1])
            cost_after_each_epoch.append(cost)
        return cost_after_each_epoch

    def predict(self, x):
        """
        Predict output of input x.

        Args:
             x: data to be predicted. Shape (n_samples, input_dimension)

        Return:
            Array-like. Predicted output of input x. Shape (n_samples, output_dimension).
        """
        prob = self.__forward(A=x.T, drop_out=False)
        return prob.T

    def __mini_batch_train(self, x, y, learning_rate=0.05):
        m = x.shape[1]
        output = self.__forward(A=x, drop_out=True)
        cost = self.loss(output, y, m)
        for j in self.layers:   # L2 Regularization
            cost += self.lambd * np.sum(j.W ** 2) / (2 * m)
        self.__backward(output, y, m, drop_out=True)
        self.__update(learning_rate=learning_rate)
        return cost

    def __forward(self, A, drop_out=False):     # true when training, false when predicting
        for i in self.layers:
            A = i.forward(A, drop_out)
        return A

    def __backward(self, A, y, m, drop_out=False):      # true when training, false when checking
        layer = self.layers[-1]
        dZ = A - y
        dA = layer.backward(dA=None, m=m, dZ=dZ, drop=drop_out)
        for i in self.layers[-2:: -1]:
            dA = i.backward(dA=dA, m=m, dZ=None, drop=drop_out)

    def __update(self, learning_rate):
        for i in self.layers:
            i.update(learning_rate)





from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
data = load_digits()
mask = np.random.rand(data.data.shape[0]) < 0.7
a = data.data[mask]
norm = Normalizer().fit(a)
a = norm.transform(a)

b = OneHotEncoder().fit_transform(data.target[mask].reshape(-1, 1)).toarray()
mlp = MLP(a.shape[1], b.shape[1], hidden_layer=[50, 10], activation=["relu", "relu"], drop_out=[0.1, 0.], loss="cross_entropy", lambd=0.3, optimize="Adam")


costs = []
for ii in range(100):
    costs.append(mlp.train(a, b, 1, 0.1))
for ii in range(50):
    costs.append(mlp.train(a, b, 1, 0.03))
for ii in range(100):
    costs.append(mlp.train(a, b, 1, 0.005))
costs = [i for j in costs for i in j]
plt.plot(np.arange(0, len(costs)), costs)
plt.show()

mask2 = np.random.rand(data.data.shape[0]) >= 0.7
a1 = data.data[mask2]
a1 = norm.transform(a1)
b1 = data.target[mask2]


#target_grad, grad = mlp.grad_check(a, b)
#print("checking: ")
#print(np.sum((target_grad - grad) ** 2) / grad.shape[0])

print(np.argmax(mlp.predict(a1), axis=1))
print(b1)
print(np.sum(np.argmax(mlp.predict(a1), axis=1) == b1) / b1.shape[0])
print(np.sum(np.argmax(mlp.predict(a), axis=1) == data.target[mask]) / data.target[mask].shape[0])