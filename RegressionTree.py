import numpy as np
import math


class RegressionTree:
    """
    This class implements one of CART algorithms: Regression Tree. At each node, the tree uses MSE to choose
    the best character to cut the dataset into left and right child nodes.
    After constructing, it calculates MSE at each node using validation dataset, and thus pruning dynamically.

    Args:
        x: 2-dimension numpy array, each row represents one sample.
        y: 1-dimension numpy array, contains all labels of the samples.
        samples_in_leaf: If samples less than samples_in_leaf, the node will not be cut any longer.
        loss_threshold: If MSE is less than this threshold, the node will not be cut any longer.
        training_set_ratio: The fraction of training set in all input data. Others used for validation in pruning.

    Attributes:
        x: 2-dimension numpy array, each row represents one sample.
        y: 1-dimension numpy array, contains all labels of the samples.
        x_test: 2-dimension numpy array, contains validation data used in pruning.
        y_test: 1_dimension numpy array, contains labels of validation data.
        samples_in_leaf: If samples less than samples_in_leaf, the node will not be cut any longer.
        loss_threshold: If MSE is less than this threshold, the node will not be cut any longer.
        dimension_num: How many dimensions one sample have.
        root: Pointer refers to the root node.

    Raises:
        'ValueError: Dimension not match': Means either amounts of data in x and y do not match or input data for
                                           prediction have different amount of characters from the training data.
    """
    class Node:
        """
        An internal class implements nodes in the Tree.

        Args:
            p: parent node, default None
            x: 2-dimension numpy array, each row contains a sample
            y: 1-dimension numpy array, labels of data

        Attributes:
            p: parent node, default None
            x: 2-dimension numpy array, contains all data in this node.
            y: 1-dimension numpy array, contains all labels of the data
            left: left child, contains data smaller than cut-off. None in leaves.
            right: right child, contains data larger than cut-off. None in leaves.
            classifier: a tuple, classifier[0] is the index of the dimension used to cut,
                             classifier[1] is the value of this dimension used to cut.
            label: value of this node, which means the regressed value which will be predicted if the sample stops at this node.
        """
        def __init__(self, p = None, x = None, y = None):
            """
            Constructor method
            """

            self.p = p
            self.x = x
            self.y = y
            self.left = None
            self.right = None
            self.classifier = None  # tuple
            self.label = None

    def __init__(self, x, y, samples_in_leaf = 1, loss_threshold = 0, training_set_ratio = 2/3):
        """
        Constructor method
        """

        if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
            raise ValueError("Dimension not match.")
        train_test_cut = int(y.shape[0] * training_set_ratio)
        self.x = x[:train_test_cut]
        self.y = y[:train_test_cut]
        self.x_test = x[train_test_cut:]
        self.y_test = y[train_test_cut:]
        self.samples_in_leaf = samples_in_leaf
        self.loss_threshold = loss_threshold
        self.dimension_num = x.shape[1]
        self.root = self.Node(None, self.x, self.y)
        self.__node_list = [self.root]
        self.__internal_node_list = []
        while self.__node_list != []:
            self.__cut(self.__node_list.pop(0))
        self.__pruning()

    def predict(self,x ):
        """
        Predict the labels of input samples.

        Args:
            x: 2-dimension numpy array, each row represents a sample.
               If x contains just 1 sample, one dimension is also fine.

        Raises:
            'ValueError: Dimension not match'. The dimension of input sample is different from the training data.

        Returns:
            1-dimension numpy array of labels for each samples
        """
        if x.ndim == 1:
            x = np.reshape(x,[1,x.shape[0]])
        if x.shape[1] != self.dimension_num:
            raise ValueError("Dimension not match.")
        res = []
        for i in x:
            current_node = self.root
            while current_node.left != None:
                current_node = current_node.left if i[current_node.classifier[0]] <= current_node.classifier[1] \
                    else current_node.right
            res.append(current_node.label)
        return np.array(res)

    def __pruning(self):
        """
        Prune the tree at each node dynamically.
        Calculate the MSE before and after at each node using validation dataset.
        If MSE before pruning is smaller, prune.
        """

        for i in range(len(self.__internal_node_list)):
            left = self.__internal_node_list[i].left
            right = self.__internal_node_list[i].right
            mse_before = self.__loss(self.predict(self.x_test), self.y_test)
            self.__internal_node_list[i].left = None
            self.__internal_node_list[i].right = None
            mse_after = self.__loss(self.predict(self.x_test), self.y_test)
            if mse_after <= mse_before:
                return None
            self.__internal_node_list[i].left = left
            self.__internal_node_list[i].right = right

    def __cut(self, current_node):
        """
        Cut at the node input.
        If MSE of the current node is smaller than loss_threshold, or if the node contains less data than
        samples_in_leaf, or there is no character left to cut(all value in each dimension are constants), the node will
        not be cut again, and will become leaf node.
        If not, MSE after classification by each value of each dimension will be calculated, and then choose
        the less one as the best classifier, cut the node into left and right child.

        Args:
            current_node: The node to be cut.
        """

        self.__internal_node_list.append(current_node)
        classifiers = []
        for i in range(self.dimension_num):
            values = []
            for j in current_node.x:
                values.append(j[i])
            classifiers.append(list(set(values)))
        indictor = 0
        for i in classifiers:
            if len(i) > 1:
                indictor = 1
                break

        current_node.label = np.average(current_node.y)
        if self.__loss(np.average(current_node.y), current_node.y) <= self.loss_threshold \
                or current_node.y.shape[0] <= self.samples_in_leaf \
                or indictor == 0:
            return None

        best_classifier = 0
        best_value = 0
        min_loss = math.inf
        for i in range(self.dimension_num):
            for j in classifiers[i]:
                new_loss = self.__loss_after_cut(current_node.x, current_node.y, i, j)
                if min_loss > new_loss:
                    min_loss = new_loss
                    best_value = j
                    best_classifier = i
        x1 = current_node.x[current_node.x[:,best_classifier] <= best_value]
        y1 = current_node.y[current_node.x[:,best_classifier] <= best_value]
        x2 = current_node.x[current_node.x[:,best_classifier] > best_value]
        y2 = current_node.y[current_node.x[:,best_classifier] > best_value]
        left = self.Node(current_node, x1, y1)
        right = self.Node(current_node, x2, y2)
        current_node.left = left
        current_node.right = right
        current_node.classifier = (best_classifier, best_value)
        self.__node_list.append(left)
        self.__node_list.append(right)

    def __loss(self, predicted_labels, labels):
        """
        Calculate MSE loss of between predicted labels and real labels.

        Args:
            predicted_labels: 1-dimension numpy array, contains the predicted labels.
            labels: 1-dimension numpy array, contains the return labels.
        Return:
            MSE as loss

        """
        return np.sum((predicted_labels - labels)**2)

    def __loss_after_cut(self, x, y, classifier, value):
        """
        Calculate loss after cutting data into left and right child by the given value of given classifier.

        Args:
            x: 2-dimension numpy array, each row represents a sample.
            y: 1-dimension numpy array, contains labels of samples.
            classifier(int): index of dimension used to cut the data.
            value(float): value of this dimension used to cut the data.

        Return:
            loss after cut the data by the value of this given dimension.

        """

        y1 = y[x[:, classifier] <= value]
        y2 = y[x[:, classifier] > value]
        if y2.shape[0] == 0:
            return self.__loss(np.average(y1), y1)
        return self.__loss(np.average(y1), y1) + self.__loss(np.average(y2), y2)  # broadcast







import sklearn.datasets
import random


index = random.sample(list(range(506)),506)

iris = sklearn.datasets.load_boston()
train_data = iris.data[index[:406],:]
train_y = iris.target[index[:406]]

test_data = iris.data[index[406:],:]
test_y = iris.target[index[406:]]

s = RegressionTree(train_data, train_y)




def loss( predicted_labels, labels):
    """
    Calculate MSE loss of between predicted labels and real labels.

    Args:
        predicted_labels: 1-dimension numpy array, contains the predicted labels.
        labels: 1-dimension numpy array, contains the return labels.
    Return:
        MSE as loss

    """
    return np.sum((predicted_labels - labels) ** 2)

print(loss(s.predict(train_data),train_y)/train_y.shape[0])

print(s.predict(test_data))
print(test_y)
print(loss(s.predict(test_data),test_y)/test_y.shape[0])





