import numpy as np
import math

class ClassificationTree:
    """
    This class implements one of CART algorithms: Classification Tree. At each node, the tree uses gini-impurity to choose
    the best character to classify the dataset.
    After constructing, it calculates error rate for pruning at each node, and thus pruning dynamically.

    Args:
        x: 2-dimension numpy array, each row contains characters of one sample.
        y: 1-dimension numpy array, contains all labels of the samples.
        samples_in_leaf: If samples less than samples_in_leaf, the node will not be classified any longer.
        gini_threshold: If gini impurity is less than this threshold, the node will not be classified any longer.
        training_set_ratio: The fraction of training set in all input data. Others used for validation in pruning.

    Attributes:
        x: 2-dimension numpy array, contains training data for constructing the tree, each row contains characters of one sample.
        y: 1-dimension numpy array, contains labels of the training data.
        x_test: 2-dimension numpy array, contains validation data used in pruning.
        y_test: 1_dimension numpy array, contains labels of validation data.
        samples_in_leaf: If samples less than samples_in_leaf, the node will not be classified any longer.
        gini_threshold: If gini impurity is less than this threshold, the node will not be classified any longer.
        character_num: How many characters one sample have.
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
            x: 2-dimension numpy array, each row contains characters of a sample
            y: 1-dimension numpy array, labels of data

        Attributes:
            p: parent node, default None
            x: 2-dimension numpy array, contains all data in this node.
            y: 1-dimension numpy array, contains all labels of the data
            left: left child, contains data meeting the classification conditions. None in leaves.
            right: right child, contains data dis-meeting the classification conditions. None in leaves.
            classifier: a tuple, classifier[0] is the index of the character used to classify,
                             classifier[1] is the value of this character to classify.
            label: class label, which means the label which will be predicted, if the sample stops at this node.
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

    def __init__(self, x, y, samples_in_leaf = 0, gini_threshold = 0, training_set_ratio = 2/3):
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
        self.gini_threshold = gini_threshold
        self.character_num = x.shape[1]
        self.root = self.Node(None, x, y)
        self.__node_list = [self.root]
        self.__internal_node_list = []
        while self.__node_list != []:
            self.__classify(self.__node_list.pop(0))
        self.__pruning()


    def predict(self, x):
        """
        Predict the labels of input samples.

        Args:
            x: 2-dimension numpy array, each row contains characters of a sample.
               If x contains just 1 sample, one dimension is also fine.

        Raises:
            'ValueError: Dimension not match'. The characters of input sample is different from the training data.

        Returns:
            1-dimension numpy array of class labels for each samples
        """
        if x.ndim == 1:
            x = np.reshape(x,[1,x.shape[0]])
        if x.shape[1] != self.character_num:
            raise ValueError("Dimension not match.")
        res = []
        for i in x:
            current_node = self.root
            while current_node.left != None:
                current_node = current_node.left if i[current_node.classifier[0]] == current_node.classifier[1] else current_node.right
            res.append(current_node.label)
        return np.array(res)

    def __pruning(self):
        """
        Prune the tree at each node dynamically.
        Calculate the error rate before and after at each node using validation dataset.
        If the error rate before pruning is larger, prune.
        """

        for i in range(len(self.__internal_node_list)):
            left = self.__internal_node_list[i].left
            right = self.__internal_node_list[i].right
            error_rate_before = self.__error_rate(self.predict(self.x_test), self.y_test)
            self.__internal_node_list[i].left = None
            self.__internal_node_list[i].right = None
            error_rate_after = self.__error_rate(self.predict(self.x_test), self.y_test)
            if error_rate_after <= error_rate_before:
                return None
            self.__internal_node_list[i].left = left
            self.__internal_node_list[i].right = right

    def __error_rate(self, y1, y2):
        """
        calculate error rate between 2 arrays of labels.

        Args:
              y1: 1-dimension numpy array, contains labels.
              y2: 1-dimension numpy array, contains labels.
        """
        return y1[y1!=y2].shape[0]/y1.shape[0]

    def __classify(self, current_node):
        """
        Classify at the node input.
        If gini impurity of the current node is smaller than gini_threshold, or if the node contains less nodes than
        samples_in_leaf, or there is no character left to classify, the node will not be classified again, and will
        become leaf node.
        If not, gini impurity after classification by each value of each character will be calculated, and then choose
        the less one as the best classifier, classify the node into left and right child.

        Args:
            current_node: The node to be classified. 
        """

        self.__internal_node_list.append(current_node)
        classes = list(set(current_node.y))
        classifiers = []
        for i in range(self.character_num):
            values = []
            for j in current_node.x:
                values.append(j[i])
            classifiers.append(list(set(values)))
        indictor = 0
        for i in classifiers:
            if len(i) > 1:
                indictor = 1
                break

        max_class = 0
        max_class_count = 0
        for i in classes:
            if max_class_count < np.sum(current_node.y[current_node.y == i]):
                max_class = i
                max_class_count = np.sum(current_node.y[current_node.y == i])
        current_node.label = max_class
        if self.__gini(current_node.y) <= self.gini_threshold \
                or current_node.y.shape[0] <= self.samples_in_leaf \
                or indictor == 0:
            return None

        best_classifier = 0
        best_value = 0
        min_gini = math.inf
        for i in range(self.character_num):
            for j in classifiers[i]:
                new_gini = self.__gini_after_classification(current_node.x, current_node.y, i, j)
                if min_gini > new_gini:
                    min_gini = new_gini
                    best_value = j
                    best_classifier = i
        x1 = current_node.x[current_node.x[:,best_classifier] == best_value]
        y1 = current_node.y[current_node.x[:,best_classifier] == best_value]
        x2 = current_node.x[current_node.x[:,best_classifier] != best_value]
        y2 = current_node.y[current_node.x[:,best_classifier] != best_value]
        left = self.Node(current_node, x1, y1)
        right = self.Node(current_node, x2, y2)
        current_node.left = left
        current_node.right = right
        current_node.classifier = (best_classifier, best_value)
        self.__node_list.append(left)
        self.__node_list.append(right)



    def __gini(self, y):
        """
        Calculate Gini Impurity.

        Args:
            y: 1-dimension numpy array, contains labels of all samples.

        Returns:
            Gini Impurity (float).

        """
        classes = list(set(y))
        gini = 1
        samples_num = len(y)
        for i in classes:
            count = 0
            for j in y:
                count += 1 if i == j else 0
            gini -= (count/samples_num)**2
        return gini

    def __gini_after_classification(self, x, y, classifier, value):
        """
        Calculate Gini Impurity after classifying the data by a certain value of a character.
        Classify the original data into 2 dataset, of which the first contains those whose this character equals to the value,
        while the second contains others.


        Args:
            x: 2-dimension numpy array, each row represents a sample, containing value of all characters of this sample.
            y: 1-dimension numpy array, contains labels of all samples.
            classifier(int): index of the character used to classify.
            value(int): the value of this character used to classify.

        Returns:
            Gini Impurity (float).

        """
        y1 = y[x[:,classifier]==value]
        y2 = y[x[:,classifier]!=value]
        gini_after_classification = self.__gini(y1) * (len(y1)/len(y)) \
                                    + self.__gini(y2) * (len(y2)/len(y))
        return gini_after_classification




#x = np.array([[0,0,0,1],[0,0,0,0],[0,1,0,1],[0,1,1,0],[0,0,0,0],[1,0,0,0],[1,0,0,2],[1,1,1,2],[1,0,1,3],[1,0,1,1],[2,0,1,2],[2,0,1,0],[2,1,0,0],[2,1,0,1],[2,0,0,2]])
#y = np.array([0,0,1,1,0,0,0,1,1,1,1,1,1,1,0])
#s = ClassificationTree(x,y)
#print(s.predict(x))

import sklearn.datasets
import random


index = random.sample(list(range(150)),150)
iris = sklearn.datasets.load_iris()
iris
train_data = iris.data[index[:100],:]
train_y = iris.target[index[:100]]


test_data = iris.data[index[100:],:]
test_y = iris.target[index[100:]]

s = ClassificationTree(train_data, train_y)

print(s.predict(train_data))
print(train_y)

print(s.predict(test_data))
print(test_y)





