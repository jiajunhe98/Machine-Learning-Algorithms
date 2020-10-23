import numpy as np
import math

class DecisionTree:
    """
    This class implements Decision Tree Algorithm. The Tree can be chosen to construct by ID3 or C4.5 algorithms,
    and use pre-pruning when each node is classified.

    Args:
        x: 2-dimension numpy array, each row contains characters of one sample
        y: 1-dimension numpy array, contains all labels of the samples.
        algorithm: ID3 or C4.5
        threshold: Threshold for information gain/ information gain ratio, if is smaller than threshold,
                    no child nodes will be add to this node.
        alpha: penalty factor. larger alpha, more penalty for a complex tree.

    Attributes:
        x: 2-dimension numpy array, each row contains characters of one sample
        y: 1-dimension numpy array, contains all labels of the samples.
        algorithm: ID3 or C4.5
        threshold: Threshold for information gain/ information gain ratio, if is smaller than threshold,
                    no child nodes will be add to this node.
        alpha: penalty factor. larger alpha, more penalty for a complex tree.
        characters_num: Numbers of characters for one sample.
        node_num: Numbers of Nodes.
        root: a pointer to the root node if the decision tree.

    Methods:
        predict: Predict the labels of input samples.

    Raises:
        'ValueError: Dimensions of input data do not match.': The amount of training data and labels are different.
        'ValueError: Algorithm is not recognised.': Only ID3 and C4.5 can be chosen.
        'ValueError: Dimension not match.': The characters of input sample is different from the training data.
    """

    class Node:
        """
        An internal class implements nodes in a decision tree

        Args:
            p: parent node, default None
            data: 2-dimension numpy array, each row contains characters of a sample
            y: 1-dimension numpy array, labels of data

        Attributes:
            p: parent node, default None
            child: dictionary of child nodes, key is the value of character used to classify, value is Node
            data: 2-dimension numpy array, each row contains characters of a sample
            y: 1-dimension numpy array, labels of data
            used_characters: list of indices of characters used before this node
            classify_character: the index of the character used in this node to classify, None in leaves
            class_label: class label, not None only in leaves
        """

        def __init__(self, p = None, data = None, y = None):
            """
            Constructor method
            """
            self.p = p         # parent node
            self.child = {}    # dictionary of child nodes, key is the value of character used to classify, value is node
            self.data = data   # array of data, not empty only in leaves
            self.y = y         # array of labels of data, not empty only in leaves
            self.used_characters = []      # list of indices of characters used
            self.classify_character = None # the index of the character used in this node to classify
            self.class_label = None        # class label, not None only in leaves


    def __init__(self, x, y, algorithm = "ID3", threshold = 0, alpha = 0):
        """
        Constructor method
        """
        if x.shape[0] != y.shape[0]: raise ValueError("Dimensions of input data do not match.")
        if algorithm not in ["C4.5","ID3"]: raise ValueError("Algorithm is not recognised.")
        self.__information_gain_or_ratio = self.__information_gain if algorithm == "ID3" else self.__information_gain_ratio
        self.x = x
        self.y = y
        self.alpha = alpha
        self.threshold = threshold
        self.characters_num = x.shape[1]
        self.node_num = 0
        self.root = self.Node(None, x, y)
        self.__node_list = [self.root]   # a list to store all nodes which have not been handled (input into construct_child_node)
        while self.__node_list != []:
            self.__construct_child_node(self.__node_list.pop(0))
            self.node_num += 1

    def __construct_child_node(self, current_node):
        """
        Construct child nodes for current node, or label the current node by class label.

        Args:
            current_node: current node
        Returns:
            None
        """
        best_character_index = 0
        max_gain = 0
        classes = list(set(current_node.y))
        if len(classes) == 1:
            current_node.class_label = current_node.y[0]
            return None
        for i in range(self.characters_num):
            gain = self.__information_gain_or_ratio(current_node.data, current_node.y, i)
            if  gain > max_gain:
                max_gain = gain
                best_character_index = i
        current_node.classify_character = best_character_index
        if max_gain <= self.threshold or len(current_node.used_characters) == \
                self.characters_num or \
                self.__penalty_for_new_nodes(current_node.data,current_node.y,best_character_index) <= 0:
            sample_numbers_in_each_class = [0] * len(classes)
            for i in range(len(classes)):
                for j in current_node.y:
                    if classes[i] == j:
                        sample_numbers_in_each_class[i] += 1
            label = 0
            max_label = 0
            for i in range(len(sample_numbers_in_each_class)):
                if sample_numbers_in_each_class[i] > max_label:
                    max_label = sample_numbers_in_each_class[i]
                    label = i
            current_node.class_label = classes[label]
            return None
        charachers = list(set(current_node.data[:, best_character_index]))
        for i in charachers:
            child = self.Node(current_node,current_node.data[current_node.data[:, best_character_index] == i], \
                              current_node.y[current_node.data[:, best_character_index] == i])
            child.used_characters = current_node.used_characters + [best_character_index]
            current_node.child[i] = child
            self.__node_list.append(child)
        return None

    def __information_gain(self, x, y, character_index):
        """
        Calculate information gain of input data.

        Args:
            x: 2-dimension numpy array, each row represents a sample
            y: 1-dimension numpy array, represents the class label of all samples
            character_index: which character is used to classify

        Returns:
            an float represents the information gain if classifying by this given character
        """
        classes = list(set(y))
        characters = list(set(x[:,character_index]))
        entropy = 0
        for i in classes:
            entropy -= (y[y==i].shape[0]/y.shape[0]) * math.log2(y[y==i].shape[0]/y.shape[0])
        conditional_entropy = 0
        for i in characters:
            entropy2 = 0
            y_i = y[x[:,character_index]==i]
            classes_i = list(set(y_i))
            for j in classes_i:
                entropy2 += (y_i[y_i==j].shape[0]/y_i.shape[0]) * math.log2(y_i[y_i==j].shape[0]/y_i.shape[0])
            conditional_entropy -= entropy2 * y_i.shape[0] / y.shape[0]
        return -conditional_entropy + entropy

    def __information_gain_ratio(self, x, y, character_index):
        """
        Calculate information gain ratio of input data.

        Args:
            x: 2-dimension numpy array, each row represents a sample
            y: 1-dimension numpy array, represents the class label of all samples
            character_index: which character is used to classify

        Returns:
            an float represents the information gain ratio if classifying by this given character
        """
        entropy = 0
        characters = list(set(x[:,character_index]))
        for i in characters:
            y_i = y[x[:, character_index] == i]
            entropy -= (y_i.shape[0]/y.shape[0]) * math.log2(y_i.shape[0]/y.shape[0])
        return self.__information_gain(x,y,character_index) / entropy

    def predict(self, x):
        """
        Predict the labels of input samples.

        Args:
            x: 2-dimension numpy array, each row contains characters of a sample. If just 1 sample, x is 1 dimensional.

        Raises:
            'ValueError': Dimension not match. The characters of input sample is different from the training data.

        Returns:
            1-dimension numpy array of class labels for each samples
        """
        if x.ndim == 1:
            x = np.reshape(x,[1,x.shape[0]])
        if x.shape[1] != self.characters_num:
            raise ValueError("Dimension not match.")
        res = []
        for i in x:
            current_node = self.root
            while current_node.class_label == None:
                current_node = current_node.child[i[current_node.classify_character]]
            res.append(current_node.class_label)
        return np.array(res)

    def __penalty_for_new_nodes(self, x, y, character_index):
        """
        Calculate the penalty if classify by the given character.
        By calculating this penalty, pruning will be automatically done when constructing the tree.

        Args:
            x: 2-dimension numpy array, each row represents a sample
            y: 1-dimension numpy array, represents the class label of all samples
            character_index: which character is used to classify

        Returns:
            an float represents the information gain if classifying by this given character
        """
        classes = list(set(y))
        characters = list(set(x[:,character_index]))
        loss = self.alpha
        for i in classes:
            loss -= (y[y==i].shape[0]) * math.log2(y[y==i].shape[0]/y.shape[0])
        new_loss = 0
        for i in characters:
            loss2 = 0
            y_i = y[x[:,character_index]==i]
            classes_i = list(set(y_i))
            for j in classes_i:
                loss2 += (y_i[y_i==j].shape[0]) * math.log2(y_i[y_i==j].shape[0]/y_i.shape[0])
            new_loss -= loss2 * y_i.shape[0] / y.shape[0] - self.alpha
        return -new_loss + loss



x = np.array([[0,0,0,1],[0,0,0,0],[0,1,0,1],[0,1,1,0],[0,0,0,0],[1,0,0,0],[1,0,0,2],[1,1,1,2],[1,0,1,3],[1,0,1,1],[2,0,1,2],[2,0,1,0],[2,1,0,0],[2,1,0,1],[2,0,0,2]])
y = np.array([0,0,1,1,0,0,0,1,1,1,1,1,1,1,0])
s = DecisionTree(x,y)
print(s.predict(x))


