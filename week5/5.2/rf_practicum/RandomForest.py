import numpy as np
from collections import Counter

from DecisionTree import DecisionTree


class RandomForest(object):
    '''A Random Forest class'''

    def __init__(self, num_trees, num_features):
        '''
           num_trees:  number of trees to create in the forest:
        num_features:  the number of features to consider when choosing the
                           best split for each node of the decision trees
        '''
        self.num_trees = num_trees
        self.num_features = num_features
        self.forest = None

    def fit(self, X, y):
        '''
        X:  two dimensional numpy array representing feature matrix
                for test data
        y:  numpy array representing labels for test data
        '''
        self.forest = self.build_forest(X, y, self.num_trees, X.shape[0], \
                                        self.num_features)

    def build_forest(self, X, y, num_trees, num_samples, num_features):
        """
             Repeat num_trees times:

       Create a random sample of the data with replacement
       Build a decision tree with that sample

    Return the list of the decision trees created
        :param X:
        :param y:
        :param y:
        :param num_trees:
        :param num_samples:
        :param num_features:
        :return:
        """

        # * Return a list of num_trees DecisionTrees.



        # Modify build_forest in your RandomForest class to pass the num_features parameter to the Decision Trees.
        self.num_features = num_features

        trees_list = []
        num_samples = X.shape[0]/2

        for i in range(num_trees):
            # dt = DecisionTree(num_features = num_features)
            dt = DecisionTree(num_features = num_features)

            idx_1 = np.random.choice(range(len(X)), size=num_samples, replace=True)

            # X_sample = np.random.choice(X, size=num_samples, replace=True)

            X_sample = X[idx_1,:]

            y_sample = np.random.choice(y, size=num_samples, replace=True)
            dt.fit(X_sample,y_sample)
            trees_list.append(dt)

        return trees_list



    def predict(self, X):
        """
        Return a numpy array of the labels predicted for the given test data.


        In the predict method, you should have each Decision Tree classify each data point. Choose the label with the majority of trees. Break ties by choosing one of the labels arbitrarily.
        """

        # * Each one of the trees is allowed to predict on the same row of input data. The majority vote
        # is the output of the whole forest. This becomes a single prediction.


        results = np.array([tree.predict(X) for tree in self.forest]).T

        return np.asarray([Counter(row).most_common(1)[0][0] for row in results])


        # prediction_list = []
        # for tree in self.forest:
        #     prediction_list.append(tree.predict(X))
        #
        # return np.apply_along_axis(np.array(prediction_list).sum(axis=0) / len(prediction_list), axis=0, arr=np.array(prediction_list))

    def score(self, X, y):

        """
        Return the accuracy of the Random Forest for the given test data.
        In the score method, you should first classify the data points and count the percent of them which match the given labels.
        """

        # * In this case you simply compute the accuracy formula as we have defined in class. Compare predicted y to
        # the actual input y.

        # error = (y - self.predict(X))**2
        #
        # return 1-(error**.5/len(y))

        return sum(self.predict(X) == y) / len(y)