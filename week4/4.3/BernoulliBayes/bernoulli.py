# Put import statements here
<<<<<<< HEAD
<<<<<<< HEAD
from collections import Counter
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s- %(message)s')
=======
>>>>>>> zipfian/master
=======
>>>>>>> zipfian/master


class BernoulliBayes(object):

    def __init__(self):
        self.prior = {}
        self.pvc = {}

    def fit(self,X,y):
<<<<<<< HEAD
<<<<<<< HEAD
        """
        Input:
            - X: (2d numpy array) Contains input data
            - y: (numpy array) Contains labels
        Output:
            - None, fit model to input data
        """

        counts_y = Counter(y)
        vocabulary = np.ravel(X)

        # for each class c in C
        for c in counts_y.keys():
            # count all documents in D belonging to that class, Nc
            index = np.where(X == c)
            D = X[index]
            # update the prior[c] with Nc
            self.prior[c] = len(D)
            # for each word v in V

            counts_d = Counter(D)
            self.pvc[c] = {}

            for word in vocabulary:
                # count all docs in D containing v belonging to that class, Ncv
                # if word in D:
                    # Ncv = counts_d[word]

                # logging.debug(D)

                Ncv = sum([1 for x in D if word in x])

                # print(Ncv)

                # add in the count to the conditional probability table P(v, c) = (N_{cv} + 1)/(Nc + 2)
                self.pvc[word] = (Ncv + 1) / (self.prior[c] + 2)
=======
=======
>>>>>>> zipfian/master
        '''
        Input: 
            - X: (2d numpy array) Contains input data
            - y: (numpy array) Contains labels
        Ouput: 
            - None, fit model to input data
        '''
        # for each class c in C
            # count all documents in D belonging to that class, Nc
            # update the prior[c] with Nc
            # for each word v in V
                # count all docs in D containing v belonging to that class, Ncv
                # add in the count to the conditional probability table P(v, c) = (N_{cv} + 1)/(Nc + 2)
<<<<<<< HEAD
>>>>>>> zipfian/master
=======
>>>>>>> zipfian/master
        # store P(v,c) and Priors



    def predict(self,X):
<<<<<<< HEAD
<<<<<<< HEAD
        pass
=======
>>>>>>> zipfian/master
=======
>>>>>>> zipfian/master
        # For each point in X
            # for each class c in C
                    # initialize score[c] = log(prior[c])
                    # for all v in V:
                        # if v is 1:
                            # score[c] += log(P(v, c))
                        # else:
                            # score[c] += log(1 - P(v, c)
            # predict argmax(score[c])
        # Return predictions
         