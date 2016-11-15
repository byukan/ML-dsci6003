import numpy as np
from gradient_descent import GradientDescent
import test_import

__author__ = "You"


class LogisticRegression(object):
    def __init__(self, fit_intercept=True, scale=True, norm="L2"):
        """
        INPUT: GradientDescent, function, function, function
        OUTPUT: None

        Initialize class variables. Takes three functions:
        cost: the cost function to be minimized
        gradient: function to calculate the gradient of the cost function
        predict: function to calculate the predicted values (0 or 1) for
        the given data
        """

        gradient_choices = {None: self.cost_gradient, "L1": self.cost_gradient_lasso, "L2": self.cost_gradient_ridge}

        # * You'll need to initialize alpha, gama, and the weights (coefficients for the regression)

        # * You'll also need to store the number of iterations

        # * You'll also need to store a boolean value for whether or
        # * not you fit the intercept and scale

        self.alpha = None
        self.gamma = None
        self.coeffs = None
        self.num_iterations = 0
        self.fit_intercept = fit_intercept
        self.scale = scale
        self.normalize = False

        # I give these lines to you
        if norm:
            self.norm = norm
            self.normalize = True
        self.gradient = gradient_choices[norm]

    def fit(self, X, y, alpha=0.01, num_iterations=10000, gamma=0.):
        """
        INPUT: 2 dimensional numpy array, numpy array, float, int, float
        OUTPUT: numpy array

        Main routine to train the model coefficients to the data
        the given coefficients.
        """
        # * You'll need to store the dimensions of the input here
        (m, n) = (float(x) for x in X.shape)

        # * You'll also need to store the inputs for
        # * alpha (the lagrange multiplier) and gamma
        (self.alpha, self.gamma) = (alpha, gamma)

        # * you'll need to update the stored value of num_iterations
        self.num_iterations += num_iterations

        # * randomly initialize the regression weights
        self.coeffs = np.random.randn(n)

        # * Create an instance of GradientDescent
        gradient = GradientDescent(self.fit_intercept, self.normalize, self.gradient)

        # * Run gradient descent
        gradient.run(X, y, self.coeffs, self.alpha, num_iterations)

        # * store the coefficients obtained from the gradient descent
        self.coeffs = gradient.coeffs

    def predict(self, X):
        """
        INPUT: 2 dimensional numpy array, numpy array
        OUTPUT: numpy array

        Calculate the predicted values (0 or 1) for the given data with
        the given coefficients.
        """

        # * The hypothesis function wil predict probabilities (floats between 0 and 1) for each input.

        # * you will need to be able to return a set of values between 0 and 1 for each of these.

        # * return a bool (t/f) value for each percentage, such that percentages above 0.5 are
        # * returned as 1, else 0.
        return np.around(self.hypothesis(X, self.coeffs)).astype(bool)

    def hypothesis(self, X, coeffs):
        """
        INPUT: 2 dimensional numpy array, numpy array
        OUTPUT: numpy array

        Calculate the predicted percentages (floats between 0 and 1)
        for the given data with the given coefficients.
        """

        # * The hypothesis function is going to return a proposed probability for each of the test data points
        # * this will be done using the logistic function and the coefficients you've derived from the gradient descent
        return 1 / (1 + np.exp(-X.dot(coeffs)))

    def cost_function(self, X, y, coeffs):
        """
        INPUT: 2 dimensional numpy array, numpy array, numpy array
        OUTPUT: float

        * without regularization *
        Calculate the value of the cost function for the data with the
        given coefficients.
        """

        # * call the hypothesis function to return a set of probabilities into a single vector h

        # * return the log-likelihood for each of these predictions  1/M sum y_i*h_i + (1-y_i)*(1-h_i)
        # * using the dot product will help
        h = self.hypothesis(X, coeffs)
        (m, n) = (float(x) for x in X.shape)  # m is number of rows, n is number of columns (the features)
        return (1 / m) * (y.dot(h) + (1 - y).dot(1 - h))

    def cost_regularized(self, X, y, coeffs):
        """
        INPUT: 2 dimensional numpy array, numpy array, numpy array
        OUTPUT: float

        * with regularization *
        Calculate the value of the cost function with regularization
        for the data with the given coefficients.
        """
        h = self.hypothesis(X, coeffs)
        (m, n) = (float(x) for x in X.shape)
        return (1. / m) * (y.dot(h) + (1 - y).dot(1 - h) + self.gamma * coeffs.dot(coeffs.T) / 2)

    def cost_gradient(self, X, y, coeffs):
        """
        INPUT: 2 dimensional numpy array, numpy array, numpy array
        OUTPUT: numpy array

        * without regularization *

        Calculate the gradient of the cost function at the given value
        for the coeffs.

        Return an array of the same size as the coeffs array.
        """

        # This function is not used in the above code, just kept here for measuring the current state of cost

        # * Calculate the hypothesis function with the input coefficients

        # * Return Sum x_i*(y_i - h_i)

        h = self.hypothesis(X, coeffs)
        return X.T.dot(y - h)

    def gradient_regularized(self, X, y, coeffs):
        """
        INPUT: 2 dimensional numpy array, numpy array, numpy array
        OUTPUT: numpy array

        * with regularization *

        Calculate the gradient of the cost function with regularization
        at the given value for the coeffs.

        Return an array of the same size as the coeffs array.
        """
        (m, n) = (float(x) for x in X.shape)
        h = self.hypothesis(X, coeffs)
        weights = coeffs[1:]
        weights = np.insert(weights, 0, 0)
        return X.T.dot(y - h) + self.gamma * weights / n

    def cost_gradient_lasso(self, X, y, coeffs):
        """
        INPUT: 2 dimensisonal numpy array, numpy array, numpy array
        OUTPUT: numpy array

        Calculate the gradient of the cost function with regularization
        at the given value for the coeffs.

        Return an array of the same size as the coeffs array.
        """
        (m, n) = (float(x) for x in X.shape)
        h = self.hypothesis(X, coeffs)
        weights = [c / np.abs(c) for c in coeffs[1:]]
        weights = np.insert(weights, 0, 0)
        return X.T.dot(y - h) + self.gamma * weights / n

    def cost_gradient_ridge(self, X, y, coeffs):
        """
        INPUT: 2 dimensional numpy array, numpy array, numpy array
        OUTPUT: numpy array

        Calculate the gradient of the cost function with regularization
        at the given value for the coeffs.

        Return an array of the same size as the coeffs array.
        """
        (m, n) = (float(x) for x in X.shape)
        h = self.hypothesis(X, coeffs)
        weights = coeffs[1:]
        weights = np.insert(weights, 0, 0)
        return X.T.dot(y - h) + self.gamma * weights / n
