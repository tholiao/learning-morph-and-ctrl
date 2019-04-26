from __future__ import division

import time

import GPy
import numpy as np
from DIRECT import solve
from scipy.optimize import minimize


class Optimizer(object):
    def __init__(self, obj_f, num_inputs, bounds, n_init, start_with_x=None,
                 start_with_y=None, log=False):
        """

        obj_f:          Objective function
        n_inputs:       Number of inputs
        lower_bounds:   Sequence of lower bounds for the parameters (ordered)
        upper_bounds:   Sequence of upper bounds for the parameters (ordered)
        context_space:  Sequence of terrain z-scales to sample from
        """
        assert n_init > 0, "Must randomly initialize values"

        self.model = None
        self.iterations = 0

        self.log = log

        self.obj_f = obj_f
        self.num_inputs = num_inputs

        self.bounds = bounds
        self.bounds_lower = bounds[0]
        self.bounds_upper = bounds[1]
        self.n_init = n_init

        self.X = None
        self.Y = None
        self.Y_mean = None
        self.Y_var = None

        if start_with_x is not None and start_with_y is not None:
            print("Restoring from logs")
            self.X = start_with_x
            self.Y = start_with_y
            self.Y_mean = np.zeros((self.X.shape[0], 1))
            self.Y_var = np.zeros((self.X.shape[0], 1))
            self.train_GP(self.X, self.Y)
            self.optimize_model()
            self.update_iterations(self.X.shape[0])

    def initialize_GP(self, n_init):
        """
        Initialize the GP with n_init randomly chosen points to evaluate
        :param n_init:
        :return:
        """
        print(self.bounds, n_init)
        self.X = self.random_parameters(self.bounds, n_init)
        self.Y = self.evaluate(self.X)
        self.Y_mean = np.zeros((self.X.shape[0], 1))
        self.Y_var = np.zeros((self.X.shape[0], 1))

        self.train_GP(self.X, self.Y)
        self.optimize_model()

    def optimize_model(self):
        self.model.optimize()

    def update_iterations(self, i=1):
        self.iterations += i
        print("Iteration #", str(self.iterations))

    def train_GP(self, X, Y, kernel=None):
        """
        Trains the GP model. The Matern 5/2 kernel is used by default

        :param X:      A 2-D input vector containing both parameters and context
        :param Y:      A 2-D output vector containing objective values
        kernel: See the GPy documentation for other kernel options
        """
        print("Training GP ", X, Y)
        if kernel is None:
            kernel = GPy.kern.Matern52(input_dim=self.num_inputs,
                                       ARD=True)
        self.model = GPy.models.GPRegression(X, Y, kernel)

    def restore(self):
        self.x = np.load()

    def optimize(self, n_iterations):
        if self.model is None:
            self.initialize_GP(self.n_init)

        for i in range(n_iterations):
            self.update_iterations()

            X = self.optimize_acq_f()
            self.update_X(X)
            Y = self.evaluate(np.array([X]))
            self.update_Y(Y)

            self.train_GP(self.X, self.Y)
            self.optimize_model()
            print("OUTER LOOP: ", self.X, self.Y)
            if self.log and self.iterations % 50 == 0:
                np.save("./logs/opt_{}_iter-{}_x".format(
                    time.strftime("%Y.%m.%d-%H.%M.%S"), self.iterations),
                    self.X)
                np.save("./logs/opt_{}_iter-{}_y".format(
                    time.strftime("%Y.%m.%d-%H.%M.%S"), self.iterations),
                    self.Y)
        print("FINISHED OPTIMIZATION")

    def evaluate(self, X):
        """
        Accepts an arbitrary n >= 1 number of parameters to evaluate
        :param X: should be an 'array of arrays'
        :return: a columnn with all the results
        """
        n = X.shape[0]
        assert n >= 1, "Have to evaluate at least one row"
        print("Evaluating: ", X)

        Y = np.zeros((n, 1))
        for i, row in enumerate(X):
            row = row.reshape((1, self.num_inputs))
            Y[i] = self.obj_f(row, cache_walker=False).reshape((1, 1))
        Y = np.array(np.abs(Y))
        print("SW - Evaluated to ", Y)
        return Y

    def optimize_acq_f(self):
        def obj_sw_DIRECT(x, user_data):
            return -self.acq_f(x), 0

        def obj_sw_LBFGS(x_sw):
            return -self.acq_f(x_sw)

        x, _, _ = solve(obj_sw_DIRECT, self.bounds_lower,
                        self.bounds_upper, maxf=500)
        x = minimize(obj_sw_LBFGS, x, method='L-BFGS-B',
                     bounds=self.reformat_bounds(self.bounds)).x
        return np.array(x).reshape((1, self.num_inputs))

    def acq_f(self, x, alpha=-1, v=.01, delta=.1):
        """
        Implementation of GP-UCB
        :param x:
        :param alpha: hyperparameter
        :param v: hyperparameter
        :param delta: hyperparameter
        :return:
        """
        x = np.reshape(x, (1, self.num_inputs))
        mean, var = self.model.predict(x)
        if alpha is -1:
            alpha = np.sqrt(v * (2 * np.log((self.iterations
                                             ** ((self.num_inputs / 2) + 2))
                                            * (np.pi ** 2) / (3 * delta))))
        return mean + (alpha * var)

    def predict_optimal(self, context):
        """
        Given a context, predict the optimizer
        :param context:
        :return: the optimizer
        """

        def obj_DIRECT(x, _):
            return -self.acq_f(np.concatenate((x, context)), alpha=0), 0

        def obj_LBFGS(x):
            return -self.acq_f(np.concatenate((x, context)), alpha=0)

        context = np.array([context])
        x, _, _ = solve(obj_DIRECT, self.bounds_lower, self.bounds_upper)
        res = minimize(obj_LBFGS, x, method='L-BFGS-B', bounds=self.bounds)
        return res.x

    def update_X(self, update):
        print("X update shape is ", update.shape)
        self.X = np.concatenate((self.X, update))
        mean, var = self.model.predict(update)
        print("m, var are {}, {}".format(mean, var))
        self.Y_mean = np.concatenate((self.Y_mean, mean))
        self.Y_var = np.concatenate((self.Y_var, var))
        print(self.Y_mean)
        print(self.Y_var)

    def update_Y(self, update):
        self.Y = np.concatenate((self.Y, update))

    def plot(self, visible_dims=None):
        pass

    @staticmethod
    def random_parameters(bounds, n_initial):
        assert len(bounds[0]) == len(bounds[1]), \
            "Number of lower and upper bounds don't match!"

        output = [np.random.uniform(bounds[0][i], bounds[1][i],
                                    (n_initial, 1))
                  for i in range(len(bounds[0]))]
        output = np.concatenate(output, axis=1)
        print("HW - Randomly generating parameters: ", output)
        return output

    @staticmethod
    def reformat_bounds(bounds):
        assert len(bounds) == 2, "Unexpected number of bounds!"
        return list(zip(*bounds))
