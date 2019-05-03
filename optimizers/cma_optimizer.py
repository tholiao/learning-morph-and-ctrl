import time

import numpy as np
import cma


class CMAESOPtimizer():
    def __init__(self, obj_f, bounds, popsize):
        np.save("params.npy", np.array([bounds, popsize]))
        self.obj_f = obj_f
        self.bounds = bounds
        self.num_variables = len(bounds[0])

        self.iterations = 0
        self.X = None
        self.Y = None

        self.sigma = .6

        self.popsize = popsize

    def transform_variable(self, val, lower_bound, upper_bound):
        """
        from [0, 100] to [lower_bound, upper_bound]
        """
        return (upper_bound - lower_bound) * (val / 100.) + lower_bound

    def normalize_variable(self, val, lower_bound, upper_bound):
        """
        from [lower_bound, upper_bound] to [0, 100]
        """
        return (val - lower_bound) / (upper_bound - lower_bound) * 100.

    def transform_x(self, x):
        x_new = []
        for val, lower_bound, upper_bound in zip(x[0],
                                                 self.bounds[0],
                                                 self.bounds[1]):
            x_new.append(self.transform_variable(val, lower_bound, upper_bound))
        return np.array(x_new)

    def normalize_x(self, x):
        x_new = []
        for val, lower_bound, upper_bound in zip(x,
                                                 self.bounds[0],
                                                 self.bounds[1]):
            x_new.append(self.normalize_variable(val, lower_bound, upper_bound))
        return np.array(x_new)

    def random_parameters(self, n_initial):
        return np.random.uniform(0, 100, (self.num_variables,))

    def wrapper(self, x):
        """
        wrap input to the obj f in a numpy array after clipping elements
        x is normalized as an input
        """
        print("CMA - ", x)
        self.iterations += 1

        x = np.array([x])
        x = self.transform_x(x)
        y = self.obj_f(x)

        if self.X is None or self.Y is None:
            self.X = x
            self.Y = y
        else:
            self.update_X(x)
            self.update_Y(y)

        if self.iterations % self.popsize == 0:
            np.save("./logs/CMA_{}_iter-{}_x".format(
                time.strftime("%Y.%m.%d-%H.%M.%S"), self.iterations), self.X)
            np.save("./logs/CMA_{}_iter-{}_y".format(
                time.strftime("%Y.%m.%d-%H.%M.%S"), self.iterations), self.Y)

        # CMA minimizes
        return -y

    def update_X(self, update):
        print("X update shape is ", update.shape)
        self.X = np.concatenate((self.X, update))

    def update_Y(self, update):
        print("Y is ", update)
        self.Y = np.concatenate((self.Y, update))

    def optimize(self, total):
        init_params = self.random_parameters(1)

        bounds_norm = [[0] * self.num_variables, [100] * self.num_variables]

        res = cma.fmin(self.wrapper, init_params, self.sigma,
                       options={"bounds": bounds_norm, "verbose": -1,
                                "maxfevals": total,
                                "popsize": self.popsize})
