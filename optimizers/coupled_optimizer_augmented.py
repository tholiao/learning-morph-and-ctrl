import numpy as np
import time

from DIRECT import solve
from scipy.optimize import minimize

from .coupled_optimizer import JointBayesOptimizer


class JointOptimizerAug(JointBayesOptimizer):
    def __init__(self, obj_f, n_uc, init_uc, bounds_uc, uc_runs_per_cn, init_cn,
                 bounds_cn, n_cn, contextual=True, uc_to_return='max',
                 start_with_x=None, start_with_y=None):

        super(JointOptimizerAug, self).__init__(obj_f, n_uc, init_uc, bounds_uc,
                                                uc_runs_per_cn, init_cn,
                                                bounds_cn, n_cn,
                                                contextual=contextual,
                                                uc_to_return=uc_to_return,
                                                start_with_x=start_with_x,
                                                start_with_y=start_with_y)


    def initialize_GP(self, n_init, x_cn):
        self.update_iterations(i=self.init_uc)

        x_uc = self.random_parameters(self.init_uc)
        x_cn = np.tile(x_cn, (self.init_uc, 1))

        self.X = np.concatenate((x_uc, x_cn), axis=1)
        self.Y = self.evaluate(self.X)
        self.Y_mean = np.zeros((self.X.shape[0], 1))
        self.Y_var = np.zeros((self.X.shape[0], 1))
        self.train_GP(self.X, self.Y)

        self.optimize_model()

        print("Done initializing GP_uc")

    def eval_hw(self, x_cn, cache_walker=True):
        """
        Used as objective function by hw_optimizer.
        Given a context, optimize x_sw and return the reward from obj_f
        :param x_cn: Used as a context during optimization
        :return: Reward from obj_f
        """
        if not self.contextual \
                or self.model is None:
            self.initialize_GP(self.init_uc, x_cn)

        print("SW - optimizing with {} as context".format(x_cn))

        for i in range(self.uc_runs_per_cn):
            self.update_iterations()

            x_uc = self.optimize_acq_f(x_cn)
            X = np.concatenate((x_uc, x_cn), axis=1)
            Y = self.evaluate(X)
            self.update_X(X)
            self.update_Y(Y)

            self.train_GP(self.X, self.Y)
            self.optimize_model()

        # Logging
        print("SOFTWARE LOG Iteration {}:".format(self.iterations))
        np.save("./logs/co_{}_iter-{}_x".format(
            time.strftime("%Y.%m.%d-%H.%M.%S"), self.iterations), self.X)
        np.save("./logs/co_{}_iter-{}_y".format(
            time.strftime("%Y.%m.%d-%H.%M.%S"), self.iterations), self.Y)
        np.save("./logs/co_{}_iter-{}_y_m".format(
            time.strftime("%Y.%m.%d-%H.%M.%S"), self.iterations), self.Y_mean)
        np.save("./logs/co_{}_iter-{}_y_v".format(
            time.strftime("%Y.%m.%d-%H.%M.%S"), self.iterations), self.Y_var)
        print(self.X)
        print(self.Y)

        return self.select_y_to_return()
