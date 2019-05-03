import numpy as np
import time

from DIRECT import solve
from scipy.optimize import minimize

from .bayes_optimizer import BayesOptimizer


class JointBayesOptimizer(BayesOptimizer):
    def __init__(self, obj_f, n_uc, init_uc, bounds_uc, uc_runs_per_cn, init_cn,
                 bounds_cn, n_cn, contextual=True, uc_to_return='max',
                 start_with_x=None, start_with_y=None):
        """

        :param obj_f:
        :param n_uc: Number of unconstrained variables (inner loop)
        :param bounds_uc: Bounds on unconstrained variables
        :param n_cn: Number of constrained variables (outer loop)
        :param bounds_cn:
        :param init_uc: number of initial runs of the unconstrained variable
        :param uc_runs_per_cn: Number of iterations to optimize
               unconstrained parameters for each set of constrained parameters
        :param init_cn: number of initial runs of the constrained variable
        :param contextual: whether the inner loop should share GP's across runs
        """

        super(JointBayesOptimizer, self).__init__(obj_f, n_cn + n_uc, bounds_uc,
                                                  init_cn, start_with_x,
                                                  start_with_y)

        self.n_cn = n_cn
        self.init_cn = init_cn
        self.hw_optimizer = BayesOptimizer(self.eval_hw, n_cn, bounds_cn, init_cn)

        self.uc_runs_per_cn = uc_runs_per_cn
        self.init_uc = init_uc
        self.n_uc = n_uc
        self.bounds = bounds_uc
        self.bounds_lower = bounds_uc[0]
        self.bounds_upper = bounds_uc[1]
        if uc_to_return not in ('max', 'min', 'avg', 'med'):
            raise ValueError('{} not valid'.format(uc_to_return))
        self.uc_to_return = uc_to_return
        self.contextual = contextual

    def random_parameters(self, n):
        """
        Initializes software parameters randomly sampled
        :param n: number of parameters to sample
        :return: n x n_sw array
        """
        assert n > 0, "Must initialize at least one set of parameters"
        assert len(self.bounds_lower) == len(self.bounds_upper), \
            "Number of lower and upper bounds don't match!"

        print("SW - Randomly generating {} sets of {} parameters"
              .format(n, self.n_uc))
        output = np.empty((n, self.n_uc))
        for i in range(self.n_uc):
            params = np.random.uniform(self.bounds_lower[i],
                                       self.bounds_upper[i],
                                       (1, n))

            output[:, i] = params
        print("SW - All params: ", output)
        return np.array(output)

    def select_y_to_return(self):
        out = None

        if self.uc_to_return == 'max':
            out = np.max(self.Y[-self.uc_runs_per_cn:])
        elif self.uc_to_return == 'min':
            out = np.min(self.Y[-self.uc_runs_per_cn:])
        elif self.uc_to_return == 'med':
            out = np.median(self.Y[-self.uc_runs_per_cn:])
        elif self.uc_to_return == 'avg':
            out = np.average(self.Y[-self.uc_runs_per_cn:])

        return np.array([[out]])

    def initialize_GP(self, n_init, x_cn):
        """

        :param x_cn:
        :return:
        """
        self.update_iterations(i=self.init_uc)

        x_uc = self.random_parameters(self.init_uc)
        x_cn = np.tile(x_cn, (self.init_uc, 1))

        self.X = np.concatenate((x_uc, x_cn), axis=1)
        self.Y = self.evaluate(self.X)
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
        print(self.X)
        print(self.Y)

        return self.select_y_to_return()

    def optimize(self, total):
        """
        :param total:
        :param inner_loop: Determines number of loops inner software
        optimization will do each time it's called by hw_optimizer
        """
        self.hw_optimizer.optimize(total)

    def optimize_acq_f(self, x_cn):
        def obj_sw_DIRECT(x_uc, user_data):
            x_uc = np.array(x_uc).reshape((1, self.n_uc))
            return -self.acq_f(np.concatenate((x_uc, x_cn), axis=1)), 0

        def obj_sw_LBFGS(x_uc):
            x_uc = np.array(x_uc).reshape((1, self.n_uc))
            return -self.acq_f(np.concatenate((x_uc, x_cn), axis=1))

        x, _, _ = solve(obj_sw_DIRECT, self.bounds_lower,
                        self.bounds_upper, maxf=500)
        x = minimize(obj_sw_LBFGS, x, method='L-BFGS-B',
                     bounds=self.reformat_bounds(self.bounds)).x
        return np.array(x).reshape((1, self.n_uc))

    def acq_f(self, x, alpha=-1, v=.01, delta=.1):
        """
        Implementation of GP-UCB
        :param x:
        :param alpha: hyperparameter
        :param v: hyperparameter
        :param delta: hyperparameter
        :return:
        """
        x = x.reshape(1, self.n_uc + self.n_cn)
        mean, var = self.model.predict(x)
        if alpha is -1:
            alpha = np.sqrt(v * (2 * np.log((self.iterations
                                             ** ((self.n_uc / 2) + 2))
                                            * (np.pi ** 2) / (3 * delta))))

        return mean + (alpha * var)
