from DIRECT import solve
from scipy.optimize import minimize

import numpy as np

from .single_optimizer import Optimizer


class ContextualOptimizer(Optimizer):
    def __init__(self, obj_f, num_inputs, num_contexts, bounds):
        super(ContextualOptimizer, self).__init__(obj_f=obj_f,
                                                  num_inputs=num_inputs
                                                             + num_contexts,
                                                  bounds=bounds,
                                                  n_init=0)

    def optimize_acq_f(self, context):
        def obj_sw_DIRECT(x, user_data):
            return -self.acq_f(x), 0

        def obj_sw_LBFGS(x_sw):
            return -self.acq_f(x_sw)

        x, _, _ = solve(obj_sw_DIRECT, self.bounds_lower,
                        self.bounds_upper, maxf=500)
        return np.array(minimize(obj_sw_LBFGS, x, method='L-BFGS-B',
                                 bounds=self.reformat_bounds(self.bounds)).x)

    def optimize(self, context, n_iterations):
        for i in range(n_iterations):
            self.update_iterations(i)

            X = self.optimize_acq_f()
            self.update_X(X)
            Y = self.evaluate(np.array([X]))
            self.update_Y(Y)

            self.train_GP()
            self.model.optimize()
