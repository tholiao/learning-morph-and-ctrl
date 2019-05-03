import numpy as np
import time

from .bayes_optimizer import BayesOptimizer


class RandomOptimizer(BayesOptimizer):
    def optimize(self, total):
        self.update_iterations(self.n_init)

        self.X = self.optimize_acq_f()
        self.Y = self.evaluate(self.X)

        # Logging
        print("SOFTWARE LOG Iteration {}:".format(self.iterations))
        np.save("./logs/ro_{}_iter-{}_x".format(
            time.strftime("%Y.%m.%d-%H.%M.%S"), self.iterations), self.X)
        np.save("./logs/ro_{}_iter-{}_y".format(
            time.strftime("%Y.%m.%d-%H.%M.%S"), self.iterations), self.Y)
        print(self.X)
        print(self.Y)

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
            np.save("./logs/random_{}_iter-{}_x".format(
                time.strftime("%Y.%m.%d-%H.%M.%S"), self.iterations), X)
            np.save("./logs/random_{}_iter-{}_y".format(
                time.strftime("%Y.%m.%d-%H.%M.%S"), self.iterations), Y)
        Y = np.array(np.abs(Y))
        print("SW - Evaluated to ", Y)
        return Y

    def optimize_acq_f(self):
        return self.random_parameters(self.bounds, self.n_init)
