import time

import numpy as np

from .bayes_optimizer import BayesOptimizer


class BatchBayesOptimizer(BayesOptimizer):
    def __init__(self, obj_f, num_inputs, bounds, batch_size, n_init,
                 start_with_x=None, start_with_y=None):
        super(BatchBayesOptimizer, self).__init__(obj_f, num_inputs, bounds, n_init,
                                                  start_with_x=start_with_x,
                                                  start_with_y=start_with_y)
        self.batch_size = batch_size

    def optimize(self, total):
        if self.model is None:
            self.initialize_GP(self.n_init)

        for i in range(total // self.batch_size):
            print("Batch #", i)
            B_x = np.zeros((self.batch_size, self.num_inputs))
            B_y = np.zeros((self.batch_size, 1))

            for j in range(self.batch_size):
                self.update_iterations()

                # Update batch with parameters
                X = self.optimize_acq_f()
                B_x[j] = X
                temp_x = B_x[:j + 1, :].reshape(j + 1, self.num_inputs)
                temp_x = np.concatenate((self.X, temp_x))

                # 'Hallucinate' results to update GP with
                Y, _ = self.model.predict(X)
                B_y[j] = Y
                temp_y = B_y[:j + 1].reshape(j + 1, 1)
                temp_y = np.concatenate((self.Y, temp_y))

                # Update GP with hallucinated results
                self.train_GP(temp_x, temp_y)
                self.model.optimize()

            print("Batch: ", B_x)
            # Update parameters and rewards
            self.update_X(B_x)
            self.update_Y(self.evaluate(B_x))

            # Logging
            print("Batch Optimizer ", self.X, self.Y)
            np.save("./logs/bo_{}_iter-{}_x".format(
                time.strftime("%Y.%m.%d-%H.%M.%S"), self.iterations), self.X)
            np.save("./logs/bo_{}_iter-{}_y".format(
                time.strftime("%Y.%m.%d-%H.%M.%S"), self.iterations), self.Y)
            # Train the GP
            self.train_GP(self.X, self.Y)
            self.optimize_model()

        print("FINISHED OPTIMIZATION")
