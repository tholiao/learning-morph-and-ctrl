import numpy as np

from .coupled_optimizer_augmented import JointOptimizerAug
from .batch_optimizer import BatchOptimizer
import utils

class JointBatchOptimizer(JointOptimizerAug):
    def __init__(self, obj_f, n_uc, init_uc, bounds_uc, uc_runs_per_cn, init_cn,
                 bounds_cn, n_cn, batch_size, contextual=True,
                 uc_to_return='max'):

        bo_x, bo_y, co_x, co_y = [None, None, None, None]
        logs = utils.recover_logs()
        if logs is not None \
                and logs[0]["init_uc"] == init_uc \
                and logs[0]["uc_runs_per_cn"] == uc_runs_per_cn \
                and logs[0]["init_cn"] == init_cn \
                and logs[0]["batch_size"] == batch_size \
                and logs[0]["contextual"] == contextual:
            params, bo_x, bo_y, co_x, co_y = logs
        else:
            params = {"init_uc": init_uc,
                      "uc_runs_per_cn": uc_runs_per_cn,
                      "init_cn": init_cn,
                      "batch_size": batch_size,
                      "contextual": contextual}

        print("params from cbo {}".format(params))
        np.save("logs/params", params)

        super(JointBatchOptimizer, self) \
            .__init__(obj_f, n_uc, init_uc, bounds_uc, uc_runs_per_cn, init_cn,
                      bounds_cn, n_cn, contextual, uc_to_return,
                      start_with_x=co_x, start_with_y=co_y)

        self.hw_optimizer = BatchOptimizer(self.eval_hw, n_cn, bounds_cn,
                                           batch_size, init_cn,
                                           start_with_x=bo_x,
                                           start_with_y=bo_y)
