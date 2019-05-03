# !/usr/bin/env python -W ignore::DeprecationWarning

import argparse
import time

import numpy as np

from objectives import HwSwDistSim, JointObj2, JointObj3
from optimizers import JointBatchOptimizer, \
    CMAESOPtimizer, \
    RandomOptimizer, \
    BayesOptimizer
from utils import *


def cli_main(args):
    init_uc = args.init_uc
    init_cn = args.init_cn
    uc_runs_per_cn = args.uc_runs_per_cn
    batch_size = args.batch_size
    total = args.total
    obj_f = args.obj_f
    contextual = args.contextual
    popsize = args.popsize
    num_inputs = N_CTRL_PARAMS[obj_f] + N_MRPH_PARAMS[obj_f]
    joint_bounds = np.hstack((np.array(CONTROLLER_BOUNDS[obj_f]),
                              np.array(MORPHOLOGY_BOUNDS[obj_f])))

    if args.obj_f == 0:
        sim = HwSwDistSim()
    elif args.obj_f == 1:
        sim = JointObj2()
    elif args.obj_f == 2:
        sim = JointObj3()
    else:
        raise ValueError("Objective function must be: 0, 1, or 2")

    if args.optimizer == 'bayesopt':
        optimizer = BayesOptimizer(obj_f=sim.get_obj_f(max_steps=401),
                                   num_inputs=num_inputs,
                                   bounds=joint_bounds,
                                   n_init=init_uc)
    elif args.optimizer == 'cmaes':
        optimizer = CMAESOPtimizer(obj_f=sim.get_obj_f(max_steps=401),
                                   bounds=joint_bounds,
                                   popsize=popsize)
    elif args.optimizer == 'random':
        optimizer = RandomOptimizer(obj_f=sim.get_obj_f(max_steps=401),
                                   num_inputs=num_inputs,
                                   bounds=joint_bounds,
                                   n_init=init_uc)
    elif args.optimizer == 'hpcbbo':
        optimizer = JointBatchOptimizer(obj_f=sim.get_obj_f(max_steps=401),
                                  n_uc=N_CTRL_PARAMS[obj_f],
                                  init_uc=init_uc,
                                  bounds_uc=CONTROLLER_BOUNDS[obj_f],
                                  uc_runs_per_cn=uc_runs_per_cn,
                                  init_cn=init_cn,
                                  bounds_cn=MORPHOLOGY_BOUNDS[obj_f],
                                  n_cn=N_MRPH_PARAMS[obj_f],
                                  batch_size=batch_size,
                                  contextual=contextual)

    optimizer.optimize(total=total)
    sim.exit()
    np.save("./logs/main_{}_iter-{}_x".format(
        time.strftime("%Y.%m.%d-%H.%M.%S"), optimizer.iterations), optimizer.X)
    np.save("./logs/main_{}_iter-{}_y".format(
        time.strftime("%Y.%m.%d-%H.%M.%S"), optimizer.iterations), optimizer.Y)


if __name__ == "__main__":
    np.set_printoptions(precision=4)

    parser = argparse.ArgumentParser()
    parser.add_argument('optimizer', type=str,
                        help="Choose an optimizer. Valid optimizers are 'cma-es', 'hpcbbo',"
                             "'bayesopt', and 'random'")
    parser.add_argument('--init_uc', type=int,
                        help="Number of initial control optimization loops",
                        default=5)
    parser.add_argument('--init_cn', type=int,
                        help="Number of initial morphology optimization loops",
                        default=5)
    parser.add_argument('--uc_runs_per_cn', type=int,
                        help="Ratio of morphology opt. loops to opt. control loops",
                        default=50)
    parser.add_argument('--batch_size', type=int,
                        help="Morphology batch size", default=5)
    parser.add_argument('--total', type=int,
                        help="Number of total morphology optimization loops",
                        default=100)
    parser.add_argument('--obj_f', type=int,
                        help="Switch between different objective functions",
                        default=0)
    parser.add_argument('--popsize', type=int,
                        help='Population size parameter for cmaes mode')
    parser.add_argument('--contextual', action="store_true",
                        help="Toggle contextual optimization. Only meaningful for optimizers"
                             "'hpcbbo' and 'bayesopt'")
    args = parser.parse_args()
    cli_main(args)
