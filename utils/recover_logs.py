import os
import re

import numpy as np


def recover_logs():
    try:
        params = np.load("logs/params.npy").item()
    except IOError:
        return None

    if not any(params):
        return None

    params["init_uc"] = int(params["init_uc"])
    params["uc_runs_per_cn"] = int(params["uc_runs_per_cn"])
    params["init_cn"] = int(params["init_cn"])
    params["batch_size"] = int(params["batch_size"])
    params["contextual"] = bool(params["contextual"])

    print("restoring from params {}".format(params))

    init_uc = params["init_uc"]
    uc_runs_per_cn = params["uc_runs_per_cn"]
    init_cn = params["init_cn"]

    ls = os.popen('ls ./logs/').read()
    try:
        s_bo_x = re.findall(r"(bo[^\s]+x.npy)", ls)[-1]
        s_bo_y = re.findall(r"(bo[^\s]+y.npy)", ls)[-1]
    except IndexError:
        return None

    bo_x = np.load('logs/' + s_bo_x)
    bo_y = np.load('logs/' + s_bo_y)

    n = int(re.findall(r"bo[^\s]+iter-([^\s]+)_y.npy", s_bo_y)[0])
    if params["contextual"]:
        target = (n) * uc_runs_per_cn + init_uc
    else:
        target = target = (n + init_cn) * (uc_runs_per_cn + init_uc)
    s_co_x = r"co[^\s]+iter-{}_x.npy".format(target)
    s_co_y = r"co[^\s]+iter-{}_y.npy".format(target)
    s_co_x = re.findall(s_co_x, ls)[0]
    s_co_y = re.findall(s_co_y, ls)[0]
    co_x = np.load('logs/' + s_co_x)
    co_y = np.load('logs/' + s_co_y)

    return [params, bo_x, bo_y, co_x, co_y]
