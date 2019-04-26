import traceback

import numpy as np

from walkers.scalable_walker import ScalableWalker
from utils.vrep_exceptions import *
from cpg.cpg_gaits import DualTripod
from .simulation import Simulation
from utils import DEFAULT_SCENE, DEFAULT_WALKER
from cpg import CpgController
import vrep_api.vrep as vrep


class DistanceSimulation(Simulation):
    def __init__(self, scene=DEFAULT_SCENE, walker=DEFAULT_WALKER):
        super(DistanceSimulation, self).__init__(scene, walker)
        self.cpg = None

    def set_cpg(self, gait, f, phase_offset):
        cpg_gait = gait
        cpg_gait.f = f
        cpg_gait.phase_offset = phase_offset
        cpg_gait.coupling_phase_biases = \
            cpg_gait.generate_coupling(cpg_gait.phase_groupings)
        self.cpg = CpgController(cpg_gait)
        for _ in range(1000):
            self.cpg.update(plot=False)

    def walk(self, steps):
        assert self.cpg is not None, "Can't walk without a cpg loaded!"

        for _ in range(steps):
            output = self.cpg.output()
            for i in range(6):
                self.walker.legs[i].extendZ(output[i][0])
                self.walker.legs[i].extendX(output[i][1])
            self.wait(1)
            self.cpg.update()

    def get_obj_f(self, max_steps, gait=DualTripod):
        self.start()
        self.load_scene()
        self.load_walker()

        def objective(x):
            x = np.asarray(x)[0]
            print('\nParameters: ' + str(x))

            try:
                # Clean up VREP
                self.run()

                # Get starting position
                start = self.get_pos(self.walker.base_handle)

                # Assign parameters
                self.walker.set_left_bias(x[2])
                self.walker.set_right_bias(x[3])
                self.set_cpg(gait, x[0], x[1])
                for _ in range(1000):
                    self.cpg.update(plot=False)

                # Run the simulation
                print('Running trial...')
                self.walk(max_steps)

                # Calculate how far the robot walked
                end = self.get_pos(self.walker.base_handle)
                dist = self.calc_dist(start, end)
                print('Distance traveled: ' + str(dist))

                # Clean up VREP
                self.stop()

                return np.array([dist])

            except (ConnectionError,
                    SceneLoadingError,
                    WalkerLoadingError,
                    MotorLoadingError,
                    HandleLoadingError) as e:
                print("Encountered an exception: {} "
                      "disconnecting from remote API server"
                      .format(e))
                vrep.simxFinish(self.client_id)
                traceback.print_exc()
                self.stop()
                raise e

        return objective

    @staticmethod
    def calc_dist(start, end):
        dist = start[0] - end[0]
        return dist + (.1 * np.abs(end[1] - start[1]))
