from gym import utils
from humanoid_reach.envs import humanoid_reach_env

import numpy as np


class UpperBodyReachWithBlockEnv(humanoid_reach_env.HumanoidReachEnv, utils.EzPickle):
    def __init__(self):
        initial_qpos = {
            "rhumerusrx": -0.4,
            "rhumerusrz": -1.0,
            # "rradiusrx" : 1.0,
        }
        humanoid_reach_env.HumanoidReachEnv.__init__(
            self, model_path='upper_body_reach_with_block.xml', n_substeps=1,
            target_range=0.1, distance_threshold=0.01,
            initial_qpos=initial_qpos, ctrl_ratio=0.01,
            impact_ratio=.5e-4)
        utils.EzPickle.__init__(self)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = humanoid_reach_env.goal_distance(achieved_goal, goal)
        reward = -d
        data = self.sim.data
        quad_ctrl_cost = self.ctrl_ratio * np.square(data.ctrl).sum()
        quad_impact_cost = self.impact_ratio * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward -= quad_ctrl_cost + quad_impact_cost
        return reward
