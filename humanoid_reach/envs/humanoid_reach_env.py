import numpy as np
from humanoid_reach.envs import humanoid_env


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


def get_robot_obs(sim):
    """Returns all joint positions and velocities associated with
    a robot.
    """
    if sim.data.qpos is not None and sim.model.joint_names:
        names = [n for n in sim.model.joint_names]
        return (
            np.array([sim.data.get_joint_qpos(name) for name in names]),
            np.array([sim.data.get_joint_qvel(name) for name in names]),
        )
    return np.zeros(0), np.zeros(0)


class HumanoidReachEnv(humanoid_env.HumanoidEnv):
    """Superclass for all Reach environments.
    """

    def __init__(
        self, model_path, n_substeps, target_range,
        distance_threshold, initial_qpos, ctrl_ratio, impact_ratio
    ):
        """Initializes a new Reach environment.

        Args:
            model_path (string): path to the environments XML file
            n_substeps (int): number of substeps the simulation runs on every call to step
            target_range (float): range of a uniform distribution for sampling initial target position
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            ctrl_ratio (float): coefficient for the control cost
            impact_ratio (float): coefficient for the impact cost
        """
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.ctrl_ratio = ctrl_ratio
        self.impact_ratio = impact_ratio
        super(HumanoidReachEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps,
            initial_qpos=initial_qpos)

    def _set_action(self, action):
        self.sim.data.ctrl[:] = action

    def _get_obs(self):
        rhand_pos = self.sim.data.get_site_xpos('rhandsite')
        achieved_goal = rhand_pos.copy()
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        robot_qpos, robot_qvel = get_robot_obs(self.sim)
        robot_velp = robot_qvel * dt

        obs = np.concatenate([
            robot_qpos, robot_velp, [-goal_distance(achieved_goal, self.goal)]
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        body_id = self.sim.model.site_name2id('target0')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2
        self.viewer.cam.azimuth = 20
        self.viewer.cam.elevation = -40.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos -
                        self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[2]
        self.sim.forward()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _sample_goal(self):
        goal = self.initial_target_xpos[:3] + \
            self.np_random.uniform(-self.target_range,
                                   self.target_range, size=3)
        goal[2] = self.initial_target_xpos[2]
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

        # Extract information for sampling goals.
        self.initial_target_xpos = self.sim.data.get_site_xpos(
            'target0').copy()
