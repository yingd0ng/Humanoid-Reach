import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='ArmReach-v0',
    entry_point='humanoid_reach.envs:ArmReachEnv',
    max_episode_steps=400
)

register(
    id='ArmReachWithBlock-v0',
    entry_point='humanoid_reach.envs:ArmReachWithBlockEnv',
    max_episode_steps=400
)

register(
    id='UpperBodyReach-v0',
    entry_point='humanoid_reach.envs:UpperBodyReachEnv',
    max_episode_steps=400
)

register(
    id='UpperBodyReachWithBlock-v0',
    entry_point='humanoid_reach.envs:UpperBodyReachWithBlockEnv',
    max_episode_steps=400
)