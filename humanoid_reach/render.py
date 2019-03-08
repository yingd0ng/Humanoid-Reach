# For usage, refer to the readme.
# For testing purpose only.

import gym
import humanoid_reach  # pylint: Add to register gym-balls environments
import numpy as np
import time

from baselines.common.cmd_util import common_arg_parser
from baselines import logger
from baselines.run import train, parse_cmdline_kwargs, build_env


def main():
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args()
    args.num_env = 1
    extra_args = parse_cmdline_kwargs(unknown_args)

    model, env = train(args, extra_args)
    env.close()
    logger.log("Running trained model")
    env = build_env(args)
    if not args.play:
        ts = time.gmtime()
        directory = time.strftime("./render/%s", ts)
        logger.log("Output video to directory:", directory)
        env.envs = [gym.wrappers.Monitor(env.envs[0], directory=directory)]
    obs = env.reset()

    def initialize_placeholders(nlstm=128, **kwargs):
        return np.zeros((args.num_env, 2 * nlstm)), np.zeros((1))

    state, dones = initialize_placeholders(**extra_args)
    NUM_VIDEO = 1
    while True:
        actions, _, state, _ = model.step(obs, S=state, M=dones)
        obs, _, done, _ = env.step(actions)
        if args.play:
            env.render()
        done = done.any() if isinstance(done, np.ndarray) else done

        if done:
            NUM_VIDEO -= 1
            if NUM_VIDEO <= 0:
                break
            obs = env.reset()

    env.close()


if __name__ == '__main__':
    main()
