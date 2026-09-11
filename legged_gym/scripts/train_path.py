"""Train the path task without randomizing initial episode age."""

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def train(args):
    env, _ = task_registry.make_env(name=args.task, args=args)
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args
    )
    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=False,
    )


if __name__ == "__main__":
    train(get_args())
