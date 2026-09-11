"""Evaluate exactly one checkpoint using the same settings as play.py."""

from legged_gym.scripts import play as play_module


if __name__ == "__main__":
    # Reuse every setting and every metric implementation from play.py.
    # The only difference is that no checkpoint loop or 10-episode screening
    # is performed.
    play_module.EXPORT_POLICY = True
    play_module.RECORD_FRAMES = False
    play_module.MOVE_CAMERA = False
    # The original play.py performs screening only in its checkpoint loop.
    # Disable that extra screening for this one-checkpoint run while keeping
    # the same full 40-episode evaluation implementation.
    play_module.SCREENING_EPISODES = 10 ** 9

    args = play_module.get_args()
    if args.checkpoint is None or args.checkpoint < 0:
        raise ValueError(
            "play_single.py requires an explicit --checkpoint, "
            "for example --checkpoint 4000."
        )

    play_module.evaluate_checkpoint(
        args,
        args.checkpoint,
    )
