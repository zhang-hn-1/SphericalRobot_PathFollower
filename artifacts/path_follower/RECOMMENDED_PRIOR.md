# Recommended analytic-prior configuration

Verified on held-out curvature (values absent from every training stage) at two
independent seeds, 2048 environments, one paired episode per environment:

| seed | split | overall | worst (type x curvature) bucket |
|---|---|---|---|
| 7777 | train | 90.4% | 56.2% |
| 7777 | held-out | 88.0% | 63.2% |
| 9999 | held-out | 88.7% | 61.8% |

The reference with only the endgame fix reaches 78.9% / 35.1% on the same
held-out runs, so this configuration adds 9-10 points overall and 24-28 points
on the worst bucket out of sample.

Selected by coordinate descent over 32 parameters against the *smoothed worst
bucket*, not the overall success rate. That choice matters: the configuration
that maximises overall reaches 94.4% on the training curvature but only
85.5-87.0% overall and 37-44% on the worst bucket out of sample.

## How to reproduce

    export PATH_PRIOR_ENDPOINT_FLOOR=1
    export PATH_ENDPOINT_PP=1
    export PATH_ENDPOINT_PP_DISTANCE=0.25
    export PATH_ENDPOINT_BLEND_DISTANCE=0.2
    export PATH_GAIN_MIN=0.24
    export PATH_NORMAL_DRIVE=0.35
    export PATH_NORMAL_HEADING_KP=2
    export PATH_S_HEADING_KP=1
    export PATH_S_MID_HEADING_KP=1.3
    export PATH_S_TIGHT_SHORT_CROSS_TRACK_KP=0.05
    export PATH_TIGHT_HEADING_KP=0.8

Note this only works with `PATH_USE_ACTION_PRIOR=1`. The learned residual was
measured inert, so the prior alone is the policy; the switches above tune it.

## Remaining gap

The worst buckets are s_curve at |k|=0.40-0.45, where failures are dominated by
`deviation` (leaving the 1.5 m corridor) rather than `timeout`. Everything else
sits at 88% or above. That is the next place to look, and it is a steering
problem on sign-changing curvature, not an endgame problem.
