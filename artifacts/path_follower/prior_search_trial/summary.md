# Analytic-prior search summary

Evaluations this session: 17; cached total: 17; search wall clock: 3.4 min of 3 min budget.

## Top 5 configurations by smoothed worst-bucket score

`smoothed` is the mean of the 3 worst eligible (path type x curvature) buckets; `worst` is the single worst.

| smoothed | worst | overall | overrides |
|---|---|---|---|
| 47.1% | 31.2% | 79.5% | prior_normal_heading_kp=2, prior_tight_cross_track_kp=0.2 |
| 46.2% | 34.4% | 79.5% | prior_normal_heading_kp=2 |
| 45.8% | 31.2% | 79.9% | prior_normal_heading_kp=2, prior_speed_kp=0.45, prior_tight_cross_track_kp=0.2 |
| 44.4% | 31.2% | 78.7% | prior_normal_heading_kp=2, prior_tight_cross_track_kp=0.2, prior_tight_heading_kp=1.3 |
| 42.3% | 31.2% | 79.3% | prior_normal_heading_kp=2, prior_tight_drive=0.35 |

## Per-bucket detail of the best configuration

| bucket | n | success | endpoint | speed | failures |
|---|---|---|---|---|---|
| left_arc@k0.2 | 21 | 66.7% | 0.199 | 0.060 | timeout=7 |
| left_arc@k0.25 | 27 | 63.0% | 0.199 | 0.062 | timeout=10 |
| left_arc@k0.3333 | 20 | 65.0% | 0.200 | 0.062 | timeout=7 |
| left_arc@k0.4 | 22 | 100.0% | 0.144 | 0.098 | - |
| left_arc@k0.5 | 17 | 100.0% | 0.200 | 0.038 | - |
| right_arc@k-0.2 | 31 | 64.5% | 0.199 | 0.080 | timeout=11 |
| right_arc@k-0.25 | 24 | 83.3% | 0.190 | 0.080 | timeout=4 |
| right_arc@k-0.3333 | 20 | 50.0% | 0.342 | 0.041 | timeout=10 |
| right_arc@k-0.4 | 22 | 100.0% | 0.152 | 0.098 | - |
| right_arc@k-0.5 | 26 | 84.6% | 0.200 | 0.037 | timeout=4 |
| s_curve@k-0.2 | 19 | 94.7% | 0.199 | 0.053 | timeout=1 |
| s_curve@k-0.25 | 21 | 100.0% | 0.199 | 0.053 | - |
| s_curve@k-0.3333 | 21 | 81.0% | 0.200 | 0.050 | timeout=4 |
| s_curve@k-0.4 | 32 | 31.2% | 0.414 | 0.000 | timeout=22 |
| s_curve@k-0.5 | 20 | 60.0% | 0.200 | 0.062 | timeout=8 |
| s_curve@k0.2 | 17 | 100.0% | 0.199 | 0.055 | - |
| s_curve@k0.25 | 15 | 93.3% | 0.199 | 0.052 | timeout=1 |
| s_curve@k0.3333 | 21 | 95.2% | 0.199 | 0.055 | timeout=1 |
| s_curve@k0.4 | 14 | 14.3% | 0.478 | 0.000 | timeout=12 |
| s_curve@k0.5 | 17 | 82.4% | 0.200 | 0.068 | timeout=3 |
| straight@k0 | 85 | 100.0% | 0.190 | 0.093 | - |
