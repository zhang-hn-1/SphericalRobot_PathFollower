# Analytic-prior search summary

Evaluations this session: 4; cached total: 4; search wall clock: 1.0 min of 1 min budget.

## Top 5 configurations by smoothed worst-bucket score

`smoothed` is the mean of the 3 worst eligible (path type x curvature) buckets; `worst` is the single worst.

| smoothed | worst | overall | overrides |
|---|---|---|---|
| 50.4% | 47.7% | 72.0% | prior_normal_drive=0.35 |
| 48.4% | 47.7% | 73.1% | prior_normal_drive=0.35, prior_normal_stop_distance=1.2 |
| 43.4% | 43.1% | 75.5% | defaults |
| 37.2% | 33.0% | 64.6% | prior_normal_drive=0.35, prior_normal_stop_distance=0.6 |

## Per-bucket detail of the best configuration

| bucket | n | success | endpoint | speed | failures |
|---|---|---|---|---|---|
| left_arc@k0.2 | 94 | 57.4% | 0.189 | 0.100 | deviation=32, timeout=8 |
| left_arc@k0.25 | 107 | 55.1% | 0.197 | 0.099 | deviation=41, timeout=7 |
| left_arc@k0.3333 | 101 | 75.2% | 0.181 | 0.099 | deviation=20, timeout=5 |
| left_arc@k0.4 | 69 | 87.0% | 0.159 | 0.085 | timeout=9 |
| left_arc@k0.5 | 89 | 91.0% | 0.199 | 0.067 | timeout=8 |
| right_arc@k-0.2 | 91 | 62.6% | 0.191 | 0.099 | deviation=30, timeout=4 |
| right_arc@k-0.25 | 83 | 55.4% | 0.198 | 0.099 | deviation=35, timeout=2 |
| right_arc@k-0.3333 | 88 | 72.7% | 0.186 | 0.099 | deviation=20, timeout=4 |
| right_arc@k-0.4 | 106 | 79.2% | 0.171 | 0.090 | timeout=22 |
| right_arc@k-0.5 | 104 | 88.5% | 0.199 | 0.071 | deviation=1, timeout=11 |
| s_curve@k-0.2 | 80 | 88.8% | 0.187 | 0.091 | deviation=2, timeout=7 |
| s_curve@k-0.25 | 80 | 78.8% | 0.187 | 0.090 | deviation=1, timeout=17 |
| s_curve@k-0.3333 | 76 | 80.3% | 0.184 | 0.091 | timeout=15 |
| s_curve@k-0.4 | 85 | 48.2% | 0.349 | 0.100 | deviation=15, timeout=29 |
| s_curve@k-0.5 | 83 | 79.5% | 0.176 | 0.094 | deviation=4, timeout=13 |
| s_curve@k0.2 | 63 | 88.9% | 0.189 | 0.090 | deviation=1, timeout=6 |
| s_curve@k0.25 | 55 | 92.7% | 0.183 | 0.086 | timeout=4 |
| s_curve@k0.3333 | 79 | 89.9% | 0.180 | 0.090 | timeout=8 |
| s_curve@k0.4 | 65 | 47.7% | 0.764 | 0.190 | deviation=14, timeout=20 |
| s_curve@k0.5 | 75 | 77.3% | 0.179 | 0.089 | deviation=7, timeout=10 |
| straight@k0 | 375 | 61.9% | 0.193 | 0.096 | deviation=69, timeout=74 |
