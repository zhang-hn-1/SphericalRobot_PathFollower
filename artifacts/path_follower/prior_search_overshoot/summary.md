# Analytic-prior search summary

Evaluations this session: 113; cached total: 113; search wall clock: 30.1 min of 30 min budget.

## Top 5 configurations by smoothed worst-bucket score

`smoothed` is the mean of the 3 worst eligible (path type x curvature) buckets; `worst` is the single worst.

| smoothed | worst | overall | overrides |
|---|---|---|---|
| 81.6% | 76.9% | 89.9% | prior_endpoint_blend_distance=0.2, prior_endpoint_max_curvature=0.6, prior_gain_max=0.85, prior_gain_offset=0.062, prior_normal_drive=0.35, prior_normal_heading_kp=2, prior_s_cross_track_kp=0.15, prior_s_heading_kp=1.3, prior_s_mid_drive=0.35, prior_s_mid_heading_kp=0.7, prior_s_mid_stop_distance=2.6, prior_s_regular_long_cross_track_kp=0.1, prior_s_short_drive=0.35, prior_s_tight_short_cross_track_kp=0.05, prior_speed_kp=1.05 |
| 81.5% | 76.9% | 90.1% | prior_endpoint_blend_distance=0.2, prior_endpoint_max_curvature=0.6, prior_gain_max=0.85, prior_gain_offset=0.062, prior_normal_drive=0.35, prior_normal_heading_kp=2, prior_s_cross_track_kp=0.15, prior_s_heading_kp=1.3, prior_s_mid_drive=0.35, prior_s_mid_heading_kp=0.7, prior_s_mid_stop_distance=2.6, prior_s_short_drive=0.35, prior_s_tight_short_cross_track_kp=0.05, prior_speed_kp=1.05 |
| 81.5% | 75.4% | 89.7% | prior_endpoint_blend_distance=0.2, prior_endpoint_max_curvature=0.6, prior_gain_max=0.85, prior_gain_offset=0.062, prior_normal_drive=0.35, prior_normal_heading_kp=2, prior_s_cross_track_kp=0.15, prior_s_heading_kp=1.3, prior_s_mid_drive=0.35, prior_s_mid_heading_kp=0.7, prior_s_mid_stop_distance=2.6, prior_s_short_drive=0.35, prior_s_tight_long_heading_kp=0.7, prior_s_tight_short_cross_track_kp=0.05, prior_speed_kp=1.05 |
| 80.8% | 75.4% | 91.2% | prior_endpoint_blend_distance=0.2, prior_endpoint_max_curvature=0.6, prior_gain_max=0.85, prior_gain_offset=0.062, prior_normal_drive=0.35, prior_normal_heading_kp=2, prior_s_cross_track_kp=0.15, prior_s_heading_kp=1.3, prior_s_mid_drive=0.35, prior_s_mid_heading_kp=0.7, prior_s_mid_stop_distance=2.6, prior_s_short_drive=0.35, prior_s_tight_short_cross_track_kp=0.05, prior_s_tight_short_heading_kp=0.3, prior_speed_kp=1.05 |
| 80.1% | 73.8% | 90.5% | prior_endpoint_blend_distance=0.2, prior_endpoint_max_curvature=0.6, prior_gain_max=0.85, prior_gain_offset=0.062, prior_normal_drive=0.35, prior_normal_heading_kp=2, prior_s_cross_track_kp=0.15, prior_s_heading_kp=1.3, prior_s_mid_drive=0.35, prior_s_mid_heading_kp=0.7, prior_s_mid_stop_distance=2.6, prior_s_short_drive=0.35, prior_s_tight_long_heading_kp=0.3, prior_s_tight_short_cross_track_kp=0.05, prior_speed_kp=1.05 |

## Per-bucket detail of the best configuration

| bucket | n | success | endpoint | speed | failures |
|---|---|---|---|---|---|
| left_arc@k0.2 | 94 | 94.7% | 0.127 | 0.092 | timeout=5 |
| left_arc@k0.25 | 107 | 91.6% | 0.140 | 0.093 | timeout=9 |
| left_arc@k0.3333 | 101 | 92.1% | 0.172 | 0.096 | deviation=2, timeout=6 |
| left_arc@k0.4 | 69 | 91.3% | 0.171 | 0.091 | timeout=6 |
| left_arc@k0.5 | 89 | 91.0% | 0.199 | 0.072 | timeout=8 |
| right_arc@k-0.2 | 91 | 94.5% | 0.139 | 0.090 | timeout=5 |
| right_arc@k-0.25 | 83 | 92.8% | 0.154 | 0.093 | deviation=1, timeout=5 |
| right_arc@k-0.3333 | 88 | 88.6% | 0.164 | 0.096 | deviation=2, timeout=8 |
| right_arc@k-0.4 | 106 | 86.8% | 0.165 | 0.092 | timeout=14 |
| right_arc@k-0.5 | 104 | 88.5% | 0.199 | 0.071 | timeout=12 |
| s_curve@k-0.2 | 80 | 88.8% | 0.183 | 0.092 | timeout=9 |
| s_curve@k-0.25 | 80 | 93.8% | 0.180 | 0.092 | timeout=5 |
| s_curve@k-0.3333 | 76 | 93.4% | 0.191 | 0.085 | timeout=5 |
| s_curve@k-0.4 | 85 | 81.2% | 0.187 | 0.081 | timeout=16 |
| s_curve@k-0.5 | 83 | 89.2% | 0.177 | 0.091 | timeout=9 |
| s_curve@k0.2 | 63 | 96.8% | 0.182 | 0.092 | timeout=2 |
| s_curve@k0.25 | 55 | 90.9% | 0.177 | 0.084 | timeout=5 |
| s_curve@k0.3333 | 79 | 91.1% | 0.189 | 0.086 | timeout=7 |
| s_curve@k0.4 | 65 | 76.9% | 0.182 | 0.083 | timeout=15 |
| s_curve@k0.5 | 75 | 86.7% | 0.174 | 0.093 | timeout=10 |
| straight@k0 | 375 | 89.1% | 0.171 | 0.093 | timeout=41 |
