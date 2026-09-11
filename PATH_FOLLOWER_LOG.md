# Rotunbot geometric path follower experiment log

## Scope

- Server-only project: `/data/lzq/workspace/SphericalRobot_PathFollower_20260911`
- Robot model: `Rotunbot_test2.urdf`
- Task: geometric path following with no required travel speed and no terminal yaw.
- Actions: joint-1 target velocity and joint-2 target angle through the existing servo model.
- Final planner interface: a current 10-point local pose path, later supplied by NeuPAN.

## Source audit

- `rotunbot_target_repro`: retained the 50 Hz direct-joint control idea and the 20-frame DWL history family.
- `rotunbot_vel_clean`: retained actuator target filtering, target rate limits, servo gains, stop threshold, and lessons from signed-curvature training.
- Legacy `rotunbot_tra`: not used as the final task because it follows one time-indexed absolute XY point on one fixed curve, uses the old URDF, and lacks path preview/projection.
- Contact yaw damping was ported from `SphericalRobot_LeggedGym-master-new-map`; generic global angular damping remains zero.

## V1 observations and policy

- Historical observation: 19 values x 20 frames.
- The first four historical values are local nearest-path geometry `[x_B, y_B, cos(dyaw), sin(dyaw)]`; no global XY target is exposed.
- Remaining historical state: base quaternion, body linear/angular velocity, joint-2 position, both joint velocities, and previous action.
- Current preview: 10 points x `[x_B, y_B, cos(dyaw), sin(dyaw)]` = 40 values.
- Actor input: latest 5 historical frames (95), 16-D DWL history latent, and 40-D preview = 151.
- Actor MLP: 151-512-256-128-2. Critic MLP: 28-512-256-128-1.
- PPO evaluation always uses the action mean; the environment clips physical policy actions to [-1, 1].

## V1 actuator and physics settings

- Policy period: 0.02 s (50 Hz).
- Joint 1: target velocity, +/-3 rad/s.
- Joint 2: target angle, +/-0.5236 rad mechanical range; no legacy +/-0.45 rad hard clip.
- Target low-pass factors: joint 1 = 0.50, joint 2 = 0.40.
- Per-policy-step target change limits: joint 1 = 0.08 rad/s target units, joint 2 = 0.02 rad.
- Contact yaw damping: shell `link1`, viscous 2.0 Nms/rad, Coulomb 0.5 Nm, maximum 2.0 Nm, active only when supported and attenuated during normal rolling.

## Path and reward

- Dense path spacing: 0.05 m; current progress is the monotonic nearest sample in a local search window.
- Preview distances: 0.2 to 2.0 m at 0.2 m increments.
- Positive reward comes from actual arc-length progress and task completion.
- Cross-track error, tangent-heading error, endpoint speed, action/target changes, torque, joint-limit proximity, and elapsed time are costs.
- Completion requires remaining arc length <=0.20 m, endpoint distance <=0.20 m, and planar speed <=0.10 m/s. Terminal yaw is unrestricted.
- A weak signed-curvature/joint-2 target cost was added after the policy repeated the known velocity-controller failure of choosing the same steering sign for both turn directions.

## Curriculum

1. Stage 0: aligned straight paths, length 2-4 m.
2. Stage 1: straight and R=4/3 m arcs, length 3-6 m.
3. Stage 2: straight, R=4/3/2 m arcs and S-curves, length 3-8 m.
4. Stage 3: same path family with up to 0.20 m lateral and 10 degree heading initial errors.

A stage advances only after a 4096-episode window reaches 75% success. Episode reset clears actions, filtered joint targets, path/preview/progress, actor and critic histories, and velocity-task histories.

## Runs and measured results

### Smoke tests

- 64 environments, 2 PPO updates: actor/critic dimensions, physics, reward lookup, checkpoint save, and reset completed.
- The second smoke run printed the enabled contact yaw damping parameters.

### Initial from-scratch run

- Run directory: `logs/rotunbot_path/Sep11_01-01-34_geometric_path_v1_from_scratch`
- Stage 0 -> 1: 82.01% curriculum-window success.
- Stage 1 -> 2: 77.22% curriculum-window success.
- `model_200`: fixed mean-action straight paths 256/256 success; terminal-speed P95 0.099 m/s.
- `model_600`: R=4 m right arcs 32/32 success; left arcs 16/32 success.
- `model_800`: immediately after Stage 2 opened, R=2 m left/right both 0/64 and S-curves 6/64; the joint-2 mean target had not separated turn signs.
- The initial process stopped only after `model_1000.pt` was saved.

### Signed-steering continuation

- Continuation source: the same from-scratch chain at `model_1000.pt`, including PPO optimizer and path curriculum state.
- Added costs: `steering_direction=-0.30`, `steering_centering=-0.10`.
- Runtime log: `artifacts/path_follower/v1_direction_resume1000/train.log`.
- This is curriculum continuation, not initialization from an external or pretrained policy.

## Evaluation protocol

- Fixed seed and deterministic mean actions.
- Report each of straight, left arc, right arc, S-curve, and initial-offset recovery separately.
- Mixed success is never used to hide a failing path type.
- Store `metrics.json`, per-episode CSV, one full path/robot trace NPZ, and a trajectory/action plot for every evaluated case.

## Current open items

- Confirm signed-steering continuation fixes both R=2 m directions rather than one side only.
- Complete Stage 3 recovery training.
- Evaluate unseen analytic paths before feeding exported NeuPAN paths.
- Add NeuPAN path resampling and online replan tests only after the low-level gates pass.

## V1 deterministic regression and V2 response

- `model_1200` after the signed-steering continuation separated the left/right steering signs, but deterministic mean-action results were still poor: R=2 m left 4.69%, R=2 m right 0%, S-curve 0%; failures were path deviation with nearly saturated steering.
- Easier controls confirmed catastrophic forgetting rather than only an R=2 limit: R=4 m left 10.94%, R=4 m right 6.25%, straight 70.31%, while `model_200` had achieved 100% straight.
- V1 rewarded monotonic nearest-point progress even when the robot cut across a curve, introduced R=2 and S-curves in one transition, permitted constant arcs longer than a half-circle, and let the Gaussian action mean leave [-1,1] before environment clipping.

## V2 from-scratch design

- Start a new random-weight run; V1 checkpoints remain immutable evidence and are not used to initialize V2.
- Encode the 20 x 19 history with temporal Conv1d (19 channels over 20 time samples), producing a 32-D latent. Append five raw recent frames, ten local preview poses, local endpoint x/y, and remaining arc length. Actor input is 170 values; MLP remains 512-256-128-2.
- Apply tanh to the Gaussian policy mean so deterministic and sampled policies share the normalized action range.
- Gate arc-length progress reward by cross-track and heading quality; require both small remaining arc length and small Euclidean endpoint distance before applying the stopping-speed cost.
- Cap each constant-curvature local arc at 120 degrees to avoid loop-like projection ambiguity.
- Use seven stages: straight; balanced R4 arcs; R5/R4/R3 arcs; add R2.5; add R2; add moderate S-curves; add full S-curves and small initial offsets.
- Advance only when every active path type independently has at least 512 trials and at least 70% success in an 8192-episode window. Straight examples can no longer hide a failed turn direction.
- PPO learning rate is 1e-4, entropy coefficient 5e-4, initial/minimum action standard deviation 0.35/0.08. V2 trains for up to 6000 updates and is evaluated checkpoint-by-checkpoint with fixed seeds and mean actions.

### V2 stage-1 steering sign correction

- `model_400` fixed mean-action evaluation: straight 97.66%, positive-curvature R4 left arc 100%, negative-curvature R4 right arc 0%.
- A successful positive-curvature left arc used a negative joint-2 normalized command (trace mean -0.442). This directly disproved the inherited sign assumption in `steering_direction`.
- Corrected the auxiliary cost to penalize `desired_curvature * joint2_target > 0`, so the desired joint-2 target has the opposite sign to geometric path curvature. The checkpoint before correction is retained; continuation remains part of the same random-weight-from-zero chain.

### V2 reward-change optimizer reset

- The first sign-corrected continuation restored the Adam state from before the reward change. Its first PPO update had surrogate loss 0.289 and the policy collapsed to 0% in every Stage-1 type for repeated 8192-episode windows.
- Stopped that branch. Restart the same verified `model_500` policy and curriculum with `PATH_LOAD_OPTIMIZER=0`, which discards stale Adam moments and uses the configured 1e-4 learning rate. No policy from outside the from-scratch V2 chain is introduced.

## V3 mirror-equivariant policy

- `model_700` Stage-2 deterministic evaluation exposed repeated one-sided collapse: R3 left 100%, R3 right 9.38%, straight 4.69%. Continuing the same unconstrained actor was rejected.
- Actor frames now contain 18 yaw-invariant values: local nearest-path pose (4), projected gravity (3), body linear velocity (3), body angular velocity (3), joint-2 position (1), both joint velocities (2), and previous action (2). Global base quaternion/yaw is removed.
- A lateral reflection flips path lateral/heading-sine features, gravity-y, linear-y, angular-x/z, joint-2 position/velocity, and action 2. The policy mean is the average of the direct prediction and the reflected prediction mapped back to action space.
- This makes joint-1 command even and joint-2 command odd under left/right reflection. A centered straight state cannot acquire a fixed steering bias, and learning one turn direction supplies the mirrored action for the other.
- V3 actor observation is 20 x 18 history plus 43 current path/goal values = 403. Temporal CNN latent is 32; latest five frames are 90; MLP input is 90+32+43=165. V3 starts from random weights because the observation representation changed.

### V3 GPU placement

- The first full V3 launch on physical GPU0 hit OOM after one update. V3 itself used about 1.2 GiB in PyTorch plus PhysX allocations, while another user's process occupied 4.5 GiB on the same 10 GiB card. No other-user process was stopped.
- Moved V3 training to otherwise idle physical GPU1. Evaluation uses GPU2/GPU3 and, only when memory permits, GPU0.

## V3b soft mirror symmetry

- Hard-equivariant V3 remained at 0% Stage-0 success through update 170, while the otherwise comparable V2 had already learned straight completion. This indicates that exact lateral reflection is not a valid symmetry of the internal mechanism/servo.
- Keep the yaw-invariant 18-D state, but decompose direct and mirrored network predictions into equivariant and non-equivariant residual components. Retain 25% of joint-1 residual and 50% of joint-2 residual.
- This permits the non-zero steering bias observed during successful straight travel while attenuating one-sided policy collapse. V3b starts from random weights on physical GPU1.

## V4 path-direction mixture of experts

- Soft-symmetry V3b remained at 0% Stage-0 success through update 184. The mechanical asymmetry is state dependent enough that even a 50% steering residual constraint blocks the learned compensation.
- Restore the verified V2 19-D history representation and unconstrained actor mapping. Split the actor into independent `straight`, `left`, and `right` MLP heads, routed by the first 0.6 m of body-frame path preview.
- Duplicate every V2 `model_500` actor tensor into all three heads. The converted checkpoint must reproduce the source policy exactly before training. Freeze the already-trained temporal history CNN; each sample updates only its routed expert head.
- Continue with fresh Adam, learning rate 5e-5, PPO clip 0.1, two learning epochs, entropy 2e-4, and checkpoints every 50 updates. This is a deterministic architecture-growth step in the same from-scratch chain.

## V4b normalized steering-sign constraint

- `model_850` deterministic results: straight 100%, R4 left 100%, R4 right 68.75%; its right-arc trace again used a negative mean joint-2 action. Experts prevented cross-task erasure but the right expert itself entered the wrong-sign optimum.
- In V4 the sign loss used `curvature * target` and then squared it, reducing a curvature-0.25 error by 16x. V4b uses `sign(curvature) * target`, weight -1.0.
- Select `model_600`, whose fixed gates were 128/128 for straight, R4 left, and R4 right. Set policy standard deviation to 0.05, learning rate to 2e-5, promote curriculum state to Stage 2, and restart with fresh Adam.

## V4c targeted right-R3 specialist stage

- `model_650` deterministic gates: straight 92.97%, R3 left 100%, R3 right 4.69%. In the mixed Stage-2 distribution a specific right-R3 episode is only about 12% of samples.
- Add training-only `PATH_TRAIN_TYPE` and `PATH_TRAIN_CURVATURE` selectors. Disable automatic curriculum while a selector is active.
- Add `PATH_TRAIN_EXPERT`; freeze all actor heads except the named expert. History encoder remains frozen. Start from verified V4 `model_600`, train only the right expert on curvature -1/3 m^-1, keep action std 0.05 and fresh Adam.

## V4d right-R3 feedforward exploration prior

- The V4c right-R3 specialist did not escape the wrong-sign local optimum; true R3 completion fell to about 1-2%. A one-sided sign cost can prefer zero command and does not supply useful magnitude.
- Add a nominal joint-2 target `clip(-2.5 * curvature, -0.9, 0.9)` only while cross-track <=0.25 m, heading error <=0.35 rad, and remaining length >0.5 m. Cost weight is -0.30; normalized wrong-sign cost is reduced to -0.50.
- Restart from verified `model_600`, not the failed V4c checkpoints. Path tracking/progress rewards remain authoritative outside the local nominal region.

## V4d diagnosis and open-loop identification

- V4d finished at update 900. The displayed `path_success_rate=1.0` was stale checkpoint curriculum state: fixed-path runs disabled `_update_path_curriculum`, so the logger never recomputed the rate. The environment now always accumulates terminal success statistics; only stage advancement is disabled for specialist runs.
- Fixed mean-action right-R3 checkpoint scan (128 episodes each): model 650 9.38%, 675 9.38%, 700 5.47%, 725 4.69%, 750 4.69%, 775 12.50%. V4d model 775 with sampled actions reached 17.97%, so mean-vs-sample behavior is a secondary effect rather than the main failure.
- The same V4d model 775 retained 256/256 success on right-R4, while right-R3 was 12.5-18.0%. The curvature transition from 0.25 to 0.333 1/m is the actual failure boundary.
- Open-loop identification swept 42 normalized joint-target pairs with six repeats each. The generated curvature depends strongly and non-monotonically on joint 1. For positive joint-1 target 0.65, joint-2 targets 0.3/0.6/0.9 generated curvatures -0.207/-0.414/-0.626 1/m. At joint-1 target 0.90 the same targets generated only -0.078/-0.182/-0.301 1/m. Therefore the V4d prior `a2=-2.5*kappa`, which ignored drive state, was physically incorrect.
- A deterministic geometry controller based on the measured map reached 56.25% right-R3 success in its first 24-parameter scan; the best setting used drive 0.65, stop distance 1.6 m, cross-track gain 0.6, and heading gain 1.5. This establishes that R3 is controllable in the current simulator and provides a better starting distribution than pure PPO.

## V5 measured-prior residual PPO

- V5 starts from random neural weights. No V1-V4 policy checkpoint is loaded.
- The environment adds a bounded neural residual to the two direct joint commands produced by the measured geometric prior. The prior uses path curvature, cross-track and heading errors, endpoint distance, and body-forward speed; PPO remains responsible for compensating nonlinear transients and stopping error.
- Add explicit lookahead curvature to the current path observation (44 path values total) and route the straight/left/right experts by that desired curvature. Tracking error can no longer switch the expert accidentally.
- Zero-initialize every actor output layer, so the initial deterministic policy exactly equals the verified analytic controller. Train the temporal encoder and all experts jointly.
- Start at curriculum Stage 2 with zero policy weights, action standard deviation 0.15 (minimum 0.03), learning rate 1e-4, PPO clip 0.2, five learning epochs, and 2048 environments. Add a penalty for motion opposite the local path tangent. The existing hard joint limits, target-rate limits, filters, servo, and contact yaw damping remain unchanged.

### V5 source-path correction

- The first V5 launch was stopped after 14 updates because its log still showed the old 43-D path input, frozen history encoder, 0.35 standard deviation, and Stage 0. Task registration imports `legged_gym/envs/rotunbot/path_tracking`; the new environment files had been copied to an unused duplicate directory. No checkpoint from this invalid launch is used.
- V5b installs the same reviewed sources into the registered directory and must prove 44-D path input, trainable history, 0.15 standard deviation, and Stage 2 in the startup log before its results are accepted.


### V5b early evaluation and specialist stabilization

- Fixed mean-action model 0: straight 100%, left R3 53.91%, right R3 55.47% (128 episodes per case). Model 25 improved to straight 100%, left R3 67.97%, right R3 60.94%.
- A 64-episode checkpoint scan found model 50 at straight 100%, left R3 68.75%, right R3 64.06%. Continued joint training regressed: model 75 left/right 48.44%/31.25%; model 100 left/right 54.69%/4.69%.
- Stop the joint run and keep model 50. V5c resumes it with fresh Adam, freezes the temporal encoder plus straight/left experts, trains only the right expert on R3, lowers LR to 2e-5, PPO epochs to 2, clip to 0.1, and std to 0.03-0.08. The same procedure will be applied to the left expert after selecting the best right checkpoint.


## Local port and evaluation hardening (local RTX 4070)

- The project runs on the local machine. Four environment blockers were fixed without modifying the shared venv: numpy >= 1.24 removed `np.float` which IsaacGym's `torch_utils.py` evaluates at import time; `ninja` was absent from PATH; setuptools 75 hides `distutils.version` from torch 1.10's tensorboard shim; and the CUDA 11.3 nvrtc has no `sm_89` target for IsaacGym's TorchScript helpers. See LOCAL_SETUP.md.
- Throughput at 2048 envs: 1.13-1.23 s per iteration, 173k steps/s, peak 6.9 GB of 8.2 GB.
- Evaluations now record the per-episode path curvature, a (path type x curvature) breakdown, and failure reasons. `PATH_LAYOUT_SPLIT=test` draws curvature from a held-out set {0.15, 0.30, 0.45} disjoint from every stage list.

### The headline success rate is not a stable statistic

- V6 `model_0` at stage 6, n=256, scored 96.5% in the archive. Re-running the identical configuration (seed 8206) reproduces 96.9%, so the code path is faithful.
- The same checkpoint at n=512, seed 4206 scores 81.6%. The difference is the sampled curvature mixture: the prior scores 32% at k=0.25 and 100% at k=0.40, and the two seeds drew different proportions of gentle versus sharp arcs.
- Report the (path type x curvature) rows; the overall number moves about 15 points on seed alone.

### Prior versus learned residual

- V6 `model_0` is the zero-initialised actor, i.e. the analytic prior with no learned contribution. At n=512, seed 4206: prior 81.6% train / 71.7% test; `model_50` 81.8% / 71.9%; `model_100` 82.8% / 73.2%; `model_125` 82.6% / 71.7%.
- The learned residual contributes nothing measurable over 125 updates and does not degrade either. The earlier apparent decline (96.5 -> 94.1 -> 91.4 across three n=256 checkpoints) was sampling noise.
- Held-out curvature costs a consistent ~10 points (81.6 -> 71.7): the first measured generalisation number the project has.

### Failure mode is an endgame stall, not a deviation

- On stage 6 the analytic prior's failures are essentially all `timeout`. Episode records show the ball stopping at a median of 0.43-0.52 m from the last path sample with terminal speed about 0.00 m/s, then standing still for the rest of the 40 s budget. Cross-track error on successful episodes is 0.03-0.07 m, so path following itself is accurate.
- Every success is also threshold-marginal: median endpoint 0.135-0.197 m against the 0.20 m limit, and median terminal speed 0.095-0.099 m/s against the 0.10 m/s limit.
- Path length is a strong confound. `max_arc_steps = 120 deg / (|k| * ds)` caps sharp arcs, so k=0.40 never exceeds 5.2 m and k=0.50 never exceeds 4.2 m, while k=0.20-0.25 can reach 8 m. Restricting to 4.0-5.0 m raises left k=0.25 from 32% to 60% and right k=0.25 from 46% to 75%. "Gentle curvature fails" is substantially "long paths fail".
- The deceleration ramp was the obvious suspect (`desired = cruise * clamp(d/stop, 0, 1)`, stop=0.90 below |k|=0.38 and 2.40 above). Sweeping it refutes the hypothesis: 0.45 gives 81.8%, 0.90 gives 81.6%, 2.40 gives 74.8% on train. A longer ramp makes the ball slow down earlier and stall further short.

## Analytic-prior search tooling

- Added `legged_gym/scripts/sweep_path_prior.py` (fixed schedule) and `search_path_prior.py` (budgeted coordinate descent), both driven through `run_local_sweep.sh`. See LOCAL_SETUP.md.
- The prior is evaluated without any checkpoint: `PATH_ZERO_INIT_ACTOR=1` zeroes weight and bias of every expert's output layer at construction, so the action equals the prior exactly. The harness asserts `max|actor output| == 0` at startup rather than assuming it.
- The environment is built once and reused, and the RNG is reseeded before every rollout, so each configuration is compared on byte-identical paths. This is what makes 512-2048 episodes enough to resolve a point or two; unpaired evaluation moves the headline number by ~15 points on the mixture alone.
- Throughput measured on the RTX 4070: 12.6 s per configuration at 512 environments, 15.5 s at 2048. Reseeding and env reuse cut roughly 40 s of IsaacGym start-up per configuration, so a night affords on the order of 1500-1600 evaluations.

### Objective resolution matters more than it looks

- Scoring by the single worst (path type x curvature) bucket is unstable at 512 environments: buckets hold 6-40 episodes, so the score quantises into steps of 2-17 points. The same configuration ranked worse or better depending only on the `min_bucket_n` floor, which is a property of the metric, not the controller.
- At 2048 environments every bucket holds at least 55 episodes and the smoothed score (mean of the 3 worst buckets) is stable; this is the configuration to search with.
- Worst cases at stage 6 with defaults, n>=55 per bucket: s_curve at |k|=0.40 scores 23.1-25.9%, right_arc at |k|=0.20 37.4%, left_arc at |k|=0.20-0.25 39.3%. S-curves are not capped by `max_angle`, so the worst cases are also the long ones, consistent with the endgame stall.

### Screen of single-parameter changes (512 envs, 31 configurations)

- No single-parameter perturbation improved the worst bucket over the defaults; the defaults sit at a local optimum in these 15 scalars.
- Several changes are clearly harmful: `prior_normal_drive` 0.25 -> 0.15 costs 18 points overall and zeroes the worst bucket; `prior_gain_slope` 0.92 -> 1.12 costs 8 points; `prior_r2_curvature` 0.45 -> 0.40 costs 7.
- A first 3-minute coordinate-descent run accepted `prior_normal_heading_kp` 1.5 -> 2.0 and `prior_tight_cross_track_kp` 0.1 -> 0.2, moving overall 71.5% -> 79.5% on a fixed path set.
