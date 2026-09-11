"""Isolated direct-motor SRU training and fixed-task evaluation entry point.

Isaac Gym is imported before torch. Only unmasked physical GPU 2 or 3 is allowed.
The original point-policy task and its checkpoints are never resumed here.
"""
import argparse
import json
import os
from pathlib import Path
import random
import signal
import sys
import time


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True, choices=(2, 3))
    parser.add_argument('--mode', choices=('smoke', 'train', 'eval'), default='smoke')
    parser.add_argument('--num-envs', type=int, default=8)
    parser.add_argument('--iterations', type=int, default=200, help='additional complete PPO updates')
    parser.add_argument('--seed', type=int, default=60)
    parser.add_argument('--split', choices=('train', 'val', 'dev20'), default=None)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True, help='trusted model_800 initialization only')
    parser.add_argument('--resume', type=Path, help='complete new motor SRU training checkpoint')
    parser.add_argument('--checkpoint', type=Path, help='complete new motor SRU evaluation checkpoint')
    parser.add_argument('--eval-count', type=int, default=40, help='first fixed split tasks; use 8 for dev smoke')
    args = parser.parse_args(argv)
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        parser.error('CUDA_VISIBLE_DEVICES must be UNSET, including no empty value: --gpu names physical GPU 2/3')
    if os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
        parser.error('Set CUDA_DEVICE_ORDER=PCI_BUS_ID to preserve physical GPU numbering')
    if args.num_envs < 1 or args.iterations < 1 or args.eval_count < 1:
        parser.error('environment, update and task counts must be positive')
    if not args.baseline.is_file():
        parser.error('baseline checkpoint does not exist')
    if args.mode == 'eval':
        if args.checkpoint is None or args.resume is not None or not args.checkpoint.is_file():
            parser.error('eval requires an existing --checkpoint and forbids --resume')
        args.split = args.split or 'val'
        if args.split == 'train':
            parser.error('fixed evaluation must use val or dev20')
    else:
        if args.checkpoint is not None:
            parser.error('--checkpoint is evaluation only; new-model training continuation uses --resume')
        if args.resume is not None and not args.resume.is_file():
            parser.error('resume checkpoint does not exist')
        args.split = args.split or ('dev20' if args.mode=='smoke' else 'train')
        if args.mode == 'train' and args.split != 'train':
            parser.error('training must use the train split')
    return args


def seed_selected(seed, device, torch, np):
    # Do not use torch.cuda.manual_seed_all: other physical cards are outside scope.
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    with torch.cuda.device(device):
        torch.cuda.manual_seed(seed)


def make_env(args, count, fixed_indices, gymapi):
    from legged_gym.motor_sru.config import make_motor_config
    from legged_gym.motor_sru.env import MotorPillarEnv
    from legged_gym.utils.helpers import class_to_dict, parse_sim_params
    cfg = make_motor_config(args.split, count, args.seed, fixed_indices)
    sim_args = argparse.Namespace(physics_engine=gymapi.SIM_PHYSX, device='cuda:{}'.format(args.gpu),
        use_gpu=True, use_gpu_pipeline=True, subscenes=0, num_threads=0)
    sim_params = parse_sim_params(sim_args, {'sim': class_to_dict(cfg.sim)})
    return MotorPillarEnv(cfg, sim_params, gymapi.SIM_PHYSX, 'cuda:{}'.format(args.gpu), headless=True)


def destroy_env(env):
    if getattr(env, 'viewer', None) is not None:
        env.gym.destroy_viewer(env.viewer)
    env.gym.destroy_sim(env.sim)


def build_runner(args, env, device, config=None):
    from legged_gym.motor_sru.policy import MotorSRUPolicy
    from legged_gym.motor_sru.runner import MotorPPORunner
    policy = MotorSRUPolicy()
    metadata = policy.load_baseline(args.baseline)
    metadata['exploration_override']=policy.initialize_exploration()
    return MotorPPORunner(env, policy, metadata, config=config, output=args.output, device=device)


def train(args, gymapi, torch, np):
    from legged_gym.motor_sru.runner import write_json
    fixed=list(range(args.num_envs)) if args.mode=='smoke' else None
    env = make_env(args, args.num_envs, fixed, gymapi)
    try:
        runner = build_runner(args, env, 'cuda:{}'.format(args.gpu))
        restoration = None if args.resume is None else runner.load(args.resume)
        if args.mode == 'smoke':
            from legged_gym.motor_sru.smoke_checks import run_smoke_checks
            unchanged={key:value.detach().clone() for key,value in runner.policy.state_dict().items()}
            smoke_result=run_smoke_checks(env,runner.policy)
            if not isinstance(smoke_result,dict):
                raise RuntimeError('Physical smoke must return a structured diagnostic report')
            write_json(args.output/'smoke_checks.json',smoke_result)
            if smoke_result.get('status')!='PASS':
                raise RuntimeError('Physical reset/sensor smoke checks did not pass')
            if any(not torch.equal(value,unchanged[key]) for key,value in runner.policy.state_dict().items()):
                raise RuntimeError('Smoke diagnostics modified policy parameters')
            runner.reset_runtime()
        def stop_after_update(signum, frame):
            runner.stop_requested = True
        signal.signal(signal.SIGTERM, stop_after_update)
        signal.signal(signal.SIGINT, stop_after_update)
        write_json(args.output/'run_manifest.json', dict(mode=args.mode, seed=args.seed, physical_gpu=args.gpu,
            started_at_unix=time.time(), contract=runner.contract, runtime=env.get_runtime_metadata(),
            restoration=restoration, baseline_is_initialization_only=True,
            termination='all finite 180 s episode ends, including timeout, have zero bootstrap'))
        write_json(args.output/'initial_tasks.json', env.export_task_state())
        if args.resume is None:
            runner.save(args.output/'model_0.pt')
        result = runner.learn(1 if args.mode == 'smoke' else args.iterations)
        write_json(args.output/'completion.json', result)
        print(json.dumps(result, ensure_ascii=False))
    finally:
        destroy_env(env)


def evaluate(args, gymapi, torch, np):
    from legged_gym.motor_sru.runner import canonical, sha256, write_json
    # Read only the declared training algorithm here; build_runner reconstructs
    # all current source/environment/baseline contract fields, and load checks all.
    checkpoint = torch.load(str(args.checkpoint), map_location='cpu')
    config = checkpoint.get('contract', {}).get('algorithm')
    if not isinstance(config, dict):
        raise ValueError('Evaluation requires the complete new motor checkpoint contract')
    episodes_path = args.output/'episodes.jsonl'
    if episodes_path.exists():
        raise ValueError('Refusing to append evaluation into an existing episode file')
    all_episodes, batches, selected_weights = [], [], None
    for first in range(0, args.eval_count, args.num_envs):
        indices = list(range(first, min(first+args.num_envs, args.eval_count)))
        seed_selected(args.seed, args.gpu, torch, np)
        env = make_env(args, len(indices), indices, gymapi)
        try:
            runner = build_runner(args, env, 'cuda:{}'.format(args.gpu), config)
            restoration = runner.load(args.checkpoint, evaluation=True)
            runner.policy.eval()
            initial = env.export_task_state()
            if selected_weights is None:
                selected_weights = {key: value.detach().cpu().clone() for key,value in runner.policy.state_dict().items()}
                write_json(args.output/'evaluation_manifest.json', dict(status='RUNNING', mode='fixed_original_motor_validation',
                    checkpoint_sha256=sha256(args.checkpoint), iteration=runner.iteration, seed=args.seed,
                    physical_gpu=args.gpu, split=args.split, task_indices=list(range(args.eval_count)),
                    contract=runner.contract, deterministic_action='Gaussian mean, original raw joint executor',
                    batch_seed_rule='same requested seed before each fixed batch; assignment recorded',
                    trajectory='every 50 Hz motor step from info.motor_navigation captured before autoreset',
                    timeout_bootstrap=False))
            rows = [[] for _ in indices]
            finished = [False for _ in indices]
            totals = [0. for _ in indices]
            snapshots = initial.get('tasks', initial.get('environments'))
            if not isinstance(snapshots, list) or len(snapshots) != len(indices):
                raise ValueError('export_task_state must provide one task snapshot per environment')
            obs, critic, state, starts = runner.obs, runner.critic, runner.state, runner.starts
            deadline = int(round(float(env.cfg.env.episode_length_s)/float(env.dt)))+2
            with torch.no_grad():
                for step in range(deadline):
                    input_fresh=obs[:,-1].detach().cpu().tolist()
                    mean, _, next_state = runner.policy.step(obs, critic, state, starts)
                    obs, critic, reward, done, info = env.step(mean.clone())
                    nav = info.get('motor_navigation', {})
                    required = ('task_index','success','collision','out_of_bounds','unstable','timeout','goal_distance',
                        'local_xy','yaw','episode_length','raw_actions','motor_targets','actual_v_world_xy',
                        'actual_yaw_rate','radius_reference_exceeded','minimum_clearance','depth_fresh','depth_stats',
                        'camera_world_pose','capture_camera_pose')
                    if any(key not in nav for key in required):
                        raise ValueError('Reset-preceding navigation trace is incomplete')
                    cpu = {key:value.detach().cpu().tolist() for key,value in nav.items() if isinstance(value, torch.Tensor)}
                    done_cpu = done.reshape(-1).bool().cpu().tolist()
                    rewards = reward.reshape(-1).cpu().tolist()
                    for local, index in enumerate(indices):
                        if finished[local]:
                            continue
                        sample = {key:value[local] for key,value in cpu.items()}
                        sample['input_depth_fresh']=input_fresh[local]
                        if bool(sample['depth_fresh']) != bool(input_fresh[local]):
                            raise ValueError('Reported capture belongs to a different input-frame freshness state')
                        if int(sample['task_index']) != index:
                            raise ValueError('Task identity changed before its first terminal frame')
                        sample['t_s'] = (step+1)*float(env.dt)
                        rows[local].append(sample)
                        totals[local] += rewards[local]
                        if done_cpu[local]:
                            if not any(bool(sample[key]) for key in ('success','collision','out_of_bounds','unstable','timeout')):
                                raise ValueError('Terminal episode has no reported termination reason')
                            start_distance=float(np.linalg.norm(np.asarray(snapshots[local]['start'])-np.asarray(snapshots[local]['goal'])))
                            episode = dict(task_index=index, task=snapshots[local], seed=args.seed, iteration=runner.iteration,
                                steps=len(rows[local]), duration_s=sample['t_s'], episode_return=totals[local],
                                final_goal_distance_m=float(sample['goal_distance']),
                                initial_goal_distance_m=start_distance,
                                minimum_goal_distance_m=min([start_distance]+[float(row['goal_distance']) for row in rows[local]]),
                                trajectory=rows[local], **{key:bool(sample[key]) for key in ('success','collision','out_of_bounds','unstable','timeout')})
                            all_episodes.append(episode)
                            with episodes_path.open('a',encoding='utf-8') as stream:
                                stream.write(json.dumps(episode,allow_nan=False)+'\n')
                            finished[local] = True
                    state, starts = next_state, done.reshape(-1).bool()
                    if all(finished):
                        break
            if not all(finished):
                raise RuntimeError('Fixed evaluation did not terminate every selected task by its horizon')
            current = runner.policy.state_dict()
            if any(not torch.equal(value.detach().cpu(), selected_weights[key]) for key,value in current.items()):
                raise RuntimeError('Evaluation modified selected model parameters')
            batches.append(dict(task_indices=indices, initial=initial, restoration=restoration, runtime=env.get_runtime_metadata()))
            write_json(args.output/'batches.json', batches)
        finally:
            destroy_env(env)
    if len(all_episodes) != args.eval_count or {r['task_index'] for r in all_episodes} != set(range(args.eval_count)):
        raise RuntimeError('Incomplete or duplicate fixed task results')
    counts={key:sum(bool(row[key]) for row in all_episodes) for key in ('success','collision','out_of_bounds','unstable','timeout')}
    category_results={}
    for category in ('required_detour','clear_direct'):
        selected=[row for row in all_episodes if row['task']['category']==category]
        flags={key:sum(bool(row[key]) for row in selected) for key in counts}
        category_results[category]=dict(episodes=len(selected),terminal_counts=flags,
            success_rate=flags['success']/len(selected) if selected else None,
            mean_final_goal_distance_m=float(np.mean([row['final_goal_distance_m'] for row in selected])) if selected else None)
    if args.eval_count==40 and (category_results['required_detour']['episodes']!=32 or category_results['clear_direct']['episodes']!=8):
        raise ValueError('Fixed 40-task protocol requires 32 detours plus 8 direct tasks')
    result=dict(status='PASS', meaning='execution integrity passed; success is reported separately',
        episodes=len(all_episodes), terminal_counts=counts, success_rate=counts['success']/len(all_episodes),
        mean_final_goal_distance_m=float(np.mean([r['final_goal_distance_m'] for r in all_episodes])),
        checkpoint_sha256=sha256(args.checkpoint), episodes_sha256=sha256(episodes_path), seed=args.seed,
        iteration=checkpoint['iteration'], selected_parameters_unchanged=True,by_category=category_results)
    write_json(args.output/'completion.json', result)
    manifest=json.loads((args.output/'evaluation_manifest.json').read_text(encoding='utf-8'))
    manifest['status']='PASS'
    manifest['completion_meaning']=result['meaning']
    write_json(args.output/'evaluation_manifest.json',manifest)
    print(json.dumps(result, ensure_ascii=False))


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode != 'eval' and args.resume is None and any(args.output.iterdir()):
        raise ValueError('Fresh training requires an empty output directory')
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    # Required Isaac Gym import order; no torch import occurs in CLI parsing.
    from isaacgym import gymapi
    import numpy as np
    import torch
    torch.cuda.set_device(args.gpu)
    seed_selected(args.seed, args.gpu, torch, np)
    if args.mode == 'eval':
        evaluate(args, gymapi, torch, np)
    else:
        train(args, gymapi, torch, np)


if __name__ == '__main__':
    main()
