"""Bounded direct-motor visual PPO with intact environment sequences and explicit reset boundaries.

The v1 depth profile uses explicit parameter groups and a twenty-update body warmup; head and raw-depth encoder learn from the start.
Finite 180 s episodes are terminal, including timeouts; critic includes a clock.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch


def depth_ppo_config():
    return dict(rollout_steps=160, minibatches=8, epochs=5, gamma=.9999, lam=.995,
                clip=.2, value_loss_coef=1., entropy_coef=.001,
                learning_rates=dict(actor_body=1e-4,head=3e-4,visual=3e-4,critic=3e-4,std=1e-4),
                schedule='constant',body_warmup_updates=20,max_grad_norm=1.,save_interval=25,
                finite_horizon=True, timeout_bootstrap=False, target_kl=.025, input_mode='depth', initialization_mode='bounded_residual')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(str(temp), str(path))


def source_contract():
    root = Path(__file__).resolve().parents[1]
    paths = list((root/'motor_depth_rl').rglob('*.py'))
    paths += [root/'motor_sru/camera_pose.py',root/'motor_sru/smoke_checks.py']
    paths += [root/name for name in ('dwl/actor_critic_dwl.py', 'scripts/train_motor_depth_rl.py',
        'envs/base/base_task.py', 'envs/base/legged_robot.py',
        'envs/base/legged_robot_config.py', 'envs/rotunbot/target_point/rotunbot_target.py',
        'envs/rotunbot/target_point/rotunbot_target_config.py',
        'envs/rotunbot/target_point/rotunbot_target_lh.py',
        'envs/rotunbot/target_point/rotunbot_target_lh_config.py',
        'envs/rotunbot/target_point/rotunbot_target_repro.py',
        'envs/rotunbot/target_point/rotunbot_target_repro_config.py')]
    return {str(path.relative_to(root)).replace('\\','/'):sha256(path) for path in sorted(paths)}


def rng_state(device):
    device = torch.device(device)
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)


def restore_rng(state, device):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if torch.device(device).type=='cuda':
        if state['torch_cuda'] is None:
            raise ValueError('Checkpoint lacks the selected CUDA generator state')
        torch.cuda.set_rng_state(state['torch_cuda'].cpu(), device)


def clone_state(state):
    return tuple(t.detach().clone() for t in state)


def gae(rewards, values, dones, last_value, gamma, lam):
    """No bootstrap across ANY done, including the finite-horizon timeout."""
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(last_value)
    for step in reversed(range(len(rewards))):
        next_value = last_value if step==len(rewards)-1 else values[step+1]
        alive = (~dones[step]).to(values.dtype)
        delta = rewards[step]+gamma*next_value*alive-values[step]
        carry = delta+gamma*lam*alive*carry
        advantages[step] = carry
    return advantages, advantages+values


class DepthPPORunner:
    def __init__(self, env, policy, baseline_metadata, config=None, output=None, device='cpu'):
        self.env, self.policy, self.device = env, policy.to(device), torch.device(device)
        if (policy.hidden_size,policy.num_actor_obs,policy.num_critic_obs,policy.num_actions)!=(128,2943,188,2):
            raise ValueError('This v1 Motor Depth profile fixes H128, actor2943, critic188, actions2')
        self.config = depth_ppo_config() if config is None else copy.deepcopy(config)
        if not self.config['finite_horizon'] or self.config['timeout_bootstrap']:
            raise ValueError('This profile requires terminal 180-second timeouts')
        if self.config['schedule']!='constant' or self.config['body_warmup_updates']<0:
            raise ValueError('depth v1 requires constant group rates and explicit nonnegative warmup')
        if policy.input_mode!=self.config['input_mode'] or policy.initialization_mode!=self.config['initialization_mode']:
            raise ValueError('Policy input/initialization mode differs from algorithm contract')
        self.baseline_metadata = canonical(baseline_metadata)
        if not self.baseline_metadata.get('checkpoint_sha256'):
            raise ValueError('Explicit baseline weight provenance is required')
        if not self.baseline_metadata.get('exploration_override'):
            raise ValueError('depth v1 requires explicit post-baseline exploration initialization provenance')
        self.contract = canonical(dict(schema='motor_depth_rl_single_ppo_v1', environment=env.get_training_contract(),
            baseline=self.baseline_metadata, algorithm=self.config, sources=source_contract(),
            actor_obs=2943, critic_obs=188, critic_remaining_time_column=187, actions=2, recurrent_width=128,
            exploration_override=self.baseline_metadata['exploration_override'],
            memory_update='fresh_depth_only; starts clear before step', distribution='tanh Normal normalized actions, stored pre-tanh samples, exact Jacobian log probability',
            minibatches='shuffle environment IDs only, retain all rollout time steps',
            restoration='weights+Adam+RNG+iteration restored; physical environment and recurrent/visual runtime reset'))
        self.iteration, self.total_steps = 0, 0
        self.optimizer = self._make_optimizer()
        self._apply_group_rates()
        self.output = None if output is None else Path(output)
        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=True)
        self.stop_requested = False
        self.last_rollout = None
        self.at_update_boundary = True
        self.reset_runtime()

    def _make_optimizer(self):
        groups=self.policy.parameter_groups()
        if set(groups)!=set(self.config['learning_rates']) or any(not values for values in groups.values()):
            raise ValueError('All named parameter groups must be present')
        grouped=[p for values in groups.values() for p in values]
        if len({id(p) for p in grouped})!=len(grouped) or {id(p) for p in grouped}!={id(p) for p in self.policy.parameters()}:
            raise ValueError('Optimizer groups omit or duplicate a parameter')
        return torch.optim.Adam([dict(name=name,params=parameters,lr=self.config['learning_rates'][name])
                                 for name,parameters in groups.items()])

    def warmup_stage(self,completed_updates=None):
        completed=self.iteration if completed_updates is None else completed_updates
        return 'visual_warmup' if completed<self.config['body_warmup_updates'] else 'joint_finetune'

    def _expected_rates(self,completed_updates):
        rates=dict(self.config['learning_rates'])
        if self.warmup_stage(completed_updates)=='visual_warmup':
            rates['actor_body']=0.
        return rates

    def group_rates(self):
        return {group['name']:float(group['lr']) for group in self.optimizer.param_groups}

    def _apply_group_rates(self):
        rates=self._expected_rates(self.iteration)
        for group in self.optimizer.param_groups:
            group['lr']=rates[group['name']]

    def reset_runtime(self):
        self.obs, self.critic = self.env.reset()
        self._validate_obs(self.obs, self.critic)
        self.state = self.policy.initial_state(self.env.num_envs, self.device)
        self.starts = torch.ones(self.env.num_envs, device=self.device, dtype=torch.bool)
        self.episode_returns = torch.zeros(self.env.num_envs, device=self.device)
        self.episode_steps = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.long)
        self.last_rollout = None
        self.at_update_boundary = True

    def _validate_obs(self, obs, critic):
        if tuple(obs.shape)!=(self.env.num_envs,2943) or tuple(critic.shape)!=(self.env.num_envs,188):
            raise ValueError('Motor Depth observation contract differs')
        if not torch.isfinite(obs).all() or not torch.isfinite(critic).all():
            raise ValueError('Nonfinite motor observations')

    def collect(self):
        if not self.at_update_boundary:
            raise RuntimeError('A rollout is already pending optimization')
        self.at_update_boundary = False
        self.policy.train()
        rows = {key:[] for key in ('obs','critic','starts','mean','std','actions','pre_tanh','log_prob','values','rewards','dones')}
        initial_state = clone_state(self.state)
        terminals = []
        telemetry = {key:[] for key in ('radius_reference_exceeded','motor_target_limit_fraction','curvature')}
        with torch.no_grad():
            for _ in range(self.config['rollout_steps']):
                mean, value, next_state = self.policy.step(self.obs,self.critic,self.state,self.starts)
                distribution = self.policy.distribution(mean)
                actions, pre_tanh = distribution.sample_with_pre_tanh()
                old_obs, old_critic, old_starts = self.obs.clone(), self.critic.clone(), self.starts.clone()
                following, critic, reward, done, info = self.env.step(actions.clone())
                self._validate_obs(following, critic)
                reward, done = reward.reshape(-1), done.reshape(-1).bool()
                if not torch.isfinite(reward).all() or len(done)!=self.env.num_envs:
                    raise ValueError('Invalid reward or done vector')
                packed = dict(obs=old_obs,critic=old_critic,starts=old_starts,mean=mean,std=distribution.base_scale,
                    actions=actions,pre_tanh=pre_tanh,log_prob=distribution.log_prob(actions,pre_tanh_value=pre_tanh).sum(-1),values=value.reshape(-1),rewards=reward,dones=done)
                for key,item in packed.items():
                    rows[key].append(item.detach().clone())
                self.episode_returns += reward
                self.episode_steps += 1
                nav = info.get('motor_navigation',{})
                for index in done.nonzero(as_tuple=True)[0].tolist():
                    event = dict(return_value=float(self.episode_returns[index]),steps=int(self.episode_steps[index]))
                    for key in ('success','collision','out_of_bounds','unstable','timeout','goal_distance','task_index','category_id','stage_id'):
                        if key in nav:
                            event[key] = nav[key][index].item()
                    terminals.append(event)
                if 'radius_reference_exceeded' in nav:
                    telemetry['radius_reference_exceeded'].append(nav['radius_reference_exceeded'].float().mean().item())
                if 'motor_targets' in nav and hasattr(self.env,'cfg'):
                    control = self.env.cfg.control
                    limits = torch.tensor([control.first_vel_limits,control.second_pos_limits],device=self.device)
                    telemetry['motor_target_limit_fraction'].append((nav['motor_targets'].abs()>=limits-1e-6).float().mean(0).tolist())
                if 'actual_v_world_xy' in nav and 'actual_yaw_rate' in nav:
                    speed = torch.linalg.vector_norm(nav['actual_v_world_xy'],dim=-1)
                    valid = speed>.03
                    if valid.any():
                        telemetry['curvature'].extend((nav['actual_yaw_rate'][valid].abs()/speed[valid]).cpu().tolist())
                self.episode_returns[done] = 0.; self.episode_steps[done] = 0
                self.obs, self.critic = following, critic
                self.state, self.starts = clone_state(next_state), done.clone()
            # Critic is non-recurrent; bootstrapping cannot advance actor memory.
            last_value = self.policy.value(self.critic).reshape(-1).detach()
        rollout = {key:torch.stack(value) for key,value in rows.items()}
        rollout['initial_state'] = initial_state
        rollout['advantages'],rollout['returns'] = gae(rollout['rewards'],rollout['values'],rollout['dones'],
            last_value,self.config['gamma'],self.config['lam'])
        rollout['terminals'],rollout['telemetry'] = terminals,telemetry
        self.last_rollout = rollout
        return rollout

    def update(self, rollout):
        if self.at_update_boundary or rollout is not self.last_rollout:
            raise RuntimeError('Optimize exactly the pending collected rollout')
        cfg = self.config
        self._apply_group_rates()
        used_stage,used_rates=self.warmup_stage(),self.group_rates()
        advantages = rollout['advantages']
        advantages = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
        count = self.env.num_envs
        batches = max(n for n in range(1,min(count,cfg['minibatches'])+1) if count%n==0)
        losses, gradient_norms, kls = [],[],[]
        early_stopped=False
        for _ in range(cfg['epochs']):
            if early_stopped: break
            permutation = torch.randperm(count,device=self.device)
            for indices in permutation.chunk(batches):
                state = tuple(t[:,indices,:] for t in rollout['initial_state'])
                mean,value,_ = self.policy.sequence(rollout['obs'][:,indices],rollout['critic'][:,indices],state,rollout['starts'][:,indices])
                distribution = self.policy.distribution(mean)
                log_prob = distribution.log_prob(rollout['actions'][:,indices],pre_tanh_value=rollout['pre_tanh'][:,indices]).sum(-1)
                ratio = (log_prob-rollout['log_prob'][:,indices]).exp()
                advantage = advantages[:,indices]
                actor_loss = torch.maximum(-ratio*advantage,-ratio.clamp(1-cfg['clip'],1+cfg['clip'])*advantage).mean()
                value = value.squeeze(-1)
                previous = rollout['values'][:,indices]
                clipped_value = previous+(value-previous).clamp(-cfg['clip'],cfg['clip'])
                targets = rollout['returns'][:,indices]
                critic_loss = torch.maximum((value-targets).square(),(clipped_value-targets).square()).mean()
                entropy = distribution.entropy().sum(-1).mean()
                with torch.no_grad():
                    old_std, old_mean = rollout['std'][:,indices],rollout['mean'][:,indices]
                    kl = (torch.log(distribution.base_scale/old_std)+(old_std.square()+(old_mean-mean).square())/
                          (2*distribution.base_scale.square())-.5).sum(-1).mean().item()
                if kl > 1.5*cfg['target_kl'] and losses:
                    early_stopped=True
                    break
                loss=actor_loss+cfg['value_loss_coef']*critic_loss-cfg['entropy_coef']*entropy
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite PPO loss')
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(self.policy.parameters(),cfg['max_grad_norm'])
                if not torch.isfinite(norm):
                    raise RuntimeError('Nonfinite PPO gradient')
                self.optimizer.step()
                losses.append([float(actor_loss.detach()),float(critic_loss.detach()),float(entropy.detach())])
                gradient_norms.append(float(norm)); kls.append(kl)
        self.iteration+=1; self.total_steps+=cfg['rollout_steps']*count
        if self.output is not None:
            with (self.output/'training_episodes.jsonl').open('a',encoding='utf-8') as stream:
                for event in rollout['terminals']:
                    stream.write(json.dumps(dict(update=self.iteration,**event),allow_nan=False)+'\n')
        self._apply_group_rates()  # checkpoint stores rates for the next update
        self.at_update_boundary=True; self.last_rollout=None
        ended=rollout['terminals']; n=len(ended)
        flags={name:sum(bool(r.get(name,False)) for r in ended) for name in ('success','collision','out_of_bounds','unstable','timeout')}
        goal_distances=[r['goal_distance'] for r in ended if 'goal_distance' in r]
        result=dict(iteration=self.iteration,total_steps=self.total_steps,episodes=n,
            success_rate=flags['success']/n if n else None,terminal_counts=flags,
            mean_terminal_goal_distance_m=float(np.mean(goal_distances)) if goal_distances else None,
            mean_episode_return=float(np.mean([r['return_value'] for r in ended])) if n else None,
            mean_reward=float(rollout['rewards'].mean()),raw_mean=rollout['mean'].mean((0,1)).tolist(),
            normalized_abs_ge_point99_fraction=(rollout['actions'].abs()>=.99).float().mean((0,1)).tolist(),
            mean_std=rollout['std'].mean((0,1)).tolist(),warmup_stage=used_stage,group_learning_rates=used_rates,
            next_warmup_stage=self.warmup_stage(),next_group_learning_rates=self.group_rates(),
            actor_loss=float(np.mean(losses,axis=0)[0]),value_loss=float(np.mean(losses,axis=0)[1]),
            entropy=float(np.mean(losses,axis=0)[2]),mean_kl=float(np.mean(kls)),max_gradient_norm=max(gradient_norms),
            minibatches_per_epoch=batches,ppo_gradient_steps=len(losses),kl_early_stopped=early_stopped,timeout_bootstrap=False)
        result['by_category']={}
        for identifier,category in enumerate(self.env.category_names):
            selected=[row for row in ended if row.get('category_id')==identifier]
            category_counts={key:sum(bool(row.get(key,False)) for row in selected) for key in flags}
            distances=[row['goal_distance'] for row in selected if 'goal_distance' in row]
            result['by_category'][category]=dict(episodes=len(selected),terminal_counts=category_counts,
                success_rate=category_counts['success']/len(selected) if selected else None,
                mean_terminal_goal_distance_m=float(np.mean(distances)) if distances else None)
        result['by_stage']={}
        for stage in range(3):
            selected=[r for r in ended if r.get('stage_id')==stage]
            counts={key:sum(bool(r.get(key,False)) for r in selected) for key in flags}
            result['by_stage'][str(stage)]=dict(episodes=len(selected),terminal_counts=counts,success_rate=counts['success']/len(selected) if selected else None)
        for name,values in rollout['telemetry'].items():
            result[name+'_mean']=np.mean(values,axis=0).tolist() if values else None
        return result

    def save(self,path):
        if not self.at_update_boundary:
            raise RuntimeError('Checkpoint only at a completed-update boundary')
        if canonical(self.env.get_training_contract()) != self.contract['environment'] or source_contract()!=self.contract['sources']:
            raise ValueError('Environment/assets/source changed after runner construction')
        checkpoint=dict(format_version=1,kind='motor_depth_rl_single_ppo',contract=self.contract,
            model_state_dict=self.policy.state_dict(),optimizer_state_dict=self.optimizer.state_dict(),
            iteration=self.iteration,total_steps=self.total_steps,group_learning_rates=self.group_rates(),
            warmup_stage=self.warmup_stage(),rng=rng_state(self.device),
            environment_runtime_at_save=canonical(self.env.get_runtime_metadata()),
            tasks_at_save=canonical(self.env.export_task_state()),
            actor_state_at_save=clone_state(self.state),starts_at_save=self.starts.clone(),
            runtime_resume_boundary='Physical simulator and observation/visual caches cannot be restored. Restore global RNG, then reset environment, all h/c, starts, episode accumulators; rollout is empty. Environment-local sampler/runtime is recorded but starts from the new requested runtime settings; not bitwise trajectory continuation.')
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.name+'.tmp')
        torch.save(checkpoint,str(temporary)); os.replace(str(temporary),str(path))
        return str(path)

    def validate_checkpoint(self,checkpoint):
        required={'format_version','kind','contract','model_state_dict','optimizer_state_dict','iteration','total_steps','group_learning_rates','warmup_stage','rng','runtime_resume_boundary','environment_runtime_at_save','tasks_at_save'}
        if not required.issubset(checkpoint) or checkpoint['format_version']!=1 or checkpoint['kind']!='motor_depth_rl_single_ppo':
            raise ValueError('Not a complete new Motor Depth checkpoint; baseline weights cannot resume training')
        if canonical(checkpoint['contract'])!=self.contract:
            raise ValueError('Motor Depth checkpoint semantic/source/baseline/map contract mismatch')
        current=self.policy.state_dict(); incoming=checkpoint['model_state_dict']
        if current.keys()!=incoming.keys() or any(not isinstance(v,torch.Tensor) or v.shape!=current[k].shape or
                v.dtype!=current[k].dtype or not torch.isfinite(v).all() for k,v in incoming.items()):
            raise ValueError('Invalid Motor Depth parameter keys/shapes/values')
        if checkpoint['iteration']<0 or checkpoint['total_steps']<0:
            raise ValueError('Invalid checkpoint iteration')
        expected_rates=self._expected_rates(checkpoint['iteration'])
        if checkpoint['group_learning_rates']!=expected_rates or checkpoint['warmup_stage']!=self.warmup_stage(checkpoint['iteration']):
            raise ValueError('Checkpoint warmup/group rates disagree with completed iteration')
        incoming_optimizer=checkpoint['optimizer_state_dict']
        groups=incoming_optimizer.get('param_groups',[])
        if len(groups)!=len(self.optimizer.param_groups):
            raise ValueError('Adam parameter groups mismatch')
        for saved_group,live_group in zip(groups,self.optimizer.param_groups):
            if saved_group.get('name')!=live_group['name'] or saved_group.get('lr')!=expected_rates[live_group['name']]:
                raise ValueError('Adam group identity/rate mismatch')
            if len(saved_group.get('params',[]))!=len(live_group['params']):
                raise ValueError('Adam parameter count mismatch')
            for identifier,parameter in zip(saved_group['params'],live_group['params']):
                state=incoming_optimizer.get('state',{}).get(identifier,{})
                if state and not {'step','exp_avg','exp_avg_sq'}.issubset(state):
                    raise ValueError('Trained checkpoint lacks complete Adam moments')
                for key in ('exp_avg','exp_avg_sq'):
                    if key in state and (state[key].shape!=parameter.shape or not torch.isfinite(state[key]).all()):
                        raise ValueError('Invalid Adam moment: '+key)
        if not {'python','numpy','torch_cpu','torch_cuda'}.issubset(checkpoint['rng']):
            raise ValueError('Checkpoint RNG state incomplete')

    def load(self,path,evaluation=False):
        checkpoint=torch.load(str(path),map_location=self.device)
        self.validate_checkpoint(checkpoint)
        self.policy.load_state_dict(checkpoint['model_state_dict'],strict=True)
        self.iteration,self.total_steps=int(checkpoint['iteration']),int(checkpoint['total_steps'])
        if not evaluation:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            restore_rng(checkpoint['rng'],self.device)
        self._apply_group_rates()
        self.reset_runtime()
        return dict(iteration=self.iteration,evaluation=evaluation,checkpoint_sha256=sha256(path),
                    runtime_reset=True,bitwise_physical_continuation=False)

    def learn(self,iterations):
        if self.output is None:
            raise ValueError('Training requires an output directory')
        status='COMPLETE'; last_path=None
        for _ in range(int(iterations)):
            begin=time.monotonic()
            result=self.update(self.collect())
            result['elapsed_s']=time.monotonic()-begin
            with (self.output/'training_metrics.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(result,allow_nan=False)+'\n')
            write_json(self.output/'latest_metrics.json',result)
            request_path=self.output/'control_request.json'
            request=None
            if request_path.exists():
                request=json.loads(request_path.read_text(encoding='utf-8'))
                if request.get('action') not in ('pause','save','stop') or not request.get('request_id'):
                    raise ValueError('Malformed control request')
            stopping=self.stop_requested or (request is not None and request['action'] in ('pause','stop'))
            if self.iteration%self.config['save_interval']==0 or request is not None or stopping:
                last_path=self.save(self.output/'model_{}.pt'.format(self.iteration))
            if request is not None:
                write_json(self.output/'control_response.json',dict(request_id=request['request_id'],action=request['action'],
                    status='SAVED_AT_UPDATE_BOUNDARY',iteration=self.iteration,checkpoint=last_path))
                request_path.unlink()
            if stopping:
                status='PAUSED'; break
        last_path=self.save(self.output/'model_{}.pt'.format(self.iteration))
        return dict(status=status,iteration=self.iteration,checkpoint=last_path,total_steps=self.total_steps)
