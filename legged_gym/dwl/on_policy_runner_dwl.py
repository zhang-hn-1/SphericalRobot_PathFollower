
import time
import os
import re
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from legged_gym.dwl.ppo_dwl import PPODWL
from legged_gym.dwl.actor_critic_dwl import ActorCriticDWL
from legged_gym.dwl.actor_critic_path_dwl import ActorCriticPathDWL
from rsl_rl.env import VecEnv


class DWLOnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCriticDWL
        actor_critic: ActorCriticDWL = actor_critic_class(
            self.env.num_short_obs, self.env.num_single_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"]) # PPODWL
        self.alg: PPODWL = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(
                    os.path.join(self.log_dir, 'model_{}.pt'.format(it)),
                    iteration=it + 1,
                )
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            episode_keys = set()
            for ep_info in locs['ep_infos']:
                episode_keys.update(ep_info.keys())
            for key in sorted(episode_keys):
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    if key not in ep_info:
                        continue
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        curriculum_string = ''
        get_curriculum_status = getattr(
            self.env, 'get_tracking_curriculum_status', None
        )
        curriculum_status = (
            get_curriculum_status() if callable(get_curriculum_status) else None
        )
        if curriculum_status:
            stage = int(curriculum_status['stage'])
            max_stage = int(curriculum_status['max_stage'])
            stage_name = curriculum_status['stage_name']
            batch_pass_rate = float(curriculum_status['batch_pass_rate'])
            ema_pass_rate = float(curriculum_status['ema_pass_rate'])
            updates = int(curriculum_status['updates'])
            min_updates = int(curriculum_status['min_updates'])
            pass_threshold = float(curriculum_status['pass_threshold'])
            detail_ema = curriculum_status.get('detail_ema', {})
            self.writer.add_scalar('Curriculum/stage', stage, locs['it'])
            self.writer.add_scalar(
                'Curriculum/batch_pass_rate', batch_pass_rate, locs['it']
            )
            self.writer.add_scalar(
                'Curriculum/ema_pass_rate', ema_pass_rate, locs['it']
            )
            self.writer.add_scalar('Curriculum/updates', updates, locs['it'])
            for detail_name, detail_value in detail_ema.items():
                self.writer.add_scalar(
                    'Curriculum/detail_' + detail_name,
                    float(detail_value),
                    locs['it'],
                )
            curriculum_string += (
                f"{f'Tracking curriculum stage:':>{pad}} "
                f"{stage}/{max_stage} ({stage_name})\n"
                f"{f'Tracking pass rate (batch/EMA):':>{pad}} "
                f"{batch_pass_rate:.1%} / {ema_pass_rate:.1%}\n"
            )
            if detail_ema:
                curriculum_string += (
                    f"{f'Tracking channel EMA (v/w/vy):':>{pad}} "
                    f"{detail_ema['speed']:.1%} / "
                    f"{detail_ema['turn']:.1%} / "
                    f"{detail_ema['lateral']:.1%}\n"
                    f"{f'Tracking mode EMA (stop/straight):':>{pad}} "
                    f"{detail_ema['stop']:.1%} / "
                    f"{detail_ema['straight']:.1%}\n"
                    f"{f'Tracking curve EMA (fwd/rev/L/R):':>{pad}} "
                    f"{detail_ema['forward_curve']:.1%} / "
                    f"{detail_ema['reverse_curve']:.1%} / "
                    f"{detail_ema['left']:.1%} / "
                    f"{detail_ema['right']:.1%}\n"
                )
            if stage < max_stage:
                curriculum_string += (
                    f"{f'Tracking stage progress:':>{pad}} "
                    f"{updates}/{min_updates} updates, "
                    f"required EMA {pass_threshold:.1%}\n"
                )
            else:
                curriculum_string += (
                    f"{f'Tracking stage progress:':>{pad}} "
                    f"final stage, {updates} measured updates\n"
                )

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += curriculum_string
        remaining_iterations = max(
            0,
            self.current_learning_iteration
            + locs['num_learning_iterations']
            - locs['it']
            - 1,
        )
        completed_iterations = max(
            1, locs['it'] - self.current_learning_iteration + 1
        )
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / completed_iterations * remaining_iterations:.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None, iteration=None):
        env_state = None
        get_env_state = getattr(self.env, "get_checkpoint_state", None)
        if get_env_state is not None:
            env_state = get_env_state()
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration if iteration is None else iteration,
            'infos': infos,
            'env_state': env_state,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        env_state = loaded_dict.get('env_state')
        required_env_state_version = self.cfg.get(
            'load_optimizer_env_state_version', None
        )
        compatible_env_state = isinstance(env_state, dict)
        if compatible_env_state and required_env_state_version is not None:
            compatible_env_state = (
                env_state.get('checkpoint_state_version')
                == required_env_state_version
            )
        task_continuation = (
            self.cfg.get('load_optimizer_when_env_state', False)
            and compatible_env_state
        )
        load_optimizer = load_optimizer and (
            self.cfg.get('load_optimizer', True) or task_continuation
        )
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        elif 'optimizer_state_dict' in loaded_dict:
            print(
                "Optimizer state not loaded; using configured learning rate "
                f"{self.alg.learning_rate:.2e}"
            )

        # Older intermediate checkpoints were saved before the runner's
        # current iteration was updated.  Recover the intended iteration from
        # their filename when the stored value is stale.
        loaded_iteration = int(loaded_dict.get('iter', 0))
        match = re.search(r"model_(\d+)\.pt$", str(path))
        if match:
            loaded_iteration = max(loaded_iteration, int(match.group(1)))
        self.current_learning_iteration = loaded_iteration

        set_env_state = getattr(self.env, "set_checkpoint_state", None)
        if set_env_state is not None:
            set_env_state(loaded_dict.get('env_state'))
        return loaded_dict.get('infos')

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
    
    def get_inference_critic(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.evaluate
