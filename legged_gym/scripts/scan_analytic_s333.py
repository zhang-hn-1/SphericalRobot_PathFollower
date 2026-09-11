"""Tune preview and feedback for S-curves with peak curvature 1/3 m^-1."""
import json,sys
from collections import defaultdict
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
import isaacgym  # noqa:F401
import numpy as np,torch
from legged_gym.envs import *  # noqa:F401,F403
from legged_gym.utils import get_args,task_registry


def main():
 args=get_args();torch.manual_seed(8401);np.random.seed(8401);configs=[]
 for drive in (.20,.25,.30,.35):
  for stop in (1.0,1.6):
   for kc in (.05,.15):
    for kh in (.5,1.0):
     for look in (.4,.8,1.2):configs.append((drive,stop,kc,kh,look))
 assignment=[c for c in configs for _ in range(2)];cfg,_=task_registry.get_cfgs(name='rotunbot_path');cfg.env.num_envs=len(assignment);cfg.noise.add_noise=False;cfg.domain_rand.push_robots=False;cfg.path.curriculum_enabled=False;cfg.seed=8401;args.seed=8401
 env,_=task_registry.make_env(name='rotunbot_path',args=args,env_cfg=cfg);env.path_curriculum_stage=5;env.forced_path_type=3;env.forced_curvature=1/3;p=torch.tensor(assignment,dtype=torch.float,device=env.device);drive,stop,kc,kh,look=[p[:,i] for i in range(5)];ids=torch.arange(env.num_envs,device=env.device);rows=[];lengths=(4.0,6.0,8.0);batch=torch.arange(env.num_envs,device=env.device)
 for length in lengths:
  env.cfg.path.length_range_stage5=[length,length];env.reset_idx(ids);env.compute_observations();active=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
  with torch.no_grad():
   for step in range(int(env.max_episode_length)+5):
    cruise=1.18*drive;desired=cruise*torch.clamp(env.path_endpoint_distance/stop,0,1);a1=desired/1.18+.65*(desired-env.base_lin_vel[:,0]);look_steps=torch.round(look/float(env.cfg.path.sample_spacing)).long();index=torch.minimum(env.path_index+look_steps,env.path_last_index);kappa=env.path_curvature[batch,index];cmd=kappa-kc*env.path_cross_track-kh*env.path_heading_error;gain=torch.clamp(.092+.92*torch.abs(a1),.18,.75);a2=-cmd/gain
    _,_,_,dones,_=env.step(torch.stack((a1,a2),1).clamp(-1,1));done=torch.nonzero(dones&active,as_tuple=False).flatten()
    for i in done.tolist():rows.append({'length':length,'config':assignment[i],'success':int(env.terminal_success[i]),'reason':int(env.terminal_reason[i])})
    active[done]=False
    if not bool(active.any()):break
 b=defaultdict(list)
 for r in rows:b[tuple(r['config'])].append(r)
 out=[]
 for c in configs:
  rs=b[c];by={str(l):float(np.mean([r['success'] for r in rs if r['length']==l])) for l in lengths};out.append({'drive':c[0],'stop_distance':c[1],'k_cross':c[2],'k_heading':c[3],'lookahead_m':c[4],'overall':float(np.mean([r['success'] for r in rs])),'minimum':min(by.values()),'by_length':by})
 out.sort(key=lambda r:(r['minimum'],r['overall']),reverse=True);dest=Path('/data/lzq/workspace/SphericalRobot_PathFollower_20260911/artifacts/path_follower/analytic_controller/s333_scan');dest.mkdir(parents=True,exist_ok=True);(dest/'summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out[:20],indent=2))


if __name__=='__main__':main()
