"""High-drive/feedforward scan for 6--8 m, peak-0.5 S-curves."""
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
 args=get_args();torch.manual_seed(8601);np.random.seed(8601);configs=[]
 for drive in (.45,.55,.65):
  for stop in (1.2,2.0):
   for kc in (0.,.05):
    for kh in (0.,.5):
     for look in (.4,.8,1.2,1.6,2.0):configs.append((drive,stop,kc,kh,look))
 assignment=[c for c in configs for _ in range(2)];cfg,_=task_registry.get_cfgs(name='rotunbot_path');cfg.env.num_envs=len(assignment);cfg.noise.add_noise=False;cfg.domain_rand.push_robots=False;cfg.path.curriculum_enabled=False;cfg.seed=8601;args.seed=8601
 env,_=task_registry.make_env(name='rotunbot_path',args=args,env_cfg=cfg);env.path_curriculum_stage=6;env.forced_path_type=3;env.forced_curvature=.5;p=torch.tensor(assignment,dtype=torch.float,device=env.device);drive,stop,kc,kh,look=[p[:,i] for i in range(5)];ids=torch.arange(env.num_envs,device=env.device);batch=torch.arange(env.num_envs,device=env.device);rows=[];lengths=(6.,7.,8.)
 for length in lengths:
  env.cfg.path.length_range_stage6=[length,length];env.reset_idx(ids);env.compute_observations();active=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
  with torch.no_grad():
   for step in range(int(env.max_episode_length)+5):
    cruise=1.18*drive;desired=cruise*torch.clamp(env.path_endpoint_distance/stop,0,1);a1=desired/1.18+.65*(desired-env.base_lin_vel[:,0]);idx=torch.minimum(env.path_index+torch.round(look/.05).long(),env.path_last_index);kappa=env.path_curvature[batch,idx];cmd=kappa-kc*env.path_cross_track-kh*env.path_heading_error;gain=torch.clamp(.092+.92*torch.abs(a1),.18,.75);a2=-cmd/gain
    _,_,_,dones,_=env.step(torch.stack((a1,a2),1).clamp(-1,1));done=torch.nonzero(dones&active,as_tuple=False).flatten()
    for i in done.tolist():rows.append({'length':length,'config':assignment[i],'success':int(env.terminal_success[i]),'reason':int(env.terminal_reason[i])})
    active[done]=False
    if not bool(active.any()):break
 b=defaultdict(list)
 for r in rows:b[tuple(r['config'])].append(r)
 out=[]
 for c in configs:
  rs=b[c];by={str(l):float(np.mean([r['success'] for r in rs if r['length']==l])) for l in lengths};out.append({'drive':c[0],'stop':c[1],'kc':c[2],'kh':c[3],'look':c[4],'overall':float(np.mean([r['success'] for r in rs])),'minimum':min(by.values()),'by_length':by})
 out.sort(key=lambda r:(r['minimum'],r['overall']),reverse=True);dest=Path('/data/lzq/workspace/SphericalRobot_PathFollower_20260911/artifacts/path_follower/analytic_controller/s05_long_scan');dest.mkdir(parents=True,exist_ok=True);(dest/'summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out[:25],indent=2))


if __name__=='__main__':main()
