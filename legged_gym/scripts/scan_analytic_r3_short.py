"""Find a low-speed controller for short 3.0--3.5 m R3 paths."""
import json, sys
from collections import defaultdict
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import isaacgym  # noqa: F401
import numpy as np
import torch
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


def main():
    args = get_args(); torch.manual_seed(7801); np.random.seed(7801)
    configs = []
    for drive in (0.25, 0.35, 0.45, 0.55):
        for stop in (0.6, 0.9, 1.2):
            for kc in (0.2, 0.35, 0.5):
                for kh in (1.0, 1.5):
                    configs.append((drive, stop, kc, kh))
    repeats = 4; assignment = [cfg for cfg in configs for _ in range(repeats)]
    env_cfg, _ = task_registry.get_cfgs(name="rotunbot_path")
    env_cfg.env.num_envs=len(assignment); env_cfg.noise.add_noise=False
    env_cfg.domain_rand.push_robots=False; env_cfg.path.curriculum_enabled=False
    args.seed=7801; env_cfg.seed=7801
    env, _ = task_registry.make_env(name="rotunbot_path", args=args, env_cfg=env_cfg)
    env.path_curriculum_stage=2; env.forced_path_type=2; env.forced_curvature=-1/3
    p=torch.tensor(assignment,dtype=torch.float,device=env.device)
    drive,stop,kc,kh=[p[:,i] for i in range(4)]
    ids=torch.arange(env.num_envs,device=env.device); records=[]
    for length in (3.0,3.25,3.5,3.75,4.0):
        env.cfg.path.length_range_stage2=[length,length]; env.reset_idx(ids); env.compute_observations()
        active=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
        with torch.no_grad():
            for step in range(int(env.max_episode_length)+5):
                cruise=1.18*drive
                desired=cruise*torch.clamp(env.path_endpoint_distance/stop,0,1)
                a1=desired/1.18+0.65*(desired-env.base_lin_vel[:,0])
                cmd=env._lookahead_curvature()-kc*env.path_cross_track-kh*env.path_heading_error
                gain=torch.clamp(0.092+0.92*torch.abs(a1),0.18,0.75)
                a2=-cmd/gain
                _,_,_,dones,_=env.step(torch.stack((a1,a2),1).clamp(-1,1))
                done_ids=torch.nonzero(dones&active,as_tuple=False).flatten()
                for idx in done_ids.tolist():
                    records.append({"length":length,"config":assignment[idx],"success":int(env.terminal_success[idx]),"reason":int(env.terminal_reason[idx]),"cross":float(env.terminal_cross_track[idx]),"endpoint":float(env.terminal_endpoint_distance[idx]),"speed":float(env.terminal_speed[idx])})
                active[done_ids]=False
                if not bool(active.any()): break
    buckets=defaultdict(list)
    for r in records: buckets[tuple(r["config"])].append(r)
    summary=[]
    for cfg in configs:
        rows=buckets[cfg]; by={}
        for length in (3.0,3.25,3.5,3.75,4.0):
            q=[r["success"] for r in rows if r["length"]==length]; by[str(length)]=float(np.mean(q))
        summary.append({"drive":cfg[0],"stop_distance":cfg[1],"k_cross":cfg[2],"k_heading":cfg[3],"overall":float(np.mean([r["success"] for r in rows])),"minimum":min(by.values()),"by_length":by})
    summary.sort(key=lambda r:(r["minimum"],r["overall"]),reverse=True)
    out=Path("/data/lzq/workspace/SphericalRobot_PathFollower_20260911/artifacts/path_follower/analytic_controller/r3_short_scan"); out.mkdir(parents=True,exist_ok=True)
    (out/"summary.json").write_text(json.dumps(summary,indent=2)); (out/"episodes.json").write_text(json.dumps(records,indent=2))
    print(json.dumps(summary[:20],indent=2))


if __name__ == "__main__": main()
