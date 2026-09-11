"""Fixed-seed evaluation of the V5d measured geometric action prior."""
import json, os, sys
from collections import Counter
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
import isaacgym  # noqa: F401
import numpy as np
import torch
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry

TYPES={"straight":0,"left_arc":1,"right_arc":2,"s_curve":3}
REASONS={0:"success",1:"timeout",2:"deviation",3:"unstable",4:"out_of_bounds"}


def main():
    args=get_args(); seed=int(os.environ.get("PATH_EVAL_SEED","7900")); stage=int(os.environ.get("PATH_EVAL_STAGE","2")); episodes=int(os.environ.get("PATH_EVAL_EPISODES","256"))
    torch.manual_seed(seed); np.random.seed(seed)
    env_cfg,_=task_registry.get_cfgs(name="rotunbot_path")
    env_cfg.env.num_envs=args.num_envs or episodes; env_cfg.noise.add_noise=False
    env_cfg.domain_rand.push_robots=False; env_cfg.path.curriculum_enabled=False
    args.seed=seed; env_cfg.seed=seed
    env,_=task_registry.make_env(name="rotunbot_path",args=args,env_cfg=env_cfg)
    env.path_curriculum_stage=stage
    type_name=os.environ.get("PATH_EVAL_TYPE","straight"); env.forced_path_type=TYPES[type_name]
    if os.environ.get("PATH_EVAL_CURVATURE") is not None: env.forced_curvature=float(os.environ["PATH_EVAL_CURVATURE"])
    ids=torch.arange(env.num_envs,device=env.device); env.reset_idx(ids); env.compute_observations()
    records=[]
    with torch.no_grad():
        for step in range(int(env.max_episode_length)*4):
            actions=env._analytic_action_prior(); _,_,_,dones,_=env.step(actions)
            done_ids=torch.nonzero(dones,as_tuple=False).flatten()
            for idx in done_ids.tolist():
                records.append({"success":int(env.terminal_success[idx]),"reason":REASONS[int(env.terminal_reason[idx])],"length":float(env.terminal_path_length[idx]),"cross":float(env.terminal_cross_track[idx]),"endpoint":float(env.terminal_endpoint_distance[idx]),"speed":float(env.terminal_speed[idx]),"step":step+1})
                if len(records)>=episodes: break
            if len(records)>=episodes: break
    records=records[:episodes]
    summary={"type":type_name,"curvature":os.environ.get("PATH_EVAL_CURVATURE"),"stage":stage,"episodes":len(records),"success_rate":float(np.mean([r["success"] for r in records])),"failure_reasons":dict(Counter(r["reason"] for r in records)),"cross_track_median_m":float(np.median([r["cross"] for r in records])),"endpoint_distance_median_m":float(np.median([r["endpoint"] for r in records])),"terminal_speed_median_mps":float(np.median([r["speed"] for r in records]))}
    out=Path(os.environ.get("PATH_EVAL_OUTPUT","/data/lzq/workspace/SphericalRobot_PathFollower_20260911/artifacts/path_follower/analytic_controller/v5d_piecewise")); out.mkdir(parents=True,exist_ok=True)
    tag=f"stage{stage}_{type_name}_k{os.environ.get('PATH_EVAL_CURVATURE','random').replace('.','p')}"
    (out/f"{tag}_metrics.json").write_text(json.dumps(summary,indent=2)); (out/f"{tag}_episodes.json").write_text(json.dumps(records,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
