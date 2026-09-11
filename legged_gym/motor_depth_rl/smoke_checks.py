"""Real GPU sensor, actuator, partial-reset and terminal-snapshot checks.

These assertions establish execution integrity, not successful navigation.
"""
import math
import time
import torch

from legged_gym.motor_sru.smoke_checks import (
    check_depth_image, compare_partial_reset, check_terminal_snapshot,
    _set_root_pose, _check_known_pillar_depth,
)


def snapshot(env):
    names=('actions','last_actions','output_actions','last_output_actions',
           'last_dof_vel','last_root_vel','last_base_lin_vel','last_base_ang_vel',
           'last_root_states','depth_image','capture_camera_pose')
    result={key:getattr(env,key).clone() for key in names}
    result['obs_history']=torch.stack(tuple(env.obs_history),dim=1).clone()
    result['critic_history']=torch.stack(tuple(env.critic_history),dim=1).clone()
    result['all_actor_roots']=env._all_root_states.reshape(env.num_envs,41,13).clone()
    for name in ('PID_FirstAxis','PID_SecondAxis'):
        for field in ('last_error','integral'):
            result[name+'.'+field]=getattr(getattr(env,name),field).clone()
    return result


@torch.inference_mode()
def run_smoke_checks(env,policy):
    begin=time.monotonic()
    if env.num_envs<2 or env.cfg.task_catalog.resample_on_reset:
        raise AssertionError('Smoke requires fixed >=2 diagnostic environments')
    if str(env.device) not in ('cuda:2','cuda:3'):
        raise AssertionError('Only physical GPU2/3 are authorized')
    if hasattr(env,'depth_encoder'):
        raise AssertionError('Raw-depth environment must not cache frozen features')
    camera=env.cfg.camera
    checks={}
    obs,critic=env.reset()
    assert obs.shape==(env.num_envs,2943) and critic.shape==(env.num_envs,188)
    assert torch.equal(obs[:,380:2940],env.depth_image.flatten(1))
    assert bool(env.depth_fresh.all()) and torch.count_nonzero(obs[:,-3:-1])==0
    assert camera.rotation==(0.,0.,0.,1.) and camera.position==(.42,0.,0.)
    assert not camera.horizontal_flip_to_robot_frame
    checks['images']=[check_depth_image(frame,camera.near_plane,camera.far_plane) for frame in env.depth_image]
    checks['actor_count']=int(env._all_root_states.shape[0])
    assert checks['actor_count']==env.num_envs*41
    for index in range(env.num_envs):
        actual=env._all_root_states[env.pillar_actor_indices[index],:2]-env.env_origins[index,:2]
        assert torch.allclose(actual,env.pillar_centers[index],atol=2e-5,rtol=0.)

    zero=torch.zeros(env.num_envs,2,device=env.device)
    before=env.depth_image.clone()
    for tick in range(1,10):
        _,_,_,done,_=env.step(zero)
        assert not done.any() and not env.depth_fresh.any()
        assert torch.equal(before,env.depth_image) and bool((env.depth_age==tick).all())
    _,_,_,done,_=env.step(zero)
    assert not done.any() and bool(env.depth_fresh.all()) and bool((env.depth_age==0).all())
    checks['camera_cadence']={'held_ticks':9,'refresh_tick':10}

    ids=torch.tensor([0],device=env.device,dtype=torch.long)
    residue=('actions','last_actions','output_actions','last_output_actions','last_dof_vel',
             'last_root_vel','last_base_lin_vel','last_base_ang_vel','last_root_states')
    for name in residue: getattr(env,name).fill_(.125)
    for history in tuple(env.obs_history)+tuple(env.critic_history): history.fill_(.125)
    for name in ('PID_FirstAxis','PID_SecondAxis'):
        for field in ('last_error','integral'): getattr(getattr(env,name),field).fill_(.125)
    old=snapshot(env)
    env.reset_idx(ids)
    new=snapshot(env)
    zero_fields=list(residue)+['depth_image','capture_camera_pose','obs_history','critic_history',
        'PID_FirstAxis.last_error','PID_FirstAxis.integral','PID_SecondAxis.last_error','PID_SecondAxis.integral']
    checks['partial_reset']=compare_partial_reset(old,new,ids,zero_fields)
    obs=env.get_observations(); critic=env.get_privileged_observations()
    assert torch.equal(env.depth_image[1:],old['depth_image'][1:])
    assert torch.count_nonzero(obs[0,-3:-1])==0
    held=obs.clone();held[:,-1]=0.
    h,c=policy.initial_state(env.num_envs,env.device)
    h.fill_(.25)
    starts=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device);starts[0]=True
    _,_,state=policy.step(held,critic,(h,c),starts)
    assert torch.count_nonzero(state[0][:,0])==0 and torch.equal(state[0][:,1:],h[:,1:])
    checks['recurrent_reset']={'selected_zero':True,'peers_preserved':True}

    env.reset()
    old_index=int(env._task_indices_cpu[0]);old_image=env.depth_image[0].clone()
    env._task_indices_cpu[0]=(old_index+1)%len(env.catalog.tasks)
    env._reset_to_tasks(ids,resample=False);env.get_observations()
    changed=float((env.depth_image[0]-old_image).abs().mean())
    assert changed>.01, 'Task pair must change the rendered depth'
    checks['paired_image_change_mae_m']=changed
    env._task_indices_cpu[0]=old_index
    env._reset_to_tasks(ids,resample=False);env.get_observations()

    local=env.root_states[0,:2]-env.env_origins[0,:2]
    delta=env.pillar_centers[0]-local
    nearest=int(torch.linalg.vector_norm(delta,dim=-1).argmin())
    yaw=math.atan2(float(delta[nearest,1]),float(delta[nearest,0]))
    _set_root_pose(env,0,yaw=yaw);env.get_observations()
    checks['known_pillar_depth']=_check_known_pillar_depth(env,0)

    target_rows=[]
    for sign in (-1.,1.):
        env.reset()
        _,_,_,done,info=env.step(torch.full_like(zero,sign))
        actual=info['motor_navigation']['motor_targets']
        expected=actual.new_tensor([.02,.04])*sign
        assert not done.any() and torch.allclose(actual,expected.expand_as(actual),atol=1e-6,rtol=0.)
        target_rows.append(actual[0].cpu().tolist())
    checks['normalized_action_rate_mapping']={'negative_first_step':target_rows[0],'positive_first_step':target_rows[1],
        'meaning':'actual limited motor targets; not a proof of robot translation'}

    env.reset()
    outside=torch.tensor([8.8,0.],device=env.device)
    _set_root_pose(env,0,local_xy=outside)
    _,_,_,done,info=env.step(zero)
    post=env.root_states[:,:2]-env.env_origins[:,:2]
    checks['terminal_before_reset']=check_terminal_snapshot(info['motor_navigation'],done,post,0,8.,outside)
    env.reset()
    return {'status':'PASS','scope':'GPU execution integrity only; no navigation success claim',
            'checks':checks,'elapsed_s':time.monotonic()-begin}
