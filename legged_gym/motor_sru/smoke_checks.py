"""Bounded simulator smoke assertions, separate from training and its rewards.

PASS requires actual rendered depth and pre-reset terminal state. CPU fixtures
exercise the assertion logic only and must never be reported as physics PASS.
"""
import hashlib
import math
import time
import traceback

import torch


def tensor_digest(state):
    digest=hashlib.sha256()
    for key,tensor in sorted(state.items()):
        value=tensor.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((str(value.dtype),tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def check_depth_image(depth,near,far):
    """Require spatially varying finite metric depth with at least some returns."""
    if not bool(torch.isfinite(depth).all()):
        raise AssertionError('nonfinite metric depth')
    valid=(depth>=float(near)) & (depth<float(far)-1e-4)
    spread=float(depth.max()-depth.min())
    deviation=float(depth.std(unbiased=False))
    if float(valid.float().mean())<.01 or spread<.1 or deviation<.01:
        raise AssertionError('depth is empty, all-far, or effectively constant')
    return dict(valid_fraction=float(valid.float().mean()),minimum_m=float(depth.min()),
        maximum_m=float(depth.max()),spatial_std_m=deviation,range_m=spread)


def compare_partial_reset(before,after,reset_ids,zero_fields):
    """Strict peer identity, plus explicit selected-buffer zero requirements."""
    if set(before)!=set(after):
        raise AssertionError('partial reset snapshot keys differ')
    count=next(iter(before.values())).shape[0]
    peer=torch.ones(count,dtype=torch.bool,device=reset_ids.device)
    peer[reset_ids]=False
    for key in before:
        if not torch.equal(before[key][peer],after[key][peer]):
            raise AssertionError('partial reset altered peer buffer: '+key)
    for key in zero_fields:
        if bool(torch.count_nonzero(after[key][reset_ids])):
            raise AssertionError('reset retained old data in '+key)
    return dict(peer_buffers_bitwise_unchanged=sorted(before),selected_zero_buffers=sorted(zero_fields))


def check_hold(before_depth,after_depth,before_feature,after_feature,calls_before,calls_after,fresh):
    if bool(fresh.any()) or calls_after!=calls_before:
        raise AssertionError('held camera frame was marked fresh or encoded again')
    if not torch.equal(before_depth,after_depth) or not torch.equal(before_feature,after_feature):
        raise AssertionError('held depth/latent cache changed')


def check_terminal_snapshot(nav,done,post_reset_xy,selected,half_extent,expected_xy):
    if not bool(done[selected]) or not bool(nav['out_of_bounds'][selected]):
        raise AssertionError('forced outside state did not produce an out-of-bounds terminal')
    captured=nav['local_xy'][selected]
    if float(captured.abs().max())<=half_extent:
        raise AssertionError('terminal info contains an in-bounds/reset position')
    if float(torch.linalg.vector_norm(captured-expected_xy))>.15:
        raise AssertionError('terminal pose differs from the controlled outside state')
    if float(post_reset_xy[selected].abs().max())>=half_extent:
        raise AssertionError('simulator did not reset the completed environment')
    if torch.allclose(captured,post_reset_xy[selected],atol=.1,rtol=0.):
        raise AssertionError('terminal trace was overwritten by reset position')
    return dict(terminal_xy=captured.detach().cpu().tolist(),
        post_reset_xy=post_reset_xy[selected].detach().cpu().tolist())


def central_cylinder_depth(distance_from_camera,radius,width,hfov_degrees):
    """Optical-axis depth of the two central pixel columns, including pixel offset."""
    focal=(float(width)/2.)/math.tan(math.radians(hfov_degrees)/2.)
    slope=.5/focal
    a=1.+slope*slope
    discriminant=distance_from_camera**2-a*(distance_from_camera**2-radius**2)
    if discriminant<=0:
        raise AssertionError('known pillar is subpixel at the optical center')
    return (distance_from_camera-math.sqrt(discriminant))/a


@torch.inference_mode()
def run_smoke_checks(env,policy):
    """Use the production render/reset/step path; perform no optimization."""
    begin=time.monotonic()
    report=dict(status='RUNNING',scope='bounded real simulator reset/sensor checks; not a navigation success evaluation',checks={})
    original_indices=env._task_indices_cpu.copy()
    hook=None
    try:
        if env.num_envs<2 or env.cfg.task_catalog.resample_on_reset:
            raise AssertionError('smoke requires >=2 environments with fixed task assignment')
        if str(env.device) not in ('cuda:2','cuda:3'):
            raise AssertionError('real smoke is restricted to physical GPU 2/3')
        camera=env.cfg.camera
        if (camera.height,camera.width,camera.motor_steps_per_frame)!=(40,64,10):
            raise AssertionError('smoke expects the declared 40x64 camera held for ten motor ticks')
        if camera.horizontal_flip_to_robot_frame:
            raise AssertionError('this base-link optical convention requires unmirrored policy pixels')
        encoder=env.depth_encoder
        if any(parameter.requires_grad for parameter in encoder.parameters()) or encoder.training:
            raise AssertionError('depth encoder is not frozen in eval mode')
        if any(module.training for module in encoder.modules() if isinstance(module,torch.nn.modules.batchnorm._BatchNorm)):
            raise AssertionError('frozen depth BatchNorm is training')
        initial_hash=tensor_digest(encoder.state_dict())
        encoded=[]
        hook=encoder.register_forward_pre_hook(lambda module,args:encoded.append(int(args[0].shape[0])))
        env.reset()
        ids=torch.tensor([0],device=env.device,dtype=torch.long)
        zero_actions=torch.zeros(env.num_envs,2,device=env.device)
        report['checks']['initial_images']=[check_depth_image(frame,camera.near_plane,camera.far_plane) for frame in env.depth_image]
        if not bool(torch.isfinite(env.depth_features).all()) or not bool(env.depth_fresh.all()):
            raise AssertionError('initial depth features are invalid or not fresh')
        original_frame=env.depth_image[0].clone()
        original_feature=env.depth_features.clone()
        depth_before=env.depth_image.clone()
        calls_before=len(encoded)
        count_before=env.depth_encode_count.clone()
        for tick in range(1,10):
            _,_,_,done,_=env.step(zero_actions)
            if bool(done.any()):
                raise AssertionError('unexpected terminal during stationary camera cadence check')
            check_hold(depth_before,env.depth_image,original_feature,env.depth_features,calls_before,len(encoded),env.depth_fresh)
            if not bool((env.depth_age==tick).all()) or not torch.equal(env.depth_encode_count,count_before):
                raise AssertionError('camera age/count differs from ten-tick cadence')
        _,_,_,done,_=env.step(zero_actions)
        if bool(done.any()) or not bool(env.depth_fresh.all()) or not bool((env.depth_age==0).all()):
            raise AssertionError('tenth motor tick failed to supply one new frame')
        if len(encoded)!=calls_before+1 or encoded[-1]!=env.num_envs or not torch.equal(env.depth_encode_count,count_before+1):
            raise AssertionError('tenth tick did not encode each environment exactly once')
        report['checks']['camera_cadence']=dict(held_ticks=9,new_frame_tick=10,duplicate_hold_encodes=0)

        # Insert deterministic old-episode residue into memory, not robot poses.
        # This proves zeroing even when a stationary physical rollout has zero actions.
        memory_names=('actions','last_actions','output_actions','last_output_actions','last_dof_vel',
                      'last_root_vel','last_base_lin_vel','last_base_ang_vel','last_root_states')
        for name in memory_names:
            getattr(env,name).fill_(.125)
        for history in tuple(env.obs_history)+tuple(env.critic_history):
            history.fill_(.125)
        for name in ('PID_FirstAxis','PID_SecondAxis'):
            getattr(env,name).last_error.fill_(.125)
            getattr(env,name).integral.fill_(.125)
        before=_runtime_snapshot(env)
        env.reset_idx(ids)
        after=_runtime_snapshot(env)
        zero_fields=list(memory_names)+['root_velocity','dof_vel','base_lin_vel','base_ang_vel',
            'obs_history','critic_history','depth_image','depth_features','capture_camera_pose',
            'PID_FirstAxis.last_error','PID_FirstAxis.integral','PID_SecondAxis.last_error','PID_SecondAxis.integral']
        report['checks']['partial_reset_before_new_observation']=compare_partial_reset(before,after,ids,zero_fields)
        if not bool(env._depth_pending[0]) or bool(env.depth_fresh[0]):
            raise AssertionError('reset does not require a new depth capture')
        calls_before=len(encoded)
        obs=env.get_observations(); critic=env.get_privileged_observations()
        if len(encoded)!=calls_before+1 or encoded[-1]!=1 or not bool(env.depth_fresh[0]):
            raise AssertionError('partial reset did not encode exactly the selected new observation')
        if not torch.equal(env.depth_features[1:],before['depth_features'][1:]) or not torch.equal(env.depth_image[1:],before['depth_image'][1:]):
            raise AssertionError('partial capture altered peer cached frames')
        _check_recurrent_reset(policy,obs,critic)
        report['checks']['recurrent_reset']=dict(fresh_reset_matches_zero_initial_state=True,
            held_reset_zero=True,peers_bitwise_preserved=True,initial_memory='deterministic injected test state; not learned rollout state')

        # A different real task changes pillar actors, root pose, and camera pixels.
        # The same production reset/capture methods are used, without a physics step.
        env.reset()
        old_task=env.catalog.tasks[int(env._task_indices_cpu[0])]
        before_map=env.depth_image[0].clone()
        replacement=(int(env._task_indices_cpu[0])+1)%len(env.catalog.tasks)
        env._task_indices_cpu[0]=replacement
        env._reset_to_tasks(ids,resample=False)
        env.get_observations()
        new_task=env.catalog.tasks[replacement]
        changed_frame=env.depth_image[0].clone()
        image_delta=float((before_map-changed_frame).abs().mean())
        changed_fraction=float(((before_map-changed_frame).abs()>.01).float().mean())
        if old_task['geometry_sha256']==new_task['geometry_sha256'] or image_delta<.02 or changed_fraction<.01:
            raise AssertionError('real rendered depth did not change after changing the task geometry')
        actual_centers=env._all_root_states[env.pillar_actor_indices[0],:2]-env.env_origins[0,:2]
        if not torch.allclose(actual_centers,env.pillar_centers[0],atol=1e-5,rtol=0.):
            raise AssertionError('pillar actor poses do not match the selected task')
        report['checks']['task_reset_changes_actual_image']=dict(old_task=old_task['task_id'],new_task=new_task['task_id'],
            old_geometry_sha256=old_task['geometry_sha256'],new_geometry_sha256=new_task['geometry_sha256'],
            mean_absolute_depth_change_m=image_delta,changed_pixel_fraction=changed_fraction)

        # Aim the real mounted camera at a known nearest cylinder. Exact ray/cylinder
        # depth prevents a stale pillar renderer from passing on image change alone.
        local_xy=env.root_states[0,:2]-env.env_origins[0,:2]
        vectors=env.pillar_centers[0]-local_xy
        nearest=int(torch.linalg.vector_norm(vectors,dim=-1).argmin())
        yaw=math.atan2(float(vectors[nearest,1]),float(vectors[nearest,0]))
        _set_root_pose(env,0,yaw=yaw)
        env.get_observations()
        pointed_frame=env.depth_image[0].clone()
        report['checks']['known_pillar_depth']=_check_known_pillar_depth(env,0)
        env._reset_to_tasks(ids,resample=False)
        env.get_observations()
        restored_frame=env.depth_image[0].clone()
        reset_error=float((restored_frame-changed_frame).abs().mean())
        yaw_delta=float((pointed_frame-changed_frame).abs().mean())
        if yaw_delta<.02 or reset_error>.01:
            raise AssertionError('controlled camera yaw/reset did not change and restore the real image')
        report['checks']['yaw_then_reset_image']=dict(pointing_change_m=yaw_delta,reset_reconstruction_error_m=reset_error,
            purpose='controlled sensor probe only; not a feasible task trajectory')
        report['depth_evidence']={key:value.detach().cpu().tolist() for key,value in
            [('original_task',before_map),('different_task',changed_frame),('pointing_at_known_pillar',pointed_frame),('after_reset',restored_frame)]}

        env._task_indices_cpu[:]=original_indices
        env.reset()
        half_extent=float(env.cfg.navigation.half_extent)
        expected_xy=torch.tensor([half_extent+.5,0.],device=env.device)
        _set_root_pose(env,0,local_xy=expected_xy)
        _,_,_,done,info=env.step(zero_actions)
        nav=info['motor_navigation']
        post_xy=env.root_states[:,:2]-env.env_origins[:,:2]
        report['checks']['terminal_before_autoreset']=check_terminal_snapshot(nav,done,post_xy,0,half_extent,expected_xy)
        if not torch.allclose(post_xy[0],env.task_starts[0],atol=1e-5,rtol=0.):
            raise AssertionError('completed environment did not return to its assigned start')
        final_hash=tensor_digest(encoder.state_dict())
        if initial_hash!=final_hash or not bool(torch.isfinite(env.depth_features).all()):
            raise AssertionError('frozen encoder parameters/BN changed or final features invalid')
        report['checks']['frozen_encoder']=dict(state_sha256_before=initial_hash,state_sha256_after=final_hash,
            checkpoint_sha256=env.depth_encoder_sha256,feature_finite=True,all_parameters_frozen=True,batchnorm_eval=True)
        report['status']='PASS'
    except Exception as error:
        report['status']='FAIL'
        report['error']='{}: {}'.format(type(error).__name__,error)
        report['traceback']=traceback.format_exc()
    finally:
        if hook is not None:
            hook.remove()
        env._task_indices_cpu[:]=original_indices
        try:
            env.reset()
        except Exception as error:
            report['status']='FAIL'
            report['cleanup_error']='{}: {}'.format(type(error).__name__,error)
        report['wall_time_s']=time.monotonic()-begin
    return report


def _runtime_snapshot(env):
    names=('root_states','dof_pos','dof_vel','base_lin_vel','base_ang_vel','actions','last_actions',
        'output_actions','last_output_actions','last_dof_vel','last_root_vel','last_base_lin_vel',
        'last_base_ang_vel','last_root_states','episode_length_buf','episode_returns','depth_image',
        'depth_features','capture_camera_pose','depth_age','depth_fresh','depth_encode_count',
        'task_indices','task_reset_count','_depth_pending','_proprio_pending')
    result={name:getattr(env,name).clone() for name in names}
    result['root_velocity']=env.root_states[:,7:13].clone()
    result['all_actor_roots']=env._all_root_states.reshape(env.num_envs,9,13).clone()
    result['obs_history']=torch.stack(tuple(env.obs_history),dim=1).clone()
    result['critic_history']=torch.stack(tuple(env.critic_history),dim=1).clone()
    for name in ('PID_FirstAxis','PID_SecondAxis'):
        for key in ('last_error','integral'):
            result[name+'.'+key]=getattr(getattr(env,name),key).clone()
    return result


def _check_recurrent_reset(policy,obs,critic):
    with torch.no_grad():
        state=tuple(torch.ones_like(item)*.25 for item in policy.initial_state(len(obs),obs.device))
        starts=torch.zeros(len(obs),device=obs.device,dtype=torch.bool); starts[0]=True
        zeros=tuple(item.clone() for item in state)
        for item in zeros:
            item[:,0]=0.
        _,_,actual=policy.step(obs,critic,state,starts)
        _,_,expected=policy.step(obs,critic,zeros,torch.zeros_like(starts))
        if any(not torch.equal(a,b) for a,b in zip(actual,expected)):
            raise AssertionError('episode-start state does not match explicitly zeroed selected memory')
        held=obs.clone(); held[:,-1]=0.
        _,_,actual=policy.step(held,critic,state,starts)
        if any(bool(torch.count_nonzero(item[:,0])) for item in actual) or any(not torch.equal(a[:,1:],b[:,1:]) for a,b in zip(actual,state)):
            raise AssertionError('held memory failed selected-zero/peer-identity reset')


def _set_root_pose(env,index,yaw=None,local_xy=None):
    from isaacgym import gymtorch
    ids=torch.tensor([index],dtype=torch.long,device=env.device)
    if local_xy is not None:
        env.root_states[index,:2]=local_xy+env.env_origins[index,:2]
    if yaw is not None:
        env.root_states[index,3:7]=torch.tensor([0.,0.,math.sin(yaw/2),math.cos(yaw/2)],device=env.device)
    env.root_states[index,7:13]=0.
    indices=env.robot_actor_indices[ids].contiguous()
    env.gym.set_actor_root_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env._all_root_states),
        gymtorch.unwrap_tensor(indices),len(indices))
    env._sync_robot_state(ids)
    env._depth_pending[ids]=True
    env._proprio_pending[ids]=True


def _check_known_pillar_depth(env,index):
    camera=env.cfg.camera
    pose=env.capture_camera_pose[index].detach().cpu().double()
    origin=pose[:3]; q=pose[3:7]
    centers=(env.pillar_centers[index]+env.env_origins[index,:2]).detach().cpu().double()
    focal=(camera.width/2.)/math.tan(math.radians(camera.horizontal_fov)/2.)
    errors=[]; evidence=[]
    for row in (camera.height//2-1,camera.height//2):
        for column in (camera.width//2-1,camera.width//2):
            native_column=camera.width-1-column if camera.horizontal_flip_to_robot_frame else column
            ray=torch.tensor([1.,-(native_column-(camera.width-1)/2.)/focal,-(row-(camera.height-1)/2.)/focal],dtype=torch.float64)
            uv=torch.cross(q[:3],ray,dim=0)
            direction=ray+2*(q[3]*uv+torch.cross(q[:3],uv,dim=0))
            delta=origin[:2]-centers
            a=float(direction[:2].square().sum())
            b=2*(delta*direction[:2]).sum(-1)
            c=delta.square().sum(-1)-float(env.cfg.navigation.pillar_radius)**2
            disc=b.square()-4*a*c
            t=(-b-torch.sqrt(disc.clamp_min(0)))/(2*a)
            z=origin[2]+t*direction[2]
            valid=(disc>=0)&(t>camera.near_plane)&(t<camera.far_plane)&(z>=0)&(z<=env.cfg.navigation.pillar_height)
            if not bool(valid.any()):
                raise AssertionError('central controlled ray does not intersect a known pillar')
            expected=float(t[valid].min())
            actual=float(env.depth_image[index,row,column])
            error=abs(actual-expected)
            errors.append(error)
            evidence.append(dict(row=row,column=column,expected_m=expected,actual_m=actual,error_m=error))
    if max(errors)>.06:
        raise AssertionError('actual central pillar depth differs from map geometry: max error {:.6f} m'.format(max(errors)))
    return dict(max_absolute_error_m=max(errors),tolerance_m=.06,rays=evidence,
        method='full mounted-camera quaternion and optical-axis pixel rays intersect actual task cylinders')
