# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .anymal_c.anymal import Anymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .rotunbot.rotunbot import Rotunbot
from .rotunbot.rotunbot_config import  RotunbotRoughCfg, RotunbotRoughCfgPPO
from .rotunbot.vel_tracking.rotunbot_vel import RotunbotVel
from .rotunbot.vel_tracking.rotunbot_vel_config import RotunbotVelCfg , RotunbotVelCfgPPO
from .rotunbot.vel_tracking.rotunbot_vel_clean import RotunbotVelClean
from .rotunbot.vel_tracking.rotunbot_vel_clean_config import RotunbotVelCleanCfg, RotunbotVelCleanCfgPPO
from .rotunbot.trajectory_tracking.rotunbot_tra import RotunbotTrajectory
from .rotunbot.trajectory_tracking.rotunbot_tra_config import RotunbotTrajectoryCfg, RotunbotTrajectoryCfgPPO
from .rotunbot.vel_tracking.rotunbot_vel import RotunbotVel
from .rotunbot.vel_tracking.rotunbot_vel_config import RotunbotVelCfg , RotunbotVelCfgPPO
from .rotunbot.rotunbot_wheel.rotunbot_wheel_config import RotunbotWheelCfg , RotunbotWheelCfgPPO
from .rotunbot.rotunbot_wheel.rotunbot_wheel import RotunbotWheel
from .rotunbot.target_point.rotunbot_target_config import RotunbotTargetCfg, RotunbotTargetCfgPPO
from .rotunbot.target_point.rotunbot_target import RotunbotTarget
from .rotunbot.target_point.rotunbot_target_obstacle_config import RotunbotTargetObstacleCfg, RotunbotTargetObstacleCfgPPO
from .rotunbot.target_point.rotunbot_target_obstacle import RotunbotTargetObstacle
from .rotunbot.vel_tracking.rotunbot_real import RotunbotReal
from .rotunbot.vel_tracking.rotunbot_real_config import RotunbotRealCfg, RotunbotRealCfgPPO
from .rotunbot.target_point.rotunbot_target_lh import RotunbotTargetLH
from .rotunbot.target_point.rotunbot_target_lh_config import RotunbotTargetLHCfg, RotunbotTargetLHCfgPPO
from .rotunbot.target_point.rotunbot_target_repro import RotunbotTargetRepro
from .rotunbot.target_point.rotunbot_target_repro_config import RotunbotTargetReproCfg, RotunbotTargetReproCfgPPO
from .rotunbot.maze.rotunbot_maze import RotunbotMaze
from .rotunbot.maze.rotunbot_maze_config import RotunbotMazeCfg, RotunbotMazeCfgPPO
from .rotunbot.path_tracking.rotunbot_path import RotunbotPath
from .rotunbot.path_tracking.rotunbot_path_config import RotunbotPathCfg, RotunbotPathCfgPPO

import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO() )
task_registry.register( "anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO() )
task_registry.register( "anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO() )
task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO() )
task_registry.register( "rotunbot", Rotunbot, RotunbotRoughCfg(), RotunbotRoughCfgPPO() )
task_registry.register( "rotunbot_vel", RotunbotVel, RotunbotVelCfg(), RotunbotVelCfgPPO() )
task_registry.register( "rotunbot_vel_clean", RotunbotVelClean, RotunbotVelCleanCfg(), RotunbotVelCleanCfgPPO() )
task_registry.register( "rotunbot_tra", RotunbotTrajectory, RotunbotTrajectoryCfg(), RotunbotTrajectoryCfgPPO() )
task_registry.register( "rotunbot_wheel", RotunbotWheel, RotunbotWheelCfg(), RotunbotWheelCfgPPO() )
task_registry.register( "rotunbot_target", RotunbotTarget, RotunbotTargetCfg(), RotunbotTargetCfgPPO() )
task_registry.register( "rotunbot_target_obstacle", RotunbotTargetObstacle, RotunbotTargetObstacleCfg(), RotunbotTargetObstacleCfgPPO() )
task_registry.register( "rotunbot_real", RotunbotReal, RotunbotRealCfg(), RotunbotRealCfgPPO() )
task_registry.register( "rotunbot_target_lh", RotunbotTargetLH, RotunbotTargetLHCfg(), RotunbotTargetLHCfgPPO() )
task_registry.register( "rotunbot_target_repro", RotunbotTargetRepro, RotunbotTargetReproCfg(), RotunbotTargetReproCfgPPO() )
task_registry.register( "rotunbot_maze", RotunbotMaze, RotunbotMazeCfg(), RotunbotMazeCfgPPO() )
task_registry.register( "rotunbot_path", RotunbotPath, RotunbotPathCfg(), RotunbotPathCfgPPO() )
