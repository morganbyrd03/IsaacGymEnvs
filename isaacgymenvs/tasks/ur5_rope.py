# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
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

import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from tasks.base.vec_task import VecTask


class UR5Rope(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self.cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self.cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self.cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1 / 60.

        num_obs = 12
        num_acts = 6

        self.cfg["env"]["numObservations"] = 18 + 46 + 3
        self.cfg["env"]["numActions"] = 6

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.franka_default_dof_pos = to_torch([1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0., 0.], device=self.device)
        self.franka_default_dof_pos = torch.zeros(32, device=self.device)
        self.franka_default_dof_pos[:7] = to_torch([1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469], device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.franka_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_franka_dofs]
        self.dof_pos = self.franka_dof_state[..., 0]
        self.dof_vel = self.franka_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        franka_asset_file = "urdf/franka_description/robots/franka_panda_rope.urdf"

        # if "asset" in self.cfg["env"]:
        #     asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        #                               self.cfg["env"]["asset"].get("assetRoot", asset_root))
        #     franka_asset_file = self.cfg["env"]["asset"].get("assetFileNameFranka", franka_asset_file)

        # load franka asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        # asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.replace_cylinder_with_capsule = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        franka_asset = self.gym.load_asset(self.sim, asset_root, franka_asset_file, asset_options)

        franka_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 400], dtype=torch.float,
                                        device=self.device)
        franka_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 80], dtype=torch.float, device=self.device)

        self.num_franka_bodies = self.gym.get_asset_rigid_body_count(franka_asset)
        self.num_franka_dofs = self.gym.get_asset_dof_count(franka_asset)

        print("DOFs, bodies: ", self.num_franka_dofs, self.num_franka_bodies)

        # Note - for this asset we are loading the actuator info from the MJCF
        actuator_props = self.gym.get_asset_actuator_properties(franka_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        self.joint_gears = to_torch(motor_efforts, device=self.device)

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        for i in range(self.num_franka_dofs):
            franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
            # if self.physics_engine == gymapi.SIM_PHYSX:
            #     franka_dof_props['stiffness'][i] = franka_dof_stiffness[i]
            #     franka_dof_props['damping'][i] = franka_dof_damping[i]
            # else:
            #     franka_dof_props['stiffness'][i] = 7000.0
            #     franka_dof_props['damping'][i] = 50.0

            self.franka_dof_lower_limits.append(-1.0)
            self.franka_dof_upper_limits.append(1.0)

        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)

        franka_start_pose = gymapi.Transform()
        franka_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        franka_start_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        self.frankas = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)


    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_franka_reward(
            self.obs_buf, self.reset_buf, self.progress_buf, self.actions, self.action_penalty_scale, self.max_episode_length
        )

    def compute_observations(self):

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # print("Rigid body state: ", self.rigid_body_states.shape)

        rope_end_state = self.rigid_body_states[:, -1, :]
        rope_end_pos = rope_end_state[:, :3]
        print("Rope end position: ", rope_end_state[0, :3])

        dof_pos = self.dof_pos - self.franka_default_dof_pos
        self.obs_buf = torch.cat((dof_pos, self.dof_vel * self.dof_vel_scale, rope_end_pos),
                                 dim=-1)

        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # reset franka
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0) + 0.25 * (
                        torch.rand((len(env_ids), self.num_franka_dofs), device=self.device) - 0.5),
            self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        self.dof_pos[env_ids, :] = pos
        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = torch.clip(actions.clone(), -1.0, 1.0).to(self.device)
        forces = torch.zeros_like(self.dof_pos, device=self.device)
        forces[:, :6] = self.actions * 400.
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_franka_reward(
        obs_buf, reset_buf, progress_buf, actions, action_penalty_scale, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float) -> Tuple[Tensor, Tensor]

    # DOF position reward
    dof_pos = obs_buf[:, :6]
    target_dof = torch.ones((1, 6), device="cuda:0")
    # target_dof[0, 1] = -1.0
    dof_reward = torch.sum(torch.square(dof_pos - target_dof), dim=-1)

    rope_pos = obs_buf[:, -3:]
    target_pos = torch.tensor([0.2, 0.8, 0.5], device="cuda:0")
    pos_reward = torch.sum(torch.exp(-torch.square(rope_pos - target_pos)), dim=-1)

    # DOF velocity penalty
    dof_vel = obs_buf[:, 6:]
    dof_vel_penalty = 0.0 * torch.sum(dof_vel ** 2, dim=-1)

    # regularization on the actions (summed for each environment)
    action_penalty = torch.sum(actions ** 2, dim=-1)

    rewards = pos_reward - dof_vel_penalty - action_penalty_scale * action_penalty * 0.0

    # reset if max length reached
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    # print("Rewards: ", rewards)
    print("Average reward: ", torch.mean(rewards))

    return rewards, reset_buf

