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

from torch.utils.tensorboard import SummaryWriter


class FrankaRope(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]

        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1 / 60.

        self.num_rope_joints = 24 * 2
        self.cfg["env"]["numObservations"] = (7+self.num_rope_joints)* 2 + 3+3+4 + 2# 25 = (7+24) * 2(angle, angvel) + 3(rope-pos) + 3(hand-pos) + 4(hand-ori) + 2(command)
        self.cfg["env"]["numActions"] = 6

        self.writer = SummaryWriter(log_dir="./runs/FrankaRope/summaries")
        self.counter = 0
        self.command_counter = 0
        self.print_counter = 0
        self.reset_counter = 0

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
        self.franka_default_dof_pos = torch.zeros(7+self.num_rope_joints, device=self.device) 
        self.franka_default_dof_pos[:7] = to_torch([1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469], device=self.device)
        
        # Die here
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.franka_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_franka_dofs]
        self.dof_pos = self.franka_dof_state[..., 0]
        self.dof_vel = self.franka_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs        
        
        self.command = torch_rand_float(-1,1,(self.num_envs, 2) ,device=self.device)   
        self.command[:,1] = torch_rand_float(0.5,1.5,(self.num_envs,1),device=self.device)[:, 0]
        # self.command = torch.tensor(0.2*np.ones((self.num_envs, 1)), dtype=torch.float32, device=self.device)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.vis_init() # For visualizing target pos

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

        # set franka dof properties
        franka_dof_props = self.gym.get_asset_dof_properties(franka_asset)
        self.franka_dof_lower_limits = []
        self.franka_dof_upper_limits = []
        for i in range(self.num_franka_dofs):
            if i < 7:
                franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
                franka_dof_props['stiffness'][i] = 0.0 #franka_dof_stiffness[i]
                franka_dof_props['damping'][i] = 0.0 #franka_dof_damping[i]
                self.franka_dof_lower_limits.append(franka_dof_props['lower'][i])
                self.franka_dof_upper_limits.append(franka_dof_props['upper'][i])
            else:
                # __import__('pdb').set_trace()
                franka_dof_props['driveMode'][i] = gymapi.DOF_MODE_NONE
                # franka_dof_props['stiffness'][i] = 0.0
                # franka_dof_props['damping'][i] = 0.0

                # franka_dof_props['hasLimits'][i] = True
                # franka_dof_props['lower'][i] = -np.pi
                # franka_dof_props['upper'][i] = np.pi
                # franka_dof_props['velocity'][i] = 10
                # franka_dof_props['effort'][i] = 10
                
                self.franka_dof_lower_limits.append(-np.inf)
                self.franka_dof_upper_limits.append(np.inf)

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

            # franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 0, 0)
            franka_actor = self.gym.create_actor(env_ptr, franka_asset, franka_start_pose, "franka", i, 1, 0) # disable self collision
            self.gym.set_actor_dof_properties(env_ptr, franka_actor, franka_dof_props)

            self.envs.append(env_ptr)
            self.frankas.append(franka_actor)


    def compute_reward(self, actions):
        
        time_since_reset = self.progress_buf * self.dt

        # DOF position reward
        dof_pos = self.obs_buf[:, :6]
        target_dof = torch.ones((1, 6), device="cuda:0")
        # target_dof[0, 1] = -1.0
        dof_reward = torch.sum(torch.square(dof_pos - target_dof), dim=-1)

        rope_pos = self.obs_buf[:, -12:-9]
        # target_pos = torch.tensor([-0.8, 0., 0.8], device="cuda:0")
        r = 0.8 * 2**0.5
        x = r*torch.cos(np.pi*self.command[:, 0])
        y = r*torch.sin(np.pi*self.command[:, 0])
        target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        target_pos[:, 0] = x
        target_pos[:, 1] = y
        target_pos[:, 2] = self.command[:, 1]

        pos_error = torch.sum(torch.square(target_pos - rope_pos), dim=1)
        pos_reward = torch.exp(-pos_error)


        # target_pos = torch.zeros((self.progress_buf.shape[0], 3), device="cuda:0")
        # target_pos[:, :2] = torch_rand_float(0.6, 1.0, (self.progress_buf.shape[0], 2), device="cuda:0")
        # target_pos[:, 2] += torch_rand_float(0.4, 1.0, (self.progress_buf.shape[0], 1), device="cuda:0")[:, 0]
        # target_pos = torch.ones((progress_buf.shape[0], 3), device="cuda:0") * target_pos
        # cycle_per_second = 1
        # w = (2 * np.pi) / cycle_per_second
        # r = 0.4
        #
        # target_pos[:, 1] += r * torch.cos(w*time_since_reset)
        # target_pos[:, 2] += r * torch.sin(w*time_since_reset)
        # pos_reward = torch.sum(torch.exp(-torch.square(rope_pos - target_pos)), dim=-1)

        # DOF velocity penalty
        dof_vel = self.obs_buf[:, 55:110]
        dof_vel_reward = 0.000001 * torch.sum(dof_vel ** 2, dim=-1)

        # regularization on the actions (summed for each environment)
        action_reward = torch.sum(actions ** 2, dim=-1) * self.action_penalty_scale
        # print("Action: ", action_reward)

        if self.print_counter % 10 == 0:
            print("Position reward: ", pos_reward.mean().detach().item())
            print("Action: ", action_reward.mean().detach().item())
            # print("Rope position: ", rope_pos.mean(dim=0))
            # print("target positoin", target_pos[0,:2])
        self.print_counter += 1

        # Hand position reward
        # target = cycle_target(time_since_reset)

        default_hand_pos = torch.tensor([0.6, 0.2, 0.7], device="cuda:0")
        default_hand_pos = torch.ones((self.progress_buf.shape[0], 3), device="cuda:0") * default_hand_pos

        cycle_per_second = 1
        w = (2 * np.pi) / cycle_per_second
        r = 0.1
        center = default_hand_pos
        target = center

        target[:, 1] += r * torch.cos(w * time_since_reset)
        target[:, 2] += r * torch.sin(w * time_since_reset)
        ###############################################
        hand_pos = self.obs_buf[:, -9:-6]

        hand_ori = self.obs_buf[:, -4:]
        # TODO: change it to correct direction
        target_ori = torch.tensor([0, 0, 0, 1], device="cuda:0")
        
        hand_pos_reward = torch.exp(torch.clip(hand_pos[:,2]-0.5,None,0))
        hand_ori_reward = torch.sum(torch.exp(-torch.square(hand_ori - target_ori)), dim=-1)

        rewards = pos_reward*hand_pos_reward - action_reward#- dof_vel_reward - action_reward
        self.rew_buf[:] = rewards

        # reset if max length reached
        self.reset_buf[:] = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

        # Logging
        # self.writer.add_scalar("rewards/position", pos_reward.mean(), self.counter)
        for key, value in locals().items():
            if key[-6:] == "reward":
                self.writer.add_scalar("rewards/" + key[:-7], value.mean(), self.counter)
        self.counter += 1


    def compute_observations(self):

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # if self.counter % 200 == 0:
        #     self.command_counter+=1
        #     angle = (self.command_counter*0.1) % 2 - 1
        #     print(angle)        
        #     self.command[:, 0] = torch.tensor(angle*np.ones((self.num_envs)), dtype=torch.float32, device=self.device)
        #     self.command[:, 1] += 0.05
        # print("Rigid body state: ", self.rigid_body_states.shape)

        rope_end_state = self.rigid_body_states[:, -1, :]
        rope_end_pos = rope_end_state[:, :3]
        # print("Rope end position: ", rope_end_state[0, :3])
        # print("Dof velocities: ", self.dof_vel[0, :])
        # print("Dof positions: ", self.dof_pos[0, :])

        hand_state = self.rigid_body_states[:, 7, :]
        hand_pos = hand_state[:, :3]
        hand_ori = hand_state[:, 3:7]

        dof_pos = self.dof_pos - self.franka_default_dof_pos
        self.obs_buf = torch.cat((dof_pos, self.dof_vel * self.dof_vel_scale, rope_end_pos, hand_pos, hand_ori, self.command),
                                 dim=-1)
        # Logging
        self.writer.add_scalar("obs/hand_x", hand_pos[0, 0], self.counter)

        return self.obs_buf

    def reset_idx(self, env_ids):
        # Reset Command
        # self.command[env_ids, :] = torch_rand_float(-1,1,(len(env_ids), 1) ,device=self.device)
        # self.print_counter = self.print_counter%10
        # angle = (self.reset_counter*0.1) % 2 - 1
        # print(angle)
        # angle = -0.9
        # self.command = torch.tensor(angle*np.ones((self.num_envs, 1)), dtype=torch.float32, device=self.device)
        new_command = torch_rand_float(-1,1,(self.num_envs, 2) ,device=self.device)   
        new_command[:,1] = torch_rand_float(0.5,1.5,(self.num_envs,1),device=self.device)[:, 0]
        self.command[env_ids, :] = new_command[env_ids, :]
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
        self.reset_counter += 1

    def pre_physics_step(self, actions):
        self.actions = torch.clip(actions.clone(), -1.0, 1.0).to(self.device)
        rigid_body_forces = torch.zeros((self.progress_buf.shape[0], self.num_bodies, 3), device=self.device)
        # rigid_body_forces[:, -3, 2] = 100.
        forces = torch.zeros_like(self.dof_pos, device=self.device)
        forces[:, :6] = self.actions * 400.
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)
        # self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(rigid_body_forces),
        #                                         None,
        #                                         gymapi.ENV_SPACE)

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)
        
        self.vis_step()
    
    def vis_init(self):
        """
        Initialize visualizing settings
        From isaacgym/python/examples/transform.py
        """
        self.vis_colors = [tuple(0.5 + 0.5 * np.random.random(3)) for _ in range(self.num_envs)]


    def vis_step(self):
        """
        Set visualizer objects
        """
        rope_pos = self.obs_buf[:, -12:-9]
        # target_pos = torch.tensor([-0.8, 0., 0.8], device="cuda:0")
        r = 0.8 * 2**0.5
        x = r*torch.cos(np.pi*self.command[:, 0])
        y = r*torch.sin(np.pi*self.command[:, 0])
        target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        target_pos[:, 0] = x
        target_pos[:, 1] = y
        target_pos[:, 2] = self.command[:, 1]

        self.gym.clear_lines(self.viewer)

        for env_id in range(self.num_envs):            
            # Get the transforms we want to visualize
            vis_rope_pos = rope_pos[env_id].detach().tolist()
            vis_targ_pos = target_pos[env_id].detach().tolist()

            vis_rope_pose = gymapi.Transform()
            vis_rope_pose.p = gymapi.Vec3(vis_rope_pos[0], vis_rope_pos[1], vis_rope_pos[2])
            vis_rope_pose.r = gymapi.Quat.from_euler_zyx(-0.5 *  np.pi, 0, 0)

            vis_targ_pose = gymapi.Transform()
            vis_targ_pose.p = gymapi.Vec3(vis_targ_pos[0], vis_targ_pos[1], vis_targ_pos[2])
            vis_targ_pose.r = gymapi.Quat.from_euler_zyx(-0.5 *  np.pi, 0, 0)

            sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * np.pi, 0, 0)
            sphere_pose = gymapi.Transform(r=sphere_rot)
            sphere_geom_rope = gymutil.WireframeSphereGeometry(0.04, 12, 12, sphere_pose, color=(1,1,0))
            sphere_geom_targ = gymutil.WireframeSphereGeometry(0.04, 12, 12, sphere_pose, color=self.vis_colors[env_id])

            gymutil.draw_lines(sphere_geom_rope, 
                                self.gym, self.viewer, self.envs[env_id], 
                                vis_rope_pose)

            gymutil.draw_lines(sphere_geom_targ, 
                                self.gym, self.viewer, self.envs[env_id], 
                                vis_targ_pose)



        pass

#####################################################################
###=========================jit functions=========================###
#####################################################################
# @torch.jit.script
# def cycle_target(t):
#     # type: (Tensor) -> Tensor

#     default_hand_pos = torch.tensor([0.2, 0.2, 0.7], device="cuda:0")

#     cycle_per_second = 1
#     w = (2 * np.pi) / cycle_per_second
#     r = 0.1
#     center = default_hand_pos 
#     target = center
#     target[1] = r * torch.cos(w*t)
#     target[2] = r * torch.sin(w*t)
#     return target

# @torch.jit.script
# def compute_franka_reward(
#         obs_buf, reset_buf, progress_buf, actions, action_penalty_scale, max_episode_length, time_since_reset
# ):
#     # type: (Tensor, Tensor, Tensor, Tensor, float, float, Tensor) -> Tuple[Tensor, Tensor]
#
#     # DOF position reward
#     dof_pos = obs_buf[:, :6]
#     target_dof = torch.ones((1, 6), device="cuda:0")
#     # target_dof[0, 1] = -1.0
#     dof_reward = torch.sum(torch.square(dof_pos - target_dof), dim=-1)
#
#     rope_pos = obs_buf[:, -10:-7]
#     # target_pos = torch.tensor([0.4, 0.4, 0.5], device="cuda:0")
#     target_pos = torch.zeros((progress_buf.shape[0], 3), device="cuda:0")
#     target_pos[:, :2] = torch_rand_float(0.6, 1.0, (progress_buf.shape[0], 2), device="cuda:0")
#     target_pos[:, 2] += torch_rand_float(0.4, 1.0, (progress_buf.shape[0], 1), device="cuda:0")[:, 0]
#     # target_pos = torch.ones((progress_buf.shape[0], 3), device="cuda:0") * target_pos
#     # cycle_per_second = 1
#     # w = (2 * np.pi) / cycle_per_second
#     # r = 0.4
#     #
#     # target_pos[:, 1] += r * torch.cos(w*time_since_reset)
#     # target_pos[:, 2] += r * torch.sin(w*time_since_reset)
#     pos_reward = torch.sum(torch.exp(-torch.square(rope_pos - target_pos)), dim=-1)
#
#     # DOF velocity penalty
#     dof_vel = obs_buf[:, 55:110]
#     dof_vel_penalty = 0.000001 * torch.sum(dof_vel ** 2, dim=-1)
#
#     # regularization on the actions (summed for each environment)
#     action_penalty = torch.sum(actions ** 2, dim=-1)
#
#     # Hand position reward
#     # target = cycle_target(time_since_reset)
#
#     default_hand_pos = torch.tensor([0.6, 0.2, 0.7], device="cuda:0")
#     default_hand_pos = torch.ones((progress_buf.shape[0], 3), device="cuda:0") * default_hand_pos
#
#     cycle_per_second = 1
#     w = (2 * np.pi) / cycle_per_second
#     r = 0.1
#     center = default_hand_pos
#     target = center
#
#     target[:, 1] += r * torch.cos(w*time_since_reset)
#     target[:, 2] += r * torch.sin(w*time_since_reset)
#     ###############################################
#     hand_pos = obs_buf[:, -7:-4]
#
#     hand_ori = obs_buf[:, -4:]
#     # TODO: change it to correct direction
#     target_ori = torch.tensor([0, 0, 0, 1], device="cuda:0")
#
#     hand_pos_reward = torch.sum(torch.exp(-torch.square(hand_pos - target)), dim=-1)
#     hand_ori_reward = torch.sum(torch.exp(-torch.square(hand_ori - target_ori)), dim=-1)
#
#     writer.add_scalar("rewards/position", pos_reward.mean(), 0)
#
#     rewards = pos_reward - dof_vel_penalty - action_penalty_scale * action_penalty * 0.01
#
#     # reset if max length reached
#     reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)
#
#     # print("Rewards: ", rewards)
#     # print("Target: ", target_pos[0, :])
#     # print("Rope: ", rope_pos[0, :])
#     # print(f"Rope End Position: {rope_pos[0,0].item()} {rope_pos[0,1].item()} {rope_pos[0,2].item()}" )
#     # print(f"Targ Position: {target_pos[0].item()} {target_pos[1].item()} {target_pos[2].item()}" )
#     # print(f"Hand Position: {hand_pos[0,0].item()} {hand_pos[0,1].item()} {hand_pos[0,2].item()}" )
#     # print(f"Targ Position: {target[0,0].item()} {target[0,1].item()} {target[0,2].item()}" )
#     # print("Average reward: ", torch.mean(rewards))
#
#     return rewards, reset_buf

