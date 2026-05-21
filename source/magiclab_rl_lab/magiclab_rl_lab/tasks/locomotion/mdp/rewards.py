from __future__ import annotations

import torch
from typing import TYPE_CHECKING

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_threshold: float, command_name: str = "base_velocity",  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < command_threshold)

def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float, command_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


# def foot_clearance_reward(
#     env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
# ) -> torch.Tensor:
#     """Reward the swinging feet for clearing a specified height off the ground"""
#     asset: RigidObject = env.scene[asset_cfg.name]
#     foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
#     foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
#     reward = foot_z_target_error * foot_velocity_tanh
#     return torch.exp(-torch.sum(reward, dim=1) / std)


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward = torch.sum(is_contact, dim=-1).float()
    return reward * (command_norm < 0.1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


def pelvis_acceleration_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize aggressive pelvis linear acceleration."""
    asset: Articulation = env.scene[asset_cfg.name]
    root_vel = asset.data.root_lin_vel_w

    if not hasattr(env, "_pelvis_prev_root_vel") or env._pelvis_prev_root_vel.shape != root_vel.shape:
        env._pelvis_prev_root_vel = root_vel.clone()

    acc = (root_vel - env._pelvis_prev_root_vel) / env.step_dt
    env._pelvis_prev_root_vel = root_vel.clone()

    reset_mask = env.episode_length_buf < 2
    reward = torch.sum(torch.square(acc), dim=-1)
    reward[reset_mask] = 0.0
    return reward


def _periodic_interval_expectation(
    phase: torch.Tensor,
    start: float,
    end: float,
    kappa: float,
) -> torch.Tensor:
    """Expected phase indicator for a circular interval.

    The paper uses Von Mises-distributed start/end times. This deterministic
    expectation keeps the same circular interval semantics and smooth boundary
    uncertainty with a sigmoid concentration parameter.
    """
    width = end - start
    if width <= 0.0:
        width += 1.0
    if width >= 1.0:
        return torch.ones_like(phase)

    center = (start + 0.5 * width) % 1.0
    phase_error = torch.atan2(
        torch.sin(2.0 * torch.pi * (phase - center)),
        torch.cos(2.0 * torch.pi * (phase - center)),
    ) / (2.0 * torch.pi)
    return torch.sigmoid(kappa * (0.5 * width - torch.abs(phase_error)))


def periodic_bipedal_gait_reward(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    stance_ratio: float = 0.5,
    kappa: float = 40.0,
    force_scale: float = 50.0,
    velocity_scale: float = 1.0,
    std: float = 0.5,
    force_weight: float = 1.0,
    velocity_weight: float = 1.0,
    beta: float = 0.0,
    command_name: str | None = "base_velocity",
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Walking-only periodic reward composition from foot force and speed.

    The force component is active in swing and the speed component is active
    in stance, matching the paper's bipedal reward structure for walking.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    foot_force = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    foot_speed = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :], dim=-1)

    offsets = torch.tensor(offset, dtype=torch.float, device=env.device)
    if offsets.numel() != foot_force.shape[1]:
        raise RuntimeError(
            f"Expected {foot_force.shape[1]} gait offsets for periodic_bipedal_gait_reward, got {offsets.numel()}."
        )

    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    foot_phase = torch.remainder(global_phase.unsqueeze(1) + offsets.unsqueeze(0), 1.0)

    stance_ratio = min(max(stance_ratio, 1.0e-3), 1.0 - 1.0e-3)
    stance_prob = _periodic_interval_expectation(foot_phase, 0.0, stance_ratio, kappa)
    swing_prob = 1.0 - stance_prob

    force_measure = torch.tanh(foot_force / force_scale)
    speed_measure = torch.tanh(foot_speed / velocity_scale)
    expected_reward = -force_weight * swing_prob * force_measure - velocity_weight * stance_prob * speed_measure
    reward = beta + torch.exp(torch.mean(expected_reward, dim=1) / std)

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > command_threshold

    upright = torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward * upright


"""
Feet Gait rewards.
"""


'''
def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward
'''

def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward



def feet_contact_number(
    env: ManagerBasedRLEnv,
    period: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Isaac Lab 风格的步态接触数奖励。"""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    
    # return float mask 1 is stance, 0 is swing
    phase = global_phase
    sin_pos = torch.sin(2 * torch.pi * phase)
    # Add double support phase
    stance_mask = torch.zeros((env.num_envs, 2), device=env.device)
    # left foot stance
    stance_mask[:, 0] = sin_pos >= 0
    # right foot stance
    stance_mask[:, 1] = sin_pos < 0

    cmd_norm = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)

    stance_mask[:, 0][cmd_norm < 0.02] = 1
    stance_mask[:, 1][cmd_norm < 0.02] = 1

    contact = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]) > 5

    reward = torch.where(contact == stance_mask, 1.0, -0.3)

    return torch.mean(reward, dim=1)


"""
Other rewards.
"""


def joint_mirror(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    mirror_joints: list[list[str]],
    joint_weights: list[float] | None = None,
    alpha: float = 0.05,
    warmup_steps: int = 30,
) -> torch.Tensor:
    """Penalize left/right statistical asymmetry over a gait window."""
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
        n_pairs = len(mirror_joints)
        if joint_weights is None:
            joint_weights = [1.0] * n_pairs
        weights = torch.tensor(joint_weights, device=env.device, dtype=torch.float)
        env._joint_mirror_weights = weights / torch.clamp(torch.sum(weights), min=1.0e-6)
        env._joint_mirror_mean_l = torch.zeros(env.num_envs, n_pairs, device=env.device)
        env._joint_mirror_mean_r = torch.zeros(env.num_envs, n_pairs, device=env.device)
        env._joint_mirror_var_l = torch.zeros(env.num_envs, n_pairs, device=env.device)
        env._joint_mirror_var_r = torch.zeros(env.num_envs, n_pairs, device=env.device)
        env._joint_mirror_count = torch.zeros(env.num_envs, 1, device=env.device)

    reset_mask = (env.episode_length_buf < 2).unsqueeze(1)
    env._joint_mirror_mean_l = torch.where(reset_mask, torch.zeros_like(env._joint_mirror_mean_l), env._joint_mirror_mean_l)
    env._joint_mirror_mean_r = torch.where(reset_mask, torch.zeros_like(env._joint_mirror_mean_r), env._joint_mirror_mean_r)
    env._joint_mirror_var_l = torch.where(reset_mask, torch.zeros_like(env._joint_mirror_var_l), env._joint_mirror_var_l)
    env._joint_mirror_var_r = torch.where(reset_mask, torch.zeros_like(env._joint_mirror_var_r), env._joint_mirror_var_r)
    env._joint_mirror_count = torch.where(reset_mask, torch.zeros_like(env._joint_mirror_count), env._joint_mirror_count)

    env._joint_mirror_count += 1
    for i, joint_pair in enumerate(env.joint_mirror_joints_cache):
        pos_l = asset.data.joint_pos[:, joint_pair[0][0]].squeeze(-1)
        pos_r = asset.data.joint_pos[:, joint_pair[1][0]].squeeze(-1)
        env._joint_mirror_mean_l[:, i] = (1.0 - alpha) * env._joint_mirror_mean_l[:, i] + alpha * pos_l
        env._joint_mirror_mean_r[:, i] = (1.0 - alpha) * env._joint_mirror_mean_r[:, i] + alpha * pos_r
        env._joint_mirror_var_l[:, i] = (
            (1.0 - alpha) * env._joint_mirror_var_l[:, i]
            + alpha * torch.square(pos_l - env._joint_mirror_mean_l[:, i])
        )
        env._joint_mirror_var_r[:, i] = (
            (1.0 - alpha) * env._joint_mirror_var_r[:, i]
            + alpha * torch.square(pos_r - env._joint_mirror_mean_r[:, i])
        )

    mean_error = torch.square(env._joint_mirror_mean_l - env._joint_mirror_mean_r)
    std_error = torch.square(torch.sqrt(env._joint_mirror_var_l + 1.0e-8) - torch.sqrt(env._joint_mirror_var_r + 1.0e-8))
    reward = torch.sum((mean_error + std_error) * env._joint_mirror_weights.unsqueeze(0), dim=-1)
    reward *= (env._joint_mirror_count.squeeze(-1) >= warmup_steps).float()
    return reward


"""
Straight-line walking rewards.

These reward terms are designed to prevent diagonal drift and crab-walking
in bipedal locomotion, following reward design principles from:
- Berkeley Humanoid (2024): explicit lateral velocity penalty in body frame
- Cassie / UC Berkeley: yaw-rate tracking and heading alignment
"""


def lateral_velocity_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise lateral (y-axis) velocity error in the yaw-aligned body frame.

    Computes (v_y_body - v_y_cmd)^2, providing a strong un-saturated gradient
    that discourages sideways drift more aggressively than the exponential
    tracker in track_lin_vel_xy_yaw_frame_exp.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # Body-frame linear velocity: [:, 0]=x, [:, 1]=y, [:, 2]=z
    vel_y_body = asset.data.root_lin_vel_b[:, 1]
    # Command: [lin_vel_x, lin_vel_y, ang_vel_z]
    cmd = env.command_manager.get_command(command_name)
    vel_y_cmd = cmd[:, 1]
    return torch.square(vel_y_body - vel_y_cmd)


def heading_velocity_alignment(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    speed_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward alignment between the velocity direction and the robot heading.

    Uses the ratio v_x / |v_xy| in the body frame as a proxy for alignment.
    Returns +1 when perfectly aligned forward, 0 when perpendicular, -1 when
    walking backward.  Only active when the robot is actually moving
    (|v_xy| > speed_threshold) and a non-zero command is given.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    vel_xy = asset.data.root_lin_vel_b[:, :2]
    speed = torch.linalg.norm(vel_xy, dim=1)
    # Alignment: v_x / |v|, clamped to [-1, 1]
    alignment = vel_xy[:, 0] / (speed + 1e-6)
    alignment = torch.clamp(alignment, -1.0, 1.0)

    # Only reward when actually moving
    moving_mask = (speed > speed_threshold).float()
    # Only reward when a command is given
    cmd_norm = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    cmd_mask = (cmd_norm > 0.05).float()

    return alignment * moving_mask * cmd_mask


def base_yaw_rate_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise yaw angular velocity error from the commanded yaw rate.

    Computes (omega_z - omega_z_cmd)^2.  Prevents the robot from slowly
    rotating its heading, which manifests as diagonal walking when
    combined with forward velocity.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # Body-frame angular velocity: [:, 2] = yaw rate
    yaw_rate = asset.data.root_ang_vel_b[:, 2]
    # Command: [lin_vel_x, lin_vel_y, ang_vel_z]
    cmd = env.command_manager.get_command(command_name)
    yaw_rate_cmd = cmd[:, 2]
    return torch.square(yaw_rate - yaw_rate_cmd)
