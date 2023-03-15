# Lint as: python3
# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Functions to build evaluation metrics for airfoil data."""
import torch

from mesh_torch import common

device = torch.device('cuda')


def _rollout(model, initial_state, num_steps):
    """Rolls out a model trajectory."""
    node_type = initial_state['node_type']
    mask = torch.logical_or(torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device)),
                            torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OUTFLOW.value], device=device)))
    mask = torch.stack((mask, mask), dim=1)

    def step_fn(velocity, density, trajectory_v, trajectory_d):
        with torch.no_grad():
            prediction_v, prediction_d = model({**initial_state,
                                'velocity': velocity, 'density': density}, is_training=False)
            # don't update boundary nodes
        next_velocity = torch.where(mask, torch.squeeze(prediction_v), torch.squeeze(velocity))
        next_density = torch.where(mask, torch.squeeze(prediction_d), torch.squeeze(density))
        trajectory_v.append(velocity)
        trajectory_d.append(density)
        return next_velocity, trajectory_v, trajectory_d

    velocity = torch.squeeze(initial_state['velocity'], 0)
    density = torch.squeeze(initial_state['dnsity'], 0)
    trajectory_v = []
    trajectory_d = []
    for step in range(num_steps):
        velocity, trajectory_v, trajectory_d = step_fn(velocity, density, trajectory_v, trajectory_d)
    return torch.stack(trajectory_v), torch.stack(trajectory_d)


def evaluate(model, trajectory):
    """Performs model rollouts and create stats."""
    traj_squeeze = {k: torch.squeeze(v, 0) for k, v in trajectory.items()}
    initial_state = {k: torch.squeeze(v, 0)[0] for k, v in trajectory.items()}
    num_steps = traj_squeeze['cells'].shape[0]
    prediction_v, prediction_d = _rollout(model, initial_state, num_steps)

    error_v = torch.mean((prediction_v - traj_squeeze['velocity'])**2, axis=-1).cpu()
    error_d = torch.mean((prediction_d - traj_squeeze['density'])**2, axis=-1).cpu()
    scalars = {'mse_%d_steps' % horizon: [torch.mean(error_v[1:horizon+1]), torch.mean(error_d[1:horizon+1])]
                for horizon in [1, 10, 20, 50, 100, 200, len(error)-1]}

    traj_ops = {
        'faces': traj_squeeze['cells'],
        'mesh_pos': traj_squeeze['mesh_pos'],
        'gt_velocity': traj_squeeze['velocity'],
        'pred_velocity': prediction_v,
        'gt_density': traj_squeeze['density'],
        'pred_density': prediction_d
    }
    return scalars, traj_ops
