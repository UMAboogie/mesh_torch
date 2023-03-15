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
"""Model for CylinderFlow."""

import torch
from torch import nn as nn
import torch.nn.functional as F
import functools

import torch_scatter
from mesh_torch import common
from mesh_torch import normalization
from mesh_torch import core_model

device = torch.device('cuda')


class Model(nn.Module):
    """Model for fluid simulation."""

    def __init__(self, params, learned_model):
        super(Model, self).__init__()
        self._params = params
        self._output_normalizer = normalization.Normalizer(size=2, name='output_normalizer')
        self._node_normalizer = normalization.Normalizer(size=2 + common.NodeType.SIZE, name='node_normalizer')
        self._edge_normalizer = normalization.Normalizer(size=3, name='edge_normalizer')  # 2D coord + length
        self._model_type = params['model'].__name__       
        self._learned_model = learned_model


    def _build_graph(self, inputs, is_training):
        """Builds input graph."""
        node_type = inputs['node_type']
        velocity = inputs['velocity']
        node_type = F.one_hot(node_type[:, 0].to(torch.int64), common.NodeType.SIZE)

        node_features = torch.cat((velocity, node_type), dim=-1)

        cells = inputs['cells']
        decomposed_cells = common.triangles_to_edges(cells)
        senders, recievers = decomposed_cells['two_way_connectivity']

        mesh_pos = inputs['mesh_pos']
        relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                             torch.index_select(mesh_pos, 0, recievers))
        edge_features = torch.cat([
            relative_mesh_pos,
            torch.norm(relative_mesh_pos, dim=-1, keepdim=True)], dim=-1)

        mesh_edges = core_model.EdgeSet(
            name='mesh_edges',
            features=self._edge_normalizer(edge_features, is_training),
            recievers=recievers,
            senders=senders)

        return core_model.MultiGraph(node_features=self._node_normalizer(node_features), edge_sets=[mesh_edges])
    
    def forward(self, inputs, is_training):
        graph = self._build_graph(inputs, is_training=is_training)
        if is_training:
            return self._learned_model(graph, world_edge_normalizer=self._edge_normalizer, is_training=is_training)
        else:
            return self._update(inputs, self._learned_model(graph, world_edge_normalizer=self._edge_normalizer, is_training=is_training))

    def loss(self, inputs):
        """L2 loss on position"""
        graph = self._build_graph(inputs, is_training=True)
        network_output = self._learned_model(graph, is_training=True)

        cur_velocity = inputs['velocity']
        target_velocity = inputs['target_velocity']
        target_velocity_change = target_velocity - cur_velocity
        target_normalized = self._output_normalizer(target_velocity_change).to(device)

        # build loss
        node_type = inputs['node_type'][:, 0]
        loss_mask = torch.logical_or(torch.eq(node_type, torch.tensor([common.NodeType.NORMAL.value], device=device).int()),
                                    torch.eq(node_type, torch.tensor([common.NodeType.OUTFLOW.value], device=device).int()))
        error = torch.sum((target_normalized - network_output) ** 2, dim=1)
        loss = torch.mean(error[loss_mask])
        return loss
    

    def _update(self, inputs, per_node_network_output):
        """Integrate model outputs."""
        #print(torch.mean(per_node_network_output**2))
        velocity_update = self._output_normalizer.inverse(per_node_network_output)
        #print(torch.mean(velocity_update**2))
        # integrate forward
        cur_velocity = inputs['velocity']
        return cur_velocity + velocity_update

    def get_output_normalizer(self):
        return self._output_normalizer

    def save_model(self, path, epoch=None):
        if epoch == None:
            #torch.save(self._learned_model, path + "_learned_model.pth")
            torch.save(self._output_normalizer, path + "/_output_normalizer.pth")
            torch.save(self._mesh_normalizer, path + "/_edge_normalizer.pth")
            #torch.save(self._sender_normalizer, path + "/_sender_normalizer.pth")
            torch.save(self._node_normalizer, path + "/_node_normalizer.pth")
        else:
            torch.save(self._output_normalizer, path + "/_output_normalizer"+str(epoch)+".pth")
            torch.save(self._mesh_normalizer, path + "/_mesh_normalizer"+str(epoch)+".pth")
            torch.save(self._node_normalizer, path + "/_node_normalizer"+str(epoch)+".pth")

    def load_model(self, path, epoch=None):
        if epoch == None:
            #self._learned_model = torch.load(path + "_learned_model.pth")
            self._output_normalizer = torch.load(path + "/_output_normalizer.pth")
            self._mesh_normalizer = torch.load(path + "/_edge_normalizer.pth")
            #self._sender_normalizer = torch.load(path + "/_sender_normalizer.pth")
            self._node_normalizer = torch.load(path + "/_node_normalizer.pth")
        else:
            #self._learned_model = torch.load(path + "_learned_model.pth")
            self._output_normalizer = torch.load(path + "/_output_normalizer"+str(epoch)+".pth")
            self._mesh_normalizer = torch.load(path + "/_edge_normalizer"+str(epoch)+".pth")
            #self._sender_normalizer = torch.load(path + "/_sender_normalizer.pth")
            self._node_normalizer = torch.load(path + "/_node_normalizer"+str(epoch)+".pth")
    
    def evaluate(self):
        self.eval()
        self._learned_model.eval()
