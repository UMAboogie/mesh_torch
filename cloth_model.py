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
"""Model for FlagSimple."""

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
    """Model for static cloth simulation."""

    def __init__(self, params, learned_model):
        super(Model, self).__init__()
        self._params = params
        self._output_normalizer = normalization.Normalizer(size=3, name='output_normalizer')
        self._node_normalizer = normalization.Normalizer(
            size=3 + common.NodeType.SIZE, name='node_normalizer')
        self._node_dynamic_normalizer = normalization.Normalizer(size=1, name='node_dynamic_normalizer')
        self._mesh_edge_normalizer = normalization.Normalizer(
            size=7, name='mesh_edge_normalizer')  # 2D coord + 3D coord + 2*length = 7
        self._world_edge_normalizer = normalization.Normalizer(size=4, name='world_edge_normalizer')
        self._model_type = params['model'].__name__
        self._learned_model = learned_model

    def unsorted_segment_operation(self, data, segment_ids, num_segments):
        """
        Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

        :param data: A tensor whose segments are to be summed.
        :param segment_ids: The segment indices tensor.
        :param num_segments: The number of segments.
        :return: A tensor of same data type as the data argument.
        """
        assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

        # segment_ids is a 1-D tensor repeat it to have the same shape as data
        if len(segment_ids.shape) == 1:
            s = torch.prod(torch.tensor(data.shape[1:])).long().to(device)
            segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:]).to(device)

        assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

        shape = [num_segments] + list(data.shape[1:])
        result = torch.zeros(*shape)
        result = torch_scatter.scatter_add(data.float(), segment_ids, dim=0, dim_size=num_segments)
        result = result.type(data.dtype)
        return result

    def _build_graph(self, inputs, is_training):
        """Builds input graph."""
        world_pos = inputs['world_pos']
        prev_world_pos = inputs['prev_world_pos']
        node_type = inputs['node_type']
        velocity = world_pos - prev_world_pos
        one_hot_node_type = F.one_hot(node_type[:, 0].to(torch.int64), common.NodeType.SIZE)

        node_features = torch.cat((velocity, one_hot_node_type), dim=-1)

        cells = inputs['cells']
        decomposed_cells = common.triangles_to_edges(cells)
        senders, recievers = decomposed_cells['two_way_connectivity']


        # find world edge
        radius = 0.03
        world_distance_matrix = torch.cdist(world_pos, world_pos, p=2)
        # print("----------------------------------")
        # print(torch.nonzero(world_distance_matrix).shape[0])
        world_connection_matrix = torch.where(world_distance_matrix < radius, True, False)
        # print(torch.nonzero(world_connection_matrix).shape[0])
        # remove self connection
        world_connection_matrix = world_connection_matrix.fill_diagonal_(False)
        # print(torch.nonzero(world_connection_matrix).shape[0])
        # remove world edge node pairs that already exist in mesh edge collection
        world_connection_matrix[senders, recievers] = torch.tensor(False, dtype=torch.bool, device=device)

        # remove recievers whose node type is obstacle
        no_connection_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
        no_connection_mask_t = torch.transpose(torch.stack([no_connection_mask] * world_pos.shape[0], dim=1), 0, 1)
        world_connection_matrix = torch.where(no_connection_mask_t, torch.tensor(False, dtype=torch.bool, device=device), world_connection_matrix)
        
        world_senders, world_recievers = torch.nonzero(world_connection_matrix, as_tuple=True)
        relative_world_pos = (torch.index_select(input=world_pos, dim=0, index=world_recievers) -
                              torch.index_select(input=world_pos, dim=0, index=world_senders))
        world_edge_features = torch.cat((
            relative_world_pos,
            torch.norm(relative_world_pos, dim=-1, keepdim=True)), dim=-1)
        world_edges = core_model.EdgeSet(
            name='world_edges',
            features=self._world_edge_normalizer(world_edge_features, is_training),
            recievers=world_recievers,
            senders=world_senders)


        mesh_pos = inputs['mesh_pos']
        relative_world_pos = (torch.index_select(input=world_pos, dim=0, index=senders) -
                              torch.index_select(input=world_pos, dim=0, index=recievers))
        relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                             torch.index_select(mesh_pos, 0, recievers))
        edge_features = torch.cat((
            relative_world_pos,
            torch.norm(relative_world_pos, dim=-1, keepdim=True),
            relative_mesh_pos,
            torch.norm(relative_mesh_pos, dim=-1, keepdim=True)), dim=-1)

        mesh_edges = core_model.EdgeSet(
            name='mesh_edges',
            features=self._mesh_edge_normalizer(edge_features, is_training),
            recievers=recievers,
            senders=senders)

        
        return (core_model.MultiGraph(node_features=self._node_normalizer(node_features),
                                            edge_sets=[mesh_edges, world_edges]))

    def forward(self, inputs, is_training):
        graph = self._build_graph(inputs, is_training=is_training)
        if is_training:
            return self._learned_model(graph,
                                      world_edge_normalizer=self._world_edge_normalizer, is_training=is_training)
        else:
            return self._update(inputs, self._learned_model(graph,
                                                           world_edge_normalizer=self._world_edge_normalizer,
                                                           is_training=is_training))

    def loss(self, inputs):
        """L2 loss on position"""
        graph = self._build_graph(inputs, is_training=True)
        network_output = self._learned_model(graph, is_training=True)

        cur_position = inputs['world_pos']
        prev_position = inputs['prev_world_pos']
        target_position = inputs['target_world_pos']
        target_acceleration = target_position - 2 * cur_position + prev_position
        target_normalized = self._output_normalizer(target_acceleration).to(device)

        # build loss
        loss_mask = torch.eq(inputs['node_type'][:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device).int())
        error = torch.sum((target_normalized - network_output) ** 2, dim=1)
        loss = torch.mean(error[loss_mask])
        return loss



    def _update(self, inputs, per_node_network_output):
        """Integrate model outputs."""

        acceleration = self._output_normalizer.inverse(per_node_network_output)

        # integrate forward
        cur_position = inputs['world_pos']
        prev_position = inputs['prev_world_pos']
        position = 2 * cur_position + acceleration - prev_position
        return position

    def get_output_normalizer(self):
        return self._output_normalizer

    def save_model(self, path, epoch=None):
        if epoch == None: 
            #torch.save(self._learned_model, path + "_learned_model.pth")
            torch.save(self._output_normalizer, path + "/_output_normalizer.pth")
            torch.save(self._world_edge_normalizer, path + "/_world_edge_normalizer.pth")
            torch.save(self._mesh_edge_normalizer, path + "/_mesh_edge_normalizer.pth")
            torch.save(self._node_normalizer, path + "/_node_normalizer.pth")
            torch.save(self._node_dynamic_normalizer, path + "/_node_dynamic_normalizer.pth")
        else:
            #torch.save(self._learned_model, path + "_learned_model.pth")
            torch.save(self._output_normalizer, path + "/_output_normalizer"+str(epoch)+".pth")
            torch.save(self._world_edge_normalizer, path + "/_world_edge_normalizer"+str(epoch)+".pth")
            torch.save(self._mesh_edge_normalizer, path + "/_mesh_edge_normalizer"+str(epoch)+".pth")
            torch.save(self._node_normalizer, path + "/_node_normalizer"+str(epoch)+".pth")
            torch.save(self._node_dynamic_normalizer, path + "/_node_dynamic_normalizer"+str(epoch)+".pth")

    def load_model(self, path, epoch=None):
        if epoch == None: 
            #self._learned_model = torch.load(path + "_learned_model.pth")
            self._output_normalizer = torch.load(path + "/_output_normalizer.pth")
            self._world_edge_normalizer = torch.load(path + "/_world_edge_normalizer.pth")
            self._mesh_edge_normalizer = torch.load(path + "/_mesh_edge_normalizer.pth")
            self._node_normalizer = torch.load(path + "/_node_normalizer.pth")
            self._node_dynamic_normalizer = torch.load(path + "/_node_dynamic_normalizer.pth")
        else:
            #self._learned_model = torch.load(path + "_learned_model.pth")
            self._output_normalizer = torch.load(path + "/_output_normalizer"+str(epoch)+".pth")
            self._world_edge_normalizer = torch.load(path + "/_world_edge_normalizer"+str(epoch)+".pth")
            self._mesh_edge_normalizer = torch.load(path + "/_mesh_edge_normalizer"+str(epoch)+".pth")
            self._node_normalizer = torch.load(path + "/_node_normalizer"+str(epoch)+".pth")    
            self._node_dynamic_normalizer = torch.load(path + "/_node_dynamic_normalizer"+str(epoch)+".pth")

    def evaluate(self):
        self.eval()
        self._learned_model.eval()
