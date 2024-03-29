"""Model for DeformingPlate."""

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
        #self._stress_output_normalizer = normalization.Normalizer(size=3, name='stress_output_normalizer')
        self._node_normalizer = normalization.Normalizer(size=9, name='node_normalizer')
        self._node_dynamic_normalizer = normalization.Normalizer(size=1, name='node_dynamic_normalizer')
        self._mesh_edge_normalizer = normalization.Normalizer(size=8, name='mesh_edge_normalizer')
        self._world_edge_normalizer = normalization.Normalizer(size=4, name='world_edge_normalizer')
        self._model_type = params['model'].__name__
        self._displacement_base = None

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

        node_type = inputs['node_type']

        one_hot_node_type = F.one_hot(node_type[:, 0].to(torch.int64), common.NodeType.SIZE).float()

        cells = inputs['cells']
        decomposed_cells = common.triangles_to_edges(cells, deform=True)
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
        # only obstacle and handle node as sender and normal node as reciever
        '''no_connection_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
        no_connection_mask = torch.logical_or(no_connection_mask, torch.eq(node_type[:, 0], torch.tensor([common.NodeType.HANDLE.value], device=device)))
        no_connection_mask = torch.stack([no_connection_mask] * world_pos.shape[0], dim=1)
        no_connection_mask_t = torch.transpose(no_connection_mask, 0, 1)
        world_connection_matrix = torch.where(no_connection_mask_t, torch.tensor(0., dtype=torch.float32, device=device),
                                              world_connection_matrix)
        world_connection_matrix = torch.where(no_connection_mask, world_connection_matrix, torch.tensor(0., dtype=torch.float32, device=device))'''

        # remove recievers whose node type is obstacle
        no_connection_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
        no_connection_mask_t = torch.transpose(torch.stack([no_connection_mask] * world_pos.shape[0], dim=1), 0, 1)
        world_connection_matrix = torch.where(no_connection_mask_t, torch.tensor(False, dtype=torch.bool, device=device), world_connection_matrix)
        # remove senders whose node type is handle and normal
        connection_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
        connection_mask = torch.stack([no_connection_mask] * world_pos.shape[0], dim=1)
        world_connection_matrix = torch.where(connection_mask, world_connection_matrix, torch.tensor(False, dtype=torch.bool, device=device))
        '''no_connection_mask_t = torch.transpose(torch.stack([no_connection_mask] * world_pos.shape[0], dim=1), 0, 1)
        world_connection_matrix = torch.where(no_connection_mask_t,
                                              torch.tensor(0., dtype=torch.float32, device=device),
                                              world_connection_matrix)'''
        '''world_connection_matrix = torch.where(no_connection_mask,
                                              torch.tensor(0., dtype=torch.float32, device=device),
                                              world_connection_matrix)'''
        # remove senders whose type is normal or handle
        '''no_connection_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device))
        no_connection_mask = torch.logical_or(no_connection_mask, torch.eq(node_type[:, 0], torch.tensor([common.NodeType.HANDLE.value], device=device)))
        no_connection_mask = torch.stack([no_connection_mask] * world_pos.shape[0], dim=1)
        world_connection_matrix = torch.where(no_connection_mask, torch.tensor(0., dtype=torch.float32, device=device),
                                              world_connection_matrix)'''
        # select the closest sender
        '''world_distance_matrix = torch.where(world_connection_matrix, world_distance_matrix, torch.tensor(float('inf'), device=device))
        min_values, indices = torch.min(world_distance_matrix, 1)
        world_senders = torch.arange(0, world_pos.shape[0], dtype=torch.int32, device=device)
        world_s_r_tuple = torch.stack((world_senders, indices), dim=1)
        world_senders_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
        world_senders_mask_value = torch.logical_not(torch.isinf(min_values))
        world_senders_mask = torch.logical_and(world_senders_mask, world_senders_mask_value)
        world_s_r_tuple = world_s_r_tuple[world_senders_mask]
        world_senders, world_recievers = torch.unbind(world_s_r_tuple, dim=1)'''
        # print(world_senders.shape[0])
        world_senders, world_recievers = torch.nonzero(world_connection_matrix, as_tuple=True)

        relative_world_pos = (torch.index_select(input=world_pos, dim=0, index=world_recievers) -
                              torch.index_select(input=world_pos, dim=0, index=world_senders))

        '''relative_world_velocity = (torch.index_select(input=inputs['target|world_pos'], dim=0, index=world_senders) -
                              torch.index_select(input=inputs['world_pos'], dim=0, index=world_senders))'''


        world_edge_features = torch.cat((
            relative_world_pos,
            torch.norm(relative_world_pos, dim=-1, keepdim=True)), dim=-1)

        '''world_edge_features = torch.cat((
            relative_world_pos,
            torch.norm(relative_world_pos, dim=-1, keepdim=True),
            relative_world_velocity,
            torch.norm(relative_world_velocity, dim=-1, keepdim=True)), dim=-1)'''

        world_edges = core_model.EdgeSet(
            name='world_edges',
            features=self._world_edge_normalizer(world_edge_features, is_training),
            recievers=world_recievers,
            senders=world_senders)


        mesh_pos = inputs['mesh_pos']
        relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                             torch.index_select(mesh_pos, 0, recievers))
        all_relative_world_pos = (torch.index_select(input=world_pos, dim=0, index=senders) -
                              torch.index_select(input=world_pos, dim=0, index=recievers))
        mesh_edge_features = torch.cat((
            relative_mesh_pos,
            torch.norm(relative_mesh_pos, dim=-1, keepdim=True),
            all_relative_world_pos,
            torch.norm(all_relative_world_pos, dim=-1, keepdim=True)), dim=-1)

        mesh_edges = core_model.EdgeSet(
            name='mesh_edges',
            features=self._mesh_edge_normalizer(mesh_edge_features, is_training),
            # features=mesh_edge_features,
            recievers=recievers,
            senders=senders)

        '''obstacle_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
        obstacle_mask = torch.stack([obstacle_mask] * 3, dim=1)
        masked_target_world_pos = torch.where(obstacle_mask, target_world_pos, torch.tensor(0., dtype=torch.float32, device=device))
        masked_world_pos = torch.where(obstacle_mask, world_pos, torch.tensor(0., dtype=torch.float32, device=device))
        # kinematic_nodes_features = self._node_normalizer(masked_target_world_pos - masked_world_pos)
        kinematic_nodes_features = masked_target_world_pos - masked_world_pos
        normal_node_features = torch.cat((torch.zeros_like(world_pos), one_hot_node_type), dim=-1)
        kinematic_node_features = torch.cat((kinematic_nodes_features, one_hot_node_type), dim=-1)
        obstacle_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
        obstacle_mask = torch.stack([obstacle_mask] * 12, dim=1)
        node_features = torch.where(obstacle_mask, kinematic_node_features, normal_node_features)'''
        node_features = one_hot_node_type

        
        return (core_model.MultiGraph(node_features=node_features,
                                          edge_sets=[mesh_edges, world_edges]))

    def forward(self, inputs, is_training):
        graph = self._build_graph(inputs, is_training=is_training)
        if is_training:
            return self._learned_model(graph, world_edge_normalizer=self._world_edge_normalizer, is_training=is_training)
        else:
            return self._update(inputs, self._learned_model(graph, world_edge_normalizer=self._world_edge_normalizer, is_training=is_training))


    def loss(self, inputs):
        """L2 loss on position"""

        graph = self._build_graph(inputs, is_training=True)
        network_output = self._learned_model(graph, is_training=True)

        cur_position = inputs['world_pos']
        target_position = inputs['target_world_pos']
        target_velocity = target_position - cur_position

        node_type = inputs['node_type']
        '''scripted_node_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device))
        scripted_node_mask = torch.logical_not(scripted_node_mask)
        scripted_node_mask = torch.stack([scripted_node_mask] * 3, dim=1)
        target_velocity = torch.where(scripted_node_mask, torch.tensor(0., device=device), target_velocity)'''

        target_normalized = self._output_normalizer(target_velocity).to(device)
        
        '''node_type = inputs['node_type']
        scripted_node_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
        scripted_node_mask = torch.stack([scripted_node_mask] * 3, dim=1)
        target_normalized = torch.where(scripted_node_mask, torch.tensor(0., device=device), target_normalized)'''

        # build loss
        node_type = inputs['node_type']
        loss_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device).int())
        error = torch.sum((target_normalized - network_output) ** 2, dim=1)
        loss = torch.mean(error[loss_mask])

        return loss

    def _update(self, inputs, per_node_network_output):
        """Integrate model outputs."""
        '''output_mask = torch.eq(inputs['node_type'][:, 0], torch.tensor([common.NodeType.NORMAL.value], device=device))
        output_mask = torch.stack([output_mask] * inputs['world_pos'].shape[-1], dim=1)
        velocity = self._output_normalizer.inverse(torch.where(output_mask, per_node_network_output, torch.tensor(0., device=device)))'''
        velocity = self._output_normalizer.inverse(per_node_network_output)
        #stress = self._stress_output_normalizer.inverse(per_node_network_output)

        node_type = inputs['node_type']
        '''scripted_node_mask = torch.eq(node_type[:, 0], torch.tensor([common.NodeType.OBSTACLE.value], device=device))
        scripted_node_mask = torch.stack([scripted_node_mask] * 3, dim=1)'''

        # integrate forward
        cur_position = inputs['world_pos']
        position = cur_position + velocity
        # position = torch.where(scripted_node_mask, position + inputs['target|world_pos'] - inputs['world_pos'], position)
        return (position, cur_position, velocity)

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
