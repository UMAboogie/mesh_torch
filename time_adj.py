"""Core learned graph net model."""

import collections
from math import ceil
from collections import OrderedDict
import functools
import torch
from torch import nn as nn
import torch_scatter
from torch_scatter.composite import scatter_softmax
import torch.nn.functional as F
import time
import datetime

from absl import app

EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders',
                                             'recievers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])
MultiGraphWithPos = collections.namedtuple('Graph', ['node_features', 'edge_sets', 'target_feature', 'model_type', 'node_dynamic'])

device = torch.device('cuda')


def main(argv):
    del argv

    start_time = time.time()
    #run_step_start_datetime = datetime.datetime.fromtimestamp(start_time).strftime('%c')
    num_nodes = 1885
    adj = torch.zeros(num_nodes, num_nodes).bool().to(device)
    #senders = torch.range(0, 6, dtype=torch.int64).repeat_interleave(1885).to(device)
    #recievers = torch.range(0, 6, dtype=torch.int64).repeat(1885).to(device)
    #adj[senders, recievers] = torch.tensor(True, dtype=torch.bool, device=device)
    end_time = time.time()
    #elapsed_time_in_second = end_time - start_time
    #elapsed_time = str(datetime.timedelta(seconds=elapsed_time_in_second))
    print(end_time - start_time)

if __name__ == '__main__':
  app.run(main)