"""Runs the learner/evaluator."""
import sys
import os
import pathlib
from pathlib import Path

import pickle
from absl import app
from absl import flags
import h5py

import torch

from mesh_torch import core_model
from mesh_torch import cloth_model
from mesh_torch import cloth_eval
from mesh_torch import cfd_model
from mesh_torch import cfd_eval
from mesh_torch import deform_model
from mesh_torch import deform_eval
from mesh_torch import air_model
from mesh_torch import air_eval

from mesh_torch import common
from mesh_torch import dataset
import logging

import numpy as np
import json

import time
import datetime
import functools

import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt

device = torch.device('cuda')

# train and evaluation configuration
FLAGS = flags.FLAGS
flags.DEFINE_enum('model', 'cloth', ['cloth', 'deform', 'air', 'cfd'],
                  'Select model to run.')
flags.DEFINE_enum('mode', 'all', ['train', 'eval', 'all'],
                  'Train model, or run evaluation, or run both.')
flags.DEFINE_enum('rollout_split', 'valid', ['train', 'test', 'valid'],
                  'Dataset split to use for rollouts.')
flags.DEFINE_string('checkpoint_dir', None, 'Directory to save checkpoint')
flags.DEFINE_string('dataset_dir', None, 'Directory to load dataset from.')
flags.DEFINE_string('rollout_dir', None, 'Pickle file to save eval trajctories')
flags.DEFINE_boolean('load_chk', False, 'whether to load checkpoints')

flags.DEFINE_integer('max_epochs', 1, 'No. of max training epochs')
#flags.DEFINE_integer('num_training_steps', int(10e6), 'No. of training steps')
flags.DEFINE_integer('trajectories', 4, 'No. of training trajectories')
flags.DEFINE_integer('num_rollouts', 1, 'No. of rollout trajectories')

flags.DEFINE_integer('message_passing_steps', 5, 'No. of message passing steps')


PARAMETERS = {
    'cfd': dict(traj_num=[500,50,50], noise=0.02, gamma=1.0, field='velocity', history=False,
                size=2, batch=2, model=cfd_model, evaluator=cfd_eval, loss_type='cfd'),
    'cloth': dict(traj_num=[500,50,50], noise=0.003, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, model=cloth_model, evaluator=cloth_eval, loss_type='cloth'),
    'deform': dict(traj_num=[600,50,50], noise=0.003, gamma=1.0, field='world_pos', history=False,
                  size=3, batch=2, model=deform_model, evaluator=deform_eval, loss_type='deform'),
    'air': dict(traj_num=[500,50,50], noise=10.0, gamma=1.0, field='velocity', history=False,
                  size=3, batch=2, model=air_model, evaluator=air_eval)
}

"""
PARAMETERS = {
    'cloth': dict(traj_num=[500,50,50], noise=0.003, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, model=cloth_model, evaluator=cloth_eval, loss_type='cloth')
}
"""


def pickle_save(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def pickle_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def split_and_preprocess(trajectory_data, field, scale, gamma, add_noise_bool=True, split_bool=True):
    def add_noise(frame):
        zero_size = torch.zeros(frame[field].size(), dtype=torch.float32).to(device)
        noise = torch.normal(zero_size, std=scale).to(device)
        other = torch.Tensor([common.NodeType.NORMAL.value]).to(device)
        mask = torch.eq(frame['node_type'], other.int())[:, 0]
        mask_sequence = []
        for i in range(noise.shape[1]):
            mask_sequence.append(mask)
        mask = torch.stack(mask_sequence, dim=1)
        noise = torch.where(mask, noise, torch.zeros_like(noise))
        frame[field] += noise
        frame['target_' + field] += (1.0 - gamma) * noise
        return frame
    
    def element_operation(trajectory):
        trajectory_steps = []
        for i in range(len(trajectory["node_type"])):
            trajectory_step = {}
            for key, value in trajectory.items():
                trajectory_step[key] = value[i]
            if add_noise_bool:
                trajectory_step = add_noise(trajectory_step)
            trajectory_steps.append(trajectory_step)
        return trajectory_steps

        
    traj_tensor = {}
    for key, feature in trajectory_data.items():
        feature_tensor = torch.from_numpy(feature).to(device)
        traj_tensor[key] = feature_tensor
    if split_bool:
        traj_tensor = element_operation(traj_tensor)
            
    return traj_tensor

def load_dataset(path, split):
    fp = h5py.File(os.path.join(path, split + ".h5"), 'r')
    return fp

def squeeze_data_frame(data_frame):
    for key, value in data_frame.items():
        data_frame[key] = torch.squeeze(value,0)
    return data_frame

def logger_setup(log_path):
    # set log configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # console_output_handler = logging.StreamHandler(sys.stdout)
    # console_output_handler.setLevel(logging.INFO)
    file_log_handler = logging.FileHandler(filename=log_path, mode='w', encoding='utf-8')
    file_log_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(message)s')
    # console_output_handler.setFormatter(formatter)
    file_log_handler.setFormatter(formatter)
    # root_logger.addHandler(console_output_handler)
    root_logger.addHandler(file_log_handler)
    return root_logger

def log_run_summary(root_logger):
    root_logger.info("")
    root_logger.info("=======================Run Summary=======================")
    root_logger.info("Simulation task is " + FLAGS.model + " simulation")
    root_logger.info("Mode is " + FLAGS.mode)
    if FLAGS.mode == 'eval' or FLAGS.mode == 'all':
        root_logger.info("Evaluation set is " + FLAGS.rollout_split)
    else:
        root_logger.info("No Evaluation")
    root_logger.info(
        "Train and/or evaluation configuration are " + str(FLAGS.max_epochs) + " epochs, " + str(
            FLAGS.trajectories) + " trajectories each epoch, number of rollouts is " + str(
            FLAGS.num_rollouts))
    root_logger.info("=========================================================")
    root_logger.info("")


def learner(params, config):
    root_logger = logging.getLogger()
    """Run a learner job."""
    #ds = load_dataset(FLAGS.dataset_dir, 'train')
    ds = dataset.trajDataset(FLAGS.dataset_dir, 'train', transform=functools.partial(split_and_preprocess, field=params['field'], scale=params['noise'], gamma=params['gamma'], 
                                                                                                                                add_noise_bool=True, split_bool=True))
    data_loader = torch.utils.data.DataLoader(ds, batch_size=1)
    # model training
    is_training = True

    epoch_run_times = []
    all_trajectory_train_losses = []
    epoch_training_losses = []
    count = 0
    length_sum = 0


    for epoch in range(FLAGS.max_epochs):
                    
        current_time = time.time()
        # every time when model.train is called, model will train itself with the whole dataset
        print("Epoch " + str(epoch + 1) + "/" + str(FLAGS.max_epochs))
        
        for trajectory_index, trajectory in enumerate(data_loader):
            print(
                "    trajectory index " + str(trajectory_index + 1))
            #trajectory = ds[str(trajectory_index)]
            #trajectory = split_and_preprocess(trajectory, params['field'], params['noise'], params['gamma'], add_noise_bool=True)
            for data_frame_index, data_frame in enumerate(trajectory):
                if data_frame_index >= 1:
                    break
                count += 1
                data_frame = squeeze_data_frame(data_frame)
                cells = data_frame['cells']
                decomposed_cells = common.triangles_to_edges(cells)
                senders, recievers = decomposed_cells['two_way_connectivity']
                length = len(senders)
                print("Traj " + str(trajectory_index + 1) + ":" + str(length))
                length_sum += length

                """
                loss = model.loss(data_frame)
                if count % 1000 == 0:
                    root_logger.info("    1000 step loss " + str(loss))
                if pass_count > 0:
                    pass_count -= 1
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    trajectory_loss += loss.detach().cpu()
                """
        
    return int(length_sum / count)



def main(argv):
    del argv
    params = PARAMETERS[FLAGS.model]
    config = {}

    # record start time
    run_step_start_time = time.time()
    run_step_start_datetime = datetime.datetime.fromtimestamp(run_step_start_time).strftime('%c')

    
    config = None

    train_loss_record = None
    if FLAGS.mode == 'train' or FLAGS.mode == 'all':
        print(learner(params, config))


   






    

if __name__ == '__main__':
  app.run(main)

