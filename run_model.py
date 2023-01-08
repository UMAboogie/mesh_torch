"""Runs the learner/evaluator."""
import sys
import os
import pathlib
from pathlib import Path

import pickle
from absl import app
from absl import flags

import torch

import cloth_model
import cloth_eval
import cfd_model
import cfd_eval
import deform_model
import deform_eval
import air_model
import air_eval

import dataset 
import common
import logging

import numpy as np
import json
from common import NodeType

import time
import datetime


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
flags.DEFINE_boolean('load_chk', False, 'whether to load checkpoints')

flags.DEFINE_integer('max_epochs', 2, 'No. of max training epochs')
#flags.DEFINE_integer('num_training_steps', int(10e6), 'No. of training steps')
flags.DEFINE_integer('trajectories', 2, 'No. of training trajectories')
flags.DEFINE_integer('num_rollouts', 1, 'No. of rollout trajectories')

flags.DEFINE_integer('message_passing_steps', 5, 'No. of message passing steps')


PARAMETERS = {
    'cfd': dict(traj_num=[500,50,50], noise=0.02, gamma=1.0, field='velocity', history=False,
                size=2, batch=2, model=cfd_model, evaluator=cfd_eval, loss_type='cfd'),
    'cloth': dict(traj_num=[500,50,50], noise=0.003, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, model=cloth_model, evaluator=cloth_eval, loss_type='cloth'),
    'deform': dict(traj_num=[600,50,50], noise=0.003, gamma=1.0, field='world_pos', history=False,
                  size=3, batch=2, model=deform_model, evaluator=deform_eval, loss_type='deform')
    'air': dict(traj_num=[500,50,50], noise=[10.0, 0.01], gamma=1.0, field=['velocity', 'density'], history=False,
                  size=3, batch=2, model=air_model, evaluator=air_eval)
}

loaded_meta = False
shapes = {}
dtypes = {}
types = {}
steps = None


def learner(model, params, config):
    """Run a learner job."""
    ds = dataset.load_dataset(FLAGS.dataset_dir, 'train')
    ds = dataset.add_targets(ds, [params['field']], add_history=params['history'])
    ds = dataset.split_and_preprocess(ds, noise_field=params['field'],
                                        noise_scale=params['noise'],
                                        noise_gamma=params['gamma'])
    inputs = tf.data.make_one_shot_iterator(ds).get_next()


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1 + 1e-6)
    trained_epoch = 0
    count = 0
    pass_count = 500
    if FLAGS.load_chk:
        optimizer.load_state_dict(
            torch.load(config['model']['optimizer']))
        scheduler.load_state_dict(
            torch.load(config['model']['scheduler']))
        epoch_checkpoint = torch.load(
            config['model']['epoch_checkpoint'])
        count = torch.load(
            config['model']['count'])
        trained_epoch = epoch_checkpoint['epoch'] + 1
        root_logger.info("Loaded optimizer, scheduler and model epoch checkpoint\n")
        pass_count = 500 - count if count < 500 else 0

    # model training
    is_training = True

    epoch_run_times = []
    epoch_training_losses = []


    for epoch in range(FLAGS.max_epochs)[trained_epoch:]:
                    
        current_time = time.time()
        # every time when model.train is called, model will train itself with the whole dataset
        root_logger.info("Epoch " + str(epoch + 1) + "/" + str(FLAGS.max_epochs))
        epoch_training_loss = 0.0
        ds_iterator = iter(ds_loader)
        
        for trajectory_index in range(params['traj_num'][0]):
            root_logger.info(
                "    trajectory index " + str(trajectory_index + 1) + "/" + str(params['traj_num'][0]))
            trajectory = next(ds_iterator)
            trajectory = dataset.split_and_preprocess(params)(trajectory)
            trajectory_loss = 0.0
            for data_frame_index, data_frame in enumerate(trajectory):
                count += 1
                data_frame = squeeze_data_frame(data_frame)
                network_output = model(data_frame, is_training)
                loss = model.loss(data_frame, network_output)
                if count % 1000 == 0:
                    root_logger.info("    1000 step loss " + str(loss))
                if pass_count > 0:
                    pass_count -= 1
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    trajectory_loss += loss.detach().cpu()
            all_trajectory_train_losses.append(trajectory_loss)
            epoch_training_loss += trajectory_loss
            root_logger.info("        trajectory_loss")
            root_logger.info("        " + str(trajectory_loss))
            model.save_model(
                os.path.join(FLAGS.checkpoint_dir,
                             "trajectory_model_checkpoint"))
            torch.save(optimizer.state_dict(),
                       os.path.join(FLAGS.checkpoint_dir,
                                    "trajectory_optimizer_checkpoint" + ".pth"))
            torch.save(scheduler.state_dict(),
                       os.path.join(FLAGS.checkpoint_dir,
                                    "trajectory_scheduler_checkpoint" + ".pth"))
        epoch_training_losses.append(epoch_training_loss)
        root_logger.info("Current mean of epoch training losses")
        root_logger.info(torch.mean(torch.stack(epoch_training_losses)))
        model.save_model(
            os.path.join(FLAGS.checkpoint_dir,
                         "epoch_model_checkpoint"))
        torch.save(optimizer.state_dict(),
                   os.path.join(FLAGS.checkpoint_dir,
                                "epoch_optimizer_checkpoint" + ".pth"))
        torch.save(scheduler.state_dict(),
                   os.path.join(FLAGS.checkpoint_dir,
                                "epoch_scheduler_checkpoint" + ".pth"))
        if epoch == 13:
            scheduler.step()
            root_logger.info("Call scheduler in epoch " + str(epoch))
        torch.save({'epoch': epoch}, os.path.join(FLAGS.checkpoint_dir, "epoch_checkpoint.pth"))
        epoch_run_times.append(time.time() - current_time)
    model.save_model(os.path.join(FLAGS.checkpoint_dir, "model_checkpoint"))
    torch.save(optimizer.state_dict(), os.path.join(FLAGS.checkpoint_dir, "optimizer_checkpoint.pth"))
    torch.save(scheduler.state_dict(), os.path.join(FLAGS.checkpoint_dir, "scheduler_checkpoint.pth"))
    loss_record = {}
    loss_record['train_total_loss'] = torch.sum(torch.stack(epoch_training_losses))
    loss_record['train_mean_epoch_loss'] = torch.mean(torch.stack(epoch_training_losses)).item()
    loss_record['train_max_epoch_loss'] = torch.max(torch.stack(epoch_training_losses)).item()
    loss_record['train_min_epoch_loss'] = torch.min(torch.stack(epoch_training_losses)).item()
    loss_record['train_epoch_losses'] = epoch_training_losses
    loss_record['all_trajectory_train_losses'] = all_trajectory_train_losses
    return loss_record





def evaluator(model, params):
  """Run a model rollout trajectory."""
  ds = dataset.load_dataset(FLAGS.dataset_dir, FLAGS.rollout_split)
  ds = dataset.add_targets(ds, [params['field']], add_history=params['history'])
  inputs = tf.data.make_one_shot_iterator(ds).get_next()
  scalar_op, traj_ops = params['evaluator'].evaluate(model, inputs)
  tf.train.create_global_step()

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.checkpoint_dir,
      save_checkpoint_secs=None,
      save_checkpoint_steps=None) as sess:
    trajectories = []
    scalars = []
    for traj_idx in range(FLAGS.num_rollouts):
      logging.info('Rollout trajectory %d', traj_idx)
      scalar_data, traj_data = sess.run([scalar_op, traj_ops])
      trajectories.append(traj_data)
      scalars.append(scalar_data)
    for key in scalars[0]:
      logging.info('%s: %g', key, np.mean([x[key] for x in scalars]))
    with open(FLAGS.rollout_path, 'wb') as fp:
      pickle.dump(trajectories, fp)


def main(argv):
  del argv
  params = PARAMETERS[FLAGS.model]
  learned_model = core_model.EncodeProcessDecode(
      output_size=params['size'],
      latent_size=128,
      num_layers=2,
      message_passing_steps=FLAGS.message_passing_steps)
  model = params['model'].Model(learned_model)
  if FLAGS.mode == 'train':
    learner(model, params)
  elif FLAGS.mode == 'eval':
    evaluator(model, params)



def main(argv):
    del argv
    params = PARAMETERS[FLAGS.model]
    config = {}

    # record start time
    run_step_start_time = time.time()
    run_step_start_datetime = datetime.datetime.fromtimestamp(run_step_start_time).strftime('%c')

    log_dir = os.path.join(FLAGS.checkpoint_dir, 'log')
    root_logger = logger_setup(os.path.join(log_dir, 'log.log'))

    root_logger.info("Program started at time " + str(run_step_start_datetime))

    root_logger.info("Start training......")

    learned_model = core_model.EncodeProcessDecode(
      output_size=params['size'],
      latent_size=128,
      num_layers=2,
      message_passing_steps=FLAGS.message_passing_steps)
    model = params['model'].Model(params, learned_model)

    if FLAGS.load_chk:
        model.load_model(os.path.join(FLAGS.checkpoint_dir, 'checkpoint', "model_checkpoint"))
        root_logger.info(
            "Loaded checkpoint file in " + str(
                os.path.join(FLAGS.checkpoint_dir, 'checkpoint')) + " and starting retraining...")



    model.to(device)

    

if __name__ == '__main__':
  app.run(main)

