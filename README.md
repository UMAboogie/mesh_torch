# mesh_torch

## Overview
The code in this repository is the PyTorch version of Learning Mesh-Based Simulation with Graph Networks.


## Setup
These codes are for Python 3.6.
Install dependencies: 

    pip install -r mesh_torch/requirements.txt

Download datasets:

    mkdir -p tmp/datasets
    bash mesh_torch/download_dataset.sh (dataset_name) tmp/datasets

(dataset_name) is any of {airfoil, cylinder_flow, deforming_plate, flag_minimal, flag_simple, sphere_simple}

The dataset in this list is tfrecord file. To use this in PyTorch, you need to transform this into h5 file. In parse_$$.py, you can transform the dataset and halve it ((ex) data num: 1000 -> 500).
Transform datasets:

    python mesh_torch/preprocess/parse_cloth.py 

To use the datasets in run_model.py, you have to add targets to each trajectory snapshot. You can use add_targets.py to do so. Please change some parts in the code and run it.
Preprocess datasets:

    python mesh_torch/preprocess/add_targets.py 

## Train the model
Making a checkpoint dir, please run run_model.py with training mode.

    python -m mesh_torch.run_model --model=cloth --mode=train --checkpoint_dir=$(checkpoint) --dataset_dir=tmp/datasets_h5_pro/flag_simple_500 --load_chk=True --max_epochs=5
$(checkpoint) is directory name you like for each training.

## Evaluate the model
Making a rollout dir, please run run_model.py with evaluate mode.

    python -m mesh_torch.run_model --model=cloth --mode=eval --checkpoint_dir=$(checkpoint) --dataset_dir=tmp/datasets_h5_pro/flag_simple_500 --rollout_dir=$(rollout) --load_chk=True --num_rollouts=10
$(rollout) is directory name you like for each evaluation.

## Plot trajectory

    mkdir tmp/anim_gif
    python -m mesh_torch.plot_cloth --rollout_dir=$(rollout) --gif_dir=tmp/anim_gif --type=flag

## Others
This repository contains some other models.
$(file name)_ne.py are codes without edge features in GNN.
time_adj.py is a file to count time making adjacancy matrix.
You can run these code in GPU.    