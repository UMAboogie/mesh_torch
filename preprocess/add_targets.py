import os
import functools
import json

import torch
from torch.utils.data import IterableDataset
import h5py
import numpy as np

from absl import app



def load_and_targets(path, split, fields, add_history, model_type):
    def fn(trajectory):
        if model_type == 'air':
            out = {}
            for key, val in trajectory.items():
                out[key] = val[1:-1]
                if key in fields:
                    out['target|'+key] = val[2:]
                if key == 'density':
                    out['target|density'] = val[2:]
        else:
            out = {}
            for key, val in trajectory.items():
                out[key] = val[1:-1]
                if key in fields:
                    if add_history:
                        out['prev|'+key] = val[0:-2]
                    out['target|'+key] = val[2:]
        return out
    data_list = []
    with h5py.File(os.path.join(path, split + ".h5"), 'r') as data:
        #print(data.keys())
        for key, traj in data.items():
            data_list.append(fn(traj))
    return data_list

def split_and_preprocess(dataset, field, scale, gamma):
    def add_noise(frame):
        zero_size = torch.zeros(frame[field].size(), dtype=torch.float32).to(device)
        noise = torch.normal(zero_size, std=noise_scale).to(device)
        other = torch.Tensor([NodeType.NORMAL.value]).to(device)
        mask = torch.eq(frame['node_type'], other.int())[:, 0]
        mask_sequence = []
        for i in range(noise.shape[1]):
            mask_sequence.append(mask)
        mask = torch.stack(mask_sequence, dim=1)
        noise = torch.where(mask, noise, torch.zeros_like(noise))
        frame[noise_field] += noise
        frame['target_' + noise_field] += (1.0 - noise_gamma) * noise
        return frame


    def fn(trajectory):
        if model_type == 'air':
            out = {}
            for key, val in trajectory.items():
                out[key] = val[1:-1]
                if key in fields:
                    out['target|'+key] = val[2:]
                """
                if key == 'pressure':
                    out['target|pressure'] = val[2:]
                """
                if key == 'density':
                    out['target|density'] = val[2:]
        else:
            out = {}
            for key, val in trajectory.items():
                out[key] = val[1:-1]
                if key in fields:
                    if add_history:
                        out['prev|'+key] = val[0:-2]
                    out['target|'+key] = val[2:]
        return out
    return 0
        


"""
class MeshDataset(IterableDataset):
    def __init__(self, path, split):
"""

def main(argv):
    os.makedirs('tmp/datasets_h5_pro/airfoil_500', exist_ok=True) #Please change it
    for split in ['train', 'test', 'valid']:
        ds = load_and_targets("tmp/datasets_h5/airfoil_500", split, "velocity", add_history=False, model_type='air') #Please change it
        save_path='tmp/datasets_h5_pro/airfoil_500/'+ split  +'.h5' # Please change it
        f = h5py.File(save_path, "w")
        print(save_path)

        for index, d in enumerate(ds):
            # Please change below
            cells = d['cells']
            mesh_pos = d['mesh_pos']
            node_type = d['node_type']
            velocity = d['velocity']
            target_velocity = d['target|velocity']
            pressure = d['pressure']
            density = d['density']
            target_density = d['target|density']
            data = ("cells", "mesh_pos", "node_type", "velocity", "target_velocity", "pressure", "density", "target_density")
            g = f.create_group(str(index))
            for k in data:
             g[k] = eval(k)
            
            print(index)
        print("ok")
        f.close()
        print("File was closed.")

    """
    with h5py.File("tmp/datasets_h5_pro/deforming_plate_600/test.h5", 'r') as data:
        #print(data.keys())
        traj = data["0"]
        tar = torch.tensor(traj["node_type"][0])
        print(torch.unique(tar))
    """





if __name__ == '__main__':
    app.run(main)