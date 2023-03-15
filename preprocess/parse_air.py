import tensorflow as tf
import functools
import json
import os
import numpy as np
import h5py

def _parse(proto, meta):
  """Parses a trajectory from tf.Example."""
  feature_lists = {k: tf.io.VarLenFeature(tf.string)
                   for k in meta['field_names']}
  features = tf.io.parse_single_example(proto, feature_lists)
  out = {}
  for key, field in meta['features'].items():
    data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
    data = tf.reshape(data, field['shape'])
    if field['type'] == 'static':
      data = tf.tile(data, [meta['trajectory_length'], 1, 1])
    elif field['type'] == 'dynamic_varlen':
      length = tf.io.decode_raw(features['length_'+key].values, tf.int32)
      length = tf.reshape(length, [-1])
      data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
    elif field['type'] != 'dynamic':
      raise ValueError('invalid data format')
    out[key] = data
  return out


def load_dataset(path, split):
  """Load dataset."""
  with open(os.path.join(path, 'meta.json'), 'r') as fp:
    meta = json.loads(fp.read())
  ds = tf.data.TFRecordDataset(os.path.join(path, split+'.tfrecord'))
  ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
  ds = ds.prefetch(1)
  return ds


if __name__ == '__main__':
    tf.enable_resource_variables()
    tf.enable_eager_execution()

    tf_datasetPath='tmp/datasets/airfoil'
    os.makedirs('tmp/datasets_h5/airfoil_500', exist_ok=True)

    for split in ['train', 'test', 'valid']:
        ds = load_dataset(tf_datasetPath, split)
        save_path='tmp/datasets_h5/airfoil_500/'+ split  +'.h5'
        f = h5py.File(save_path, "w")
        print(save_path)
        num = 0

        for index, d in enumerate(ds):
          num += 1

        for index, d in enumerate(ds):
            node_type = d['node_type'].numpy()
            cells = d['cells'].numpy()
            mesh_pos = d['mesh_pos'].numpy()
            density = d['density'].numpy()
            pressure = d['pressure'].numpy()
            velocity = d['velocity'].numpy()
            data = ("node_type",  "cells", "mesh_pos", "density", "pressure", "velocity")
            # d = f.create_dataset(str(index), (len(data), ), dtype=pos.dtype)
            g = f.create_group(str(index))
            for k in data:
             g[k] = eval(k)
            
            print(index)
            if index == (num // 2)-1:
              break
        f.close()