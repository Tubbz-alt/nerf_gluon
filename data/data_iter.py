import pickle
import random
import numpy as np
import mxnet as mx
from mxnet import nd


class DataLoader(mx.io.DataIter):
    def __init__(self, ctx, cfg, pickle_files, is_validation=False):
        super(DataLoader, self).__init__()
        self.ctx = ctx
        self.cfg = cfg
        self.pickle_files = pickle_files
        self.is_validation = is_validation

        datafile = np.random.choice(pickle_files)
        cache_dict = pickle.load(open(datafile, 'rb'))
        ray_bundle = cache_dict["ray_bundle"]
        ray_origins = ray_bundle[0].reshape((-1, 3))
        if ray_origins.shape[0] > cfg.nerf.train.num_random_rays:
            self.batch_size = cfg.nerf.train.num_random_rays
        else:
            self.batch_size = ray_origins.shape[0]

        self.num_batches = len(pickle_files)
        self.num_examples = self.batch_size * len(pickle_files)
        self.cur_batch = 0
        self.reset()

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_examples

    def __next__(self):
        return self.next()

    def reset(self):
        random.shuffle(self.pickle_files)
        self.cur_batch = 0

    def set_validation(self, val):
        self.is_validation = val

    def next(self):
        if self.cur_batch < self.num_batches:
            cache_dict = pickle.load(open(self.pickle_files[self.cur_batch], 'rb'))
            ray_bundle = cache_dict["ray_bundle"]
            if self.is_validation is False:
                target_ray_values = cache_dict["target"][..., :3].reshape((-1, 3))
                ray_origins, ray_directions = (ray_bundle[0].reshape((-1, 3)), ray_bundle[1].reshape((-1, 3)))
                select_inds = np.random.choice(ray_origins.shape[0], size=self.batch_size, replace=False)
                target_ray_values = target_ray_values[select_inds]
                ray_origins, ray_directions = (ray_origins[select_inds], ray_directions[select_inds])
            else:
                target_ray_values = cache_dict["target"][..., :3]
                ray_origins, ray_directions = (ray_bundle[0], ray_bundle[1])

            self.cur_batch += 1
            ret_dict ={'target': nd.array(target_ray_values, ctx=self.ctx),
                       'ray_origins': nd.array(ray_origins, ctx=self.ctx),
                       'ray_directions': nd.array(ray_directions, ctx=self.ctx),
                       'width': cache_dict["width"],
                       'height': cache_dict["height"],
                       'focal_length': cache_dict["focal_length"]}
            return ret_dict
        else:
            self.reset()
            raise StopIteration