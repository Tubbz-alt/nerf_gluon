import os
import glob
import time
import yaml
import argparse
import numpy as np
import mxnet as mx
from mxnet import nd, gluon, autograd
from cfgnode import CfgNode
from data.data_iter import DataLoader
from nerf_utils import run_one_iter_of_nerf, get_embedding_function, evaluate_accuracy
from ranger_optimizer import Ranger
import models

# random sampling, not just one image per iter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to (.yml) config file.")
    parser.add_argument("--load-checkpoint", type=str, default="", help="Path to load saved checkpoint from.")
    configargs = parser.parse_args()

    # Read config file.
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    ctx = mx.gpu(0)
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    mx.random.seed(seed)

    train_iter = DataLoader(ctx, cfg, glob.glob(os.path.join(cfg.dataset.cachedir, "train", "*.data")))
    val_iter = DataLoader(ctx, cfg, glob.glob(os.path.join(cfg.dataset.cachedir, "val", "*.data")), is_validation=True)

    # =====================================
    # Create Positional Encoding
    encode_position_fn = get_embedding_function(num_encoding_functions=cfg.models.encoding.num_encoding_fn_xyz,
                                                include_input=cfg.models.encoding.include_input_xyz,
                                                log_sampling=cfg.models.encoding.log_sampling_xyz)
    encode_direction_fn = None
    if cfg.nerf.use_viewdirs:
        encode_direction_fn = get_embedding_function(num_encoding_functions=cfg.models.encoding.num_encoding_fn_dir,
                                                     include_input=cfg.models.encoding.include_input_dir,
                                                     log_sampling=cfg.models.encoding.log_sampling_dir)

    # =====================================
    # Initialize a coarse-resolution model.
    initializer = mx.init.Uniform(0.07)
    model_coarse = getattr(models, cfg.models.coarse.type)(num_layers=cfg.models.coarse.num_layers,
                                                           hidden_size=cfg.models.coarse.hidden_size,
                                                           skip_connect_every=cfg.models.coarse.skip_connect_every,
                                                           num_encoding_fn_xyz=cfg.models.encoding.num_encoding_fn_xyz,
                                                           num_encoding_fn_dir=cfg.models.encoding.num_encoding_fn_dir,
                                                           include_input_xyz=cfg.models.encoding.include_input_xyz,
                                                           include_input_dir=cfg.models.encoding.include_input_dir,
                                                           use_viewdirs=cfg.nerf.use_viewdirs)
    model_coarse.initialize(initializer, ctx=ctx)

    # If a fine-resolution model is specified, initialize it.
    model_fine = model_coarse
    if cfg.models.fine.shared_network is False:
        if hasattr(cfg.models, "fine"):
            model_fine = getattr(models, cfg.models.fine.type)(num_layers=cfg.models.fine.num_layers,
                                                               hidden_size=cfg.models.fine.hidden_size,
                                                               skip_connect_every=cfg.models.fine.skip_connect_every,
                                                               num_encoding_fn_xyz=cfg.models.encoding.num_encoding_fn_xyz,
                                                               num_encoding_fn_dir=cfg.models.encoding.num_encoding_fn_dir,
                                                               include_input_xyz=cfg.models.encoding.include_input_xyz,
                                                               include_input_dir=cfg.models.encoding.include_input_dir,
                                                               use_viewdirs=cfg.nerf.use_viewdirsn)
            model_fine.initialize(initializer, ctx=ctx)

    # =====================================
    # Create trainer object
    params = model_coarse.collect_params()
    if cfg.models.fine.shared_network is False:
        params.update(model_fine.collect_params())

    if cfg.optimizer.type is not "ranger":
        trainer = gluon.Trainer(params, cfg.optimizer.type, {'learning_rate': cfg.optimizer.lr,
                                                             'wd': cfg.optimizer.wd})
    else:
        optimizer = Ranger(learning_rate=cfg.optimizer.lr, wd=cfg.optimizer.wd, beta1=0.95,
                           alpha=0.5, k=6, n_sma_threshhold=5, use_gc=False, gc_conv_only=False)
        trainer = gluon.Trainer(params, optimizer)
    loss_function = gluon.loss.L2Loss()

    # =====================================
    # Run the optimizer
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    checkpoints_dir = os.path.join(cfg.experiment.logdir, cfg.experiment.id, 'checkpoints')
    saved_images_dir = os.path.join(cfg.experiment.logdir, cfg.experiment.id, 'images')
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(saved_images_dir, exist_ok=True)
    smoothing_constant = .01
    moving_loss = 0
    start_time = time.time()
    lr_decay_count = 0
    for i in range(cfg.experiment.num_epochs):
        # Learning rate decay
        if lr_decay_count < len(cfg.scheduler.update_lr_in_epochs):
            if i == cfg.scheduler.update_lr_in_epochs[lr_decay_count]:
                trainer.set_learning_rate(trainer.learning_rate * cfg.scheduler.lr_decay_factor)
                lr_decay_count+=1
                print("Update Learning Rate: " + str(trainer.learning_rate))

        for j, data_dict in enumerate(train_iter):
            with autograd.record():
                ret = run_one_iter_of_nerf(model_coarse, model_fine, data_dict, cfg, mode="train",
                                           encode_position_fn=encode_position_fn,
                                           encode_direction_fn=encode_direction_fn)
                rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine = ret
                loss = loss_function(rgb_coarse[..., :3], data_dict['target'][..., :3])
                if rgb_fine is not None:
                    loss = loss + loss_function(rgb_fine[..., :3], data_dict['target'][..., :3])
                loss.backward()
            trainer.step(data_dict['target'].shape[0])

            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (j == 0))
                           else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)

        print("Epoch: %s, Loss: %.6f" % (i, moving_loss))
        if (i % cfg.experiment.print_training_acc_every == 0 or i == cfg.experiment.num_epochs - 1) and i > 0:
            train_cumulative_loss, train_acc = evaluate_accuracy(cfg, train_iter, model_coarse, model_fine,
                                                                 loss_function, encode_position_fn, encode_direction_fn)
            print("Train_loss %.6f, Train_acc: %.6f" % (train_cumulative_loss, train_acc))
            if i % cfg.experiment.validate_every == 0 or i == cfg.experiment.num_epochs - 1:
                is_val = (i % cfg.experiment.save_image_every == 0 and i > 0)
                val_iter.set_validation(is_val)
                val_cumulative_loss, val_acc = evaluate_accuracy(cfg, val_iter, model_coarse, model_fine,
                                                                 loss_function, encode_position_fn, encode_direction_fn,
                                                                 save_images=is_val, epoch=i)
                print("Val_loss %.6f, Val_acc: %.6f" % (val_cumulative_loss, val_acc))
            print("=========================================")

        if i % cfg.experiment.save_checkpoint_every == 0 or i == cfg.experiment.num_epochs - 1:
            model_coarse.save_parameters(checkpoints_dir + "/model_coarse_" + str(i + 1) + ".params")
            if cfg.models.fine.shared_network is False:
                model_fine.save_parameters(checkpoints_dir + "/model_fine_" + str(i + 1) + ".params")

    print("--- %s seconds ---" % (time.time() - start_time))



if __name__ == "__main__":
    main()