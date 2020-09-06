import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from data.cache_blender import load_blender_data
from data.cache_llff import load_llff_data
from nerf_utils import get_ray_bundle


def cache_nerf_dataset(args):
    images, poses, render_poses, hwf = (None, None, None, None)
    i_train, i_val, i_test = None, None, None

    if args.type == "blender":
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datapath, half_res=args.blender_half_res,
                                                                      testskip=args.blender_stride)

        i_train, i_val, i_test = i_split
        H, W, focal = hwf
        H, W = int(H), int(W)
    elif args.type == "llff":
        images, poses, bds, render_poses, i_test = load_llff_data(args.datapath, factor=args.llff_downsample_factor)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        if not isinstance(i_test, list):
            i_test = [i_test]
        if args.llffhold > 0:
            i_test = np.arange(images.shape[0])[:: args.llffhold]
        i_val = i_test
        i_train = np.array([i for i in np.arange(images.shape[0]) if (i not in i_test and i not in i_val)])
        H, W, focal = hwf
        H, W = int(H), int(W)

    os.makedirs(os.path.join(args.savedir, "train"), exist_ok=True)
    os.makedirs(os.path.join(args.savedir, "val"), exist_ok=True)
    os.makedirs(os.path.join(args.savedir, "test"), exist_ok=True)
    np.random.seed(args.randomseed)

    for img_idx in tqdm(i_train):
        for j in range(args.num_variations):
            img_target = images[img_idx]
            pose_target = poses[img_idx, :3, :4]
            ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)
            coords = np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing='ij'), axis=-1)
            coords = coords.reshape((-1, 2))
            if args.sample_all is False:
                select_inds = np.random.choice(coords.shape[0], size=(args.num_random_rays), replace=False)
                select_inds = coords[select_inds]
                ray_origins = ray_origins[select_inds[:, 1], select_inds[:, 0], :]
                ray_directions = ray_directions[select_inds[:, 1], select_inds[:, 0], :]
                target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]
            else:
                target_s = img_target

            ray_bundle = np.stack([ray_origins, ray_directions], axis=0)

            cache_dict = {"height": H,
                          "width": W,
                          "focal_length": focal,
                          "ray_bundle": ray_bundle,
                          "target": target_s}
            save_path = os.path.join(args.savedir, "train", str(img_idx).zfill(4) + ".data")
            pickle.dump(cache_dict, open(save_path, "wb"))

            if args.sample_all is True:
                break

    for img_idx in tqdm(i_val):
        img_target = images[img_idx]
        pose_target = poses[img_idx, :3, :4]
        ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)
        ray_bundle = np.stack([ray_origins, ray_directions], axis=0)

        cache_dict = {"height": H,
                      "width": W,
                      "focal_length": focal,
                      "ray_bundle": ray_bundle,
                      "target": img_target}
        save_path = os.path.join(args.savedir, "val", str(img_idx).zfill(4) + ".data")
        pickle.dump(cache_dict, open(save_path, "wb"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, required=True, help="Path to root dir of dataset that needs caching.")
    parser.add_argument("--savedir", type=str, required=True, help="Path to save the cached dataset to.")
    parser.add_argument("--type", type=str.lower, required=True, choices=["blender", "llff"], help="Type of the dataset to be cached.")
    parser.add_argument("--blender_half_res", type=bool, default=False, help="Whether to load the (Blender/synthetic) datasets at half the resolution.")
    parser.add_argument("--blender_stride", type=int, default=1, help="Stride length (Blender datasets only). When set to k (k > 1), it samples every kth sample from the dataset.")
    parser.add_argument("--llff_downsample_factor", type=int, default=8, help="Downsample factor for images from the LLFF dataset.")
    parser.add_argument("--llffhold", type=int, default=8, help="Determines the hold-out images for LLFF (TODO: make better).")
    parser.add_argument("--num_random_rays", type=int, default=8, help="Number of random rays to sample per image.")
    parser.add_argument("--num_variations", type=int, default=1, help="Number of random 'ray batches' to draw per image")
    parser.add_argument("--sample_all", action="store_true", help="Sample all rays for the image. Overrides --num-random-rays.")
    parser.add_argument("--randomseed", type=int, default=3920, help="Random seeed, for repeatability")
    args = parser.parse_args()
    cache_nerf_dataset(args)