import os
import cv2
import numpy as np
import mxnet as mx
from mxnet import nd, gluon


def cumprod_exclusive_gluon(tensor):
    axis = -1
    shape = tensor.shape
    axis_size = shape[axis]
    arr = [nd.ones(shape[:-1] + (1,), dtype=np.float32, ctx=tensor.context)]
    for i in range(1, axis_size):
        arr.append((arr[i-1] *  tensor[..., i-1].reshape(tensor[..., i-1].shape + (1, ))))
    arr = nd.concat(*arr, dim=-1)
    return arr


def get_ray_bundle(height, width, focal_length, tform_cam2world):
    r"""Compute the bundle of rays passing through all pixels of an image (one ray per pixel).
    Args:
    height (int): Height of an image (number of pixels).
    width (int): Width of an image (number of pixels).
    focal_length (float or torch.Tensor): Focal length (number of pixels, i.e., calibrated intrinsics).
    tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
      transforms a 3D point from the camera frame to the "world" frame for the current example.
    Returns:
    ray_origins (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the centers of
      each ray. `ray_origins[i][j]` denotes the origin of the ray passing through pixel at
      row index `j` and column index `i`.
      (TODO: double check if explanation of row and col indices convention is right).
    ray_directions (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the
      direction of each ray (a unit vector). `ray_directions[i][j]` denotes the direction of the ray
      passing through the pixel at row index `j` and column index `i`.
      (TODO: double check if explanation of row and col indices convention is right).
    """
    # TESTED
    ii, jj = np.meshgrid(np.arange(width, dtype=tform_cam2world.dtype), np.arange(height, dtype=tform_cam2world.dtype),
                         indexing='xy')
    directions = np.stack(((ii - width * 0.5) / focal_length,
                           -(jj - height * 0.5) / focal_length,
                           -np.ones_like(ii),), axis=-1)
    ray_directions = np.sum(directions[..., None, :] * tform_cam2world[:3, :3], axis=-1)

    ray_origins = np.broadcast_to(tform_cam2world[:3, -1], ray_directions.shape)
    return ray_origins, ray_directions


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # UNTESTED, but fairly sure.

    # Shift rays origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = (
        -1.0
        / (W / (2.0 * focal))
        * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    )
    d1 = (
        -1.0
        / (H / (2.0 * focal))
        * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    )
    d2 = -2.0 * near / rays_o[..., 2]

    rays_o = nd.stack(*[o0, o1, o2], axis=-1)
    rays_d = nd.stack(*[d0, d1, d2], axis=-1)

    return rays_o, rays_d


def positional_encoding(F, tensor, num_encoding_functions=6, include_input=True, log_sampling=True):
    r"""Apply positional encoding to the input.

    Args:
        tensor: Input tensor to be positionally encoded.
        num_encoding_functions: Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input: Whether or not to include the input in the
            positional encoding (default: True).

    Returns:
    (nd.array): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    if log_sampling:
        frequency_bands = 2.0 ** F.linspace(0.0, num_encoding_functions - 1, num_encoding_functions,
                                            dtype=tensor.dtype, ctx=tensor.context)
    else:
        frequency_bands = F.linspace(2.0 ** 0.0, 2.0 ** (num_encoding_functions - 1), num_encoding_functions,
                                     dtype=tensor.dtype, ctx=tensor.context)

    for freq in frequency_bands:
        for func in [F.sin, F.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return F.concat(*encoding, dim=-1)


def get_embedding_function(num_encoding_functions=6, include_input=True, log_sampling=True):
    r"""Returns a lambda function that internally calls positional_encoding.
    """
    # return lambda x: positional_encoding(x, num_encoding_functions, include_input, log_sampling)
    return gluon.nn.HybridLambda(lambda F, x: positional_encoding(F, x, num_encoding_functions,
                                                                  include_input, log_sampling))


def sample_pdf(bins, weights, num_samples, det=False):
    r"""sample_pdf function from another concurrent pytorch implementation
    by yenchenlin (https://github.com/yenchenlin/nerf-pytorch).
    """
    weights = weights + 1e-5
    pdf = weights / np.sum(weights, axis=-1, keepdims=True)
    cdf = np.cumsum(pdf, axis=-1)
    cdf = np.concatenate((np.zeros_like(cdf[..., :1]), cdf), axis=-1)

    # Take uniform samples
    if det:
        u = np.linspace(0.0, 1.0, num=num_samples, dtype=weights.dtype)
        u = np.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])
    else:
        u = np.random.uniform(0.0, 1.0, list(cdf.shape[:-1]) + [num_samples])

    # inds = np.searchsorted(cdf, u, side='right')
    # inds = np.apply_along_axis(lambda a: a.searchsorted(950), axis=1, arr=air_pr)
    inds = []
    for i in range(cdf.shape[0]):
        inds.append(np.searchsorted(cdf[i], u[i], side='right'))
    inds = np.array(inds)
    below = np.maximum(np.zeros_like(inds - 1), inds - 1)
    above = np.minimum((cdf.shape[-1] - 1) * np.ones_like(inds), inds)
    inds_g = np.stack((below, above), axis=-1)  # (batchsize, num_samples, 2)

    matched_shape = (inds_g.shape[0], inds_g.shape[1], cdf.shape[-1])
    cdf_g = np.take_along_axis(np.broadcast_to(np.expand_dims(cdf, 1), matched_shape), inds_g, 2)
    bins_g = np.take_along_axis(np.broadcast_to(np.expand_dims(bins, 1), matched_shape), inds_g, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = np.where(denom < 1e-5, np.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples


def get_minibatches(inputs, chunksize=1024 * 8):
    r"""Takes a huge tensor (ray "bundle") and splits it into a list of minibatches.
    Each element of the list (except possibly the last) has dimension `0` of length
    `chunksize`.
    """
    return [inputs[i : i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


def run_network(network_fn, pts, ray_batch, chunksize, embed_fn, embeddirs_fn):
    pts_flat = pts.reshape((-1, pts.shape[-1]))
    embedded = embed_fn(pts_flat)
    if embeddirs_fn is not None:
        viewdirs = ray_batch[..., None, -3:]
        input_dirs = viewdirs.broadcast_to(pts.shape)
        input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = nd.concat(*[embedded, embedded_dirs], dim=-1)

    batches = get_minibatches(embedded, chunksize=chunksize)
    preds = [network_fn(batch) for batch in batches]
    radiance_field = nd.concat(*preds, dim=0)
    radiance_field = radiance_field.reshape(list(pts.shape[:-1]) + [radiance_field.shape[-1]])
    return radiance_field


def volume_render_radiance_field(radiance_field, depth_values, ray_directions, radiance_field_noise_std=0.0,
                                 white_background=False):
    # TESTED
    one_e_10 = nd.array([1e10], dtype=ray_directions.dtype, ctx=ray_directions.context).broadcast_to(depth_values[..., :1].shape)
    dists = nd.concat(*[depth_values[..., 1:] - depth_values[..., :-1], one_e_10], dim=-1)
    dists = dists * ray_directions[..., None, :].norm(ord=2, axis=-1)

    rgb = nd.sigmoid(radiance_field[..., :3])
    noise = 0.0
    if radiance_field_noise_std > 0.0:
        noise = nd.random.normal(0.0, 1.0, shape=radiance_field[..., 3].shape,
                                 dtype=radiance_field.dtype, ctx=radiance_field.context)
        noise = noise * radiance_field_noise_std
    sigma_a = nd.relu(radiance_field[..., 3] + noise)
    alpha = 1.0 - nd.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive_gluon(1.0 - alpha + 1e-10)

    rgb_map = weights[..., None] * rgb
    rgb_map = rgb_map.sum(axis=-2)
    depth_map = weights * depth_values
    depth_map = depth_map.sum(axis=-1)
    # depth_map = (weights * depth_values).sum(dim=-1)
    acc_map = weights.sum(axis=-1)
    disp_map = 1.0 / nd.maximum(1e-10 * nd.ones_like(depth_map), depth_map / acc_map)

    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def predict_and_render_radiance(ray_batch, model_coarse, model_fine, options, mode="train",
                                encode_position_fn=None, encode_direction_fn=None):
    # TESTED
    num_rays = ray_batch.shape[0]
    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
    bounds = ray_batch[..., 6:8].reshape((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]

    # TODO: Use actual values for "near" and "far" (instead of 0. and 1.) when not enabling "ndc".
    t_vals = nd.linspace(0.0, 1.0, getattr(options.nerf, mode).num_coarse, dtype=ro.dtype, ctx=ro.context)
    if not getattr(options.nerf, mode).lindisp:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    z_vals = z_vals.broadcast_to((num_rays, getattr(options.nerf, mode).num_coarse))

    if getattr(options.nerf, mode).perturb:
        # Get intervals between samples.
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = nd.concat(*[mids, z_vals[..., -1:]], dim=-1)
        lower = nd.concat(*[z_vals[..., :1], mids], dim=-1)
        # Stratified samples in those intervals.
        t_rand = nd.random.uniform(0.0, 1.0, z_vals.shape, dtype=ro.dtype, ctx=ro.context)
        z_vals = lower + (upper - lower) * t_rand
    # pts -> (num_rays, N_samples, 3)
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

    radiance_field = run_network(model_coarse, pts, ray_batch, getattr(options.nerf, mode).chunksize,
                                 encode_position_fn, encode_direction_fn)

    rfns = getattr(options.nerf, mode).radiance_field_noise_std
    wb = getattr(options.nerf, mode).white_background
    rgb_coarse, disp_coarse, acc_coarse, weights, depth_coarse, = volume_render_radiance_field(radiance_field,
                                                                                               z_vals, rd,
                                                                                               radiance_field_noise_std=rfns,
                                                                                               white_background=wb)
    rgb_fine, disp_fine, acc_fine = None, None, None
    if getattr(options.nerf, mode).num_fine > 0:
        # rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid.asnumpy(), weights[..., 1:-1].asnumpy(), getattr(options.nerf, mode).num_fine,
                               det=(getattr(options.nerf, mode).perturb == 0.0))

        z_samples = nd.array(z_samples, ctx=z_vals.context)
        z_vals = nd.sort(nd.concat(*[z_vals, z_samples], dim=-1), axis=-1)
        # pts -> (N_rays, N_samples + N_importance, 3)
        pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

        radiance_field = run_network(model_fine, pts, ray_batch, getattr(options.nerf, mode).chunksize,
                                     encode_position_fn, encode_direction_fn)
        rfns = getattr(options.nerf, mode).radiance_field_noise_std
        wb = getattr(options.nerf, mode).white_background
        rgb_fine, disp_fine, acc_fine, _, _ = volume_render_radiance_field(radiance_field, z_vals, rd,
                                                                           radiance_field_noise_std=rfns,
                                                                           white_background=wb)

    return rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine



def run_one_iter_of_nerf(model_coarse, model_fine, data_dict, options, mode="train",
                         encode_position_fn=None, encode_direction_fn=None):
    height = data_dict['height']
    width = data_dict['width']
    focal_length = data_dict['focal_length']
    ray_origins = data_dict['ray_origins']
    ray_directions = data_dict['ray_directions']
    viewdirs = None
    if options.nerf.use_viewdirs:
        # Provide ray directions as input
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(ord=2, axis=-1).reshape((ray_directions.shape[0], 1))
        viewdirs = viewdirs.reshape((-1, 3))

    # Cache shapes now, for later restoration.
    restore_shapes = [ray_directions.shape, ray_directions.shape[:-1], ray_directions.shape[:-1]]
    if model_fine:
        restore_shapes += restore_shapes

    if options.dataset.no_ndc is False:
        ro, rd = ndc_rays(height, width, focal_length, 1.0, ray_origins.reshape((-1, 3)), ray_directions.reshape((-1, 3)))
        ro = ro.reshape((-1, 3))
        rd = rd.reshape((-1, 3))
    else:
        ro = ray_origins.reshape((-1, 3))
        rd = ray_directions.reshape((-1, 3))
    near = options.dataset.near * nd.ones_like(rd[..., :1])
    far = options.dataset.far * nd.ones_like(rd[..., :1])
    rays = nd.concat(*[ro, rd, near, far], dim=-1)
    if options.nerf.use_viewdirs:
        rays = nd.concat(*[rays, viewdirs], dim=-1)

    batches = get_minibatches(rays, chunksize=getattr(options.nerf, mode).chunksize)
    pred = [predict_and_render_radiance(batch, model_coarse, model_fine, options, encode_position_fn=encode_position_fn,
                                        encode_direction_fn=encode_direction_fn) for batch in batches]
    synthesized_images = list(zip(*pred))
    synthesized_images = [nd.concat(*image, dim=0) if image[0] is not None else None
                          for image in synthesized_images]
    if mode == "validation":
        synthesized_images = [image.reshape(shape) if image is not None else None
                              for (image, shape) in zip(synthesized_images, restore_shapes)]
        if len(synthesized_images) == 3:
            synthesized_images.append(None)
            synthesized_images.append(None)
            synthesized_images.append(None)
    return tuple(synthesized_images)


def evaluate_accuracy(cfg, dataset, model_coarse, model_fine, loss_func, encode_position_fn, encode_direction_fn,
                      save_images=False, epoch=None):
    accuracy = mx.gluon.metric.MSE()
    cumulative_loss = 0
    num_samples = 0
    num_batches = 0
    images_folder = os.path.join(cfg.experiment.logdir, cfg.experiment.id, 'images')
    dataset.reset()
    for j, data_dict in enumerate(dataset):
        num_samples += data_dict['target'].shape[0]
        num_batches += 1

        ret = run_one_iter_of_nerf(model_coarse, model_fine, data_dict, cfg, mode="validation",
                                   encode_position_fn=encode_position_fn,
                                   encode_direction_fn=encode_direction_fn)
        rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine = ret
        if save_images:
            target_flatten = data_dict['target'].reshape((-1, 3))
            rgb_coarse_flatten = rgb_coarse[..., :3].reshape((-1, 3))
            rgb_fine_flatten = rgb_fine[..., :3].reshape((-1, 3))

            loss = loss_func(rgb_coarse_flatten, target_flatten)
            loss = loss + loss_func(rgb_fine_flatten, target_flatten)
            accuracy.update(target_flatten, rgb_coarse_flatten)
            accuracy.update(target_flatten, rgb_fine_flatten)
        else:
            loss = loss_func(rgb_coarse[..., :3], data_dict['target'][..., :3])
            loss = loss + loss_func(rgb_fine[..., :3], data_dict['target'][..., :3])
            accuracy.update(data_dict['target'][..., :3], rgb_coarse[..., :3])
            accuracy.update(data_dict['target'][..., :3], rgb_fine[..., :3])

        cumulative_loss += nd.sum(loss).asscalar()

        if save_images:
            target_filename = images_folder + "/epoch_" + str(epoch) + ", img_" + str(j) + "_0_target.png"
            rgb_coarse_filename = images_folder + "/epoch_" + str(epoch) + ", img_" + str(j) + "_1_rgb_coarse.png"
            rgb_fine_filename = images_folder + "/epoch_" + str(epoch) + ", img_" + str(j) + "_2_rgb_fine.png"
            target_image = cv2.cvtColor(data_dict['target'].asnumpy() * 255, cv2.COLOR_RGB2BGR)
            rgb_coarse_image = cv2.cvtColor(rgb_coarse[..., :3].asnumpy() * 255, cv2.COLOR_RGB2BGR)
            rgb_fine_image = cv2.cvtColor(rgb_fine[..., :3].asnumpy() * 255, cv2.COLOR_RGB2BGR)
            cv2.imwrite(target_filename, target_image)
            cv2.imwrite(rgb_coarse_filename, rgb_coarse_image)
            cv2.imwrite(rgb_fine_filename, rgb_fine_image)
            if j == 3:
                break

    return cumulative_loss / num_samples, accuracy.get()[1]