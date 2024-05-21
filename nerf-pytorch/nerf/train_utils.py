import torch
import sys

from .nerf_helpers import get_minibatches, ndc_rays
from .nerf_helpers import sample_pdf_2 as sample_pdf
from .nerf_helpers import dump_rays
from .volume_rendering_utils import volume_render_radiance_field


def run_network(network_fn, pts, ray_batch, chunksize, embed_fn, embeddirs_fn, expressions = None, latent_code = None):
    """
                     network_fn,     pts, ray_batch, chunksize,                             embed_fn,           embeddirs_fn,        expressions = None, latent_code = None
    pass:            model_coarse,   pts, ray_batch, getattr(options.nerf, mode).chunksize, encode_position_fn, encode_direction_fn, expressions,        latent_code
    """

    pts_flat = pts.reshape((-1, pts.shape[-1]))
    embedded = embed_fn(pts_flat)
    if embeddirs_fn is not None:
        viewdirs = ray_batch[..., None, -3:]
        input_dirs = viewdirs.expand(pts.shape)
        input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat((embedded, embedded_dirs), dim=-1)

    batches = get_minibatches(embedded, chunksize=chunksize)
    # print(type(batches), len(batches))
    # print(batches)
    """
    first iteration of batches: 
    <class 'list'> 64 
    """
    if expressions is None:
        preds = [network_fn(batch) for batch in batches]  #Run forward of ConditionalBlendshapePaperNeRFModel!!!!
    elif latent_code is not None:
        preds = [network_fn(batch, expressions, latent_code) for batch in batches]
    else:
        preds = [network_fn(batch, expressions) for batch in batches]
    radiance_field = torch.cat(preds, dim=0)
    """
    pred for coarse from ConditionalBlendshapePaperNeRFModel:
        list of nn.Linear
        activation: Relu(x)
        return torch.cat((rgb, alpha), dim=-1)
    """
    radiance_field = radiance_field.reshape(
        list(pts.shape[:-1]) + [radiance_field.shape[-1]]
    )
    # print(type(radiance_field), len(radiance_field))
    # print(radiance_field)
    """
    pred --> radiance_field --> reshape():
    <class 'torch.Tensor'> 65536
    """

    del embedded, input_dirs_flat
    return radiance_field


def predict_and_render_radiance(
    ray_batch,
    model_coarse,
    model_fine,
    options,                    #cfg
    mode="train",               #validation, in rendering
    encode_position_fn=None,
    encode_direction_fn=None,
    expressions = None,
    background_prior = None,
    latent_code = None,
    ray_dirs_fake = None        #The ray ablation thing <-- torch.cat(ro, rd_ablation, near, far)
):
    # TESTED
    num_rays = ray_batch.shape[0]
    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6].clone() # TODO remove clone ablation rays
    bounds = ray_batch[..., 6:8].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]
    # TODO: Use actual values for "near" and "far" (instead of 0. and 1.)
    # when not enabling "ndc".
    t_vals = torch.linspace(
        0.0,
        1.0,
        getattr(options.nerf, mode).num_coarse,
        dtype=ro.dtype,
        device=ro.device,
    )
    if not getattr(options.nerf, mode).lindisp:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    z_vals = z_vals.expand([num_rays, getattr(options.nerf, mode).num_coarse])

    if getattr(options.nerf, mode).perturb:
        # Get intervals between samples.
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        t_rand = torch.rand(z_vals.shape, dtype=ro.dtype, device=ro.device)
        z_vals = lower + (upper - lower) * t_rand
    # pts -> (num_rays, N_samples, 3)
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]
    """
    print(num_rays, ro.shape, rd.shape, ro.ndimension(), rd.ndimension())
    65536 torch.Size([65536, 3]) torch.Size([65536, 3]) 2 2
    """
    # Uncomment to dump a ply file visualizing camera rays and sampling points
    #dump_rays(ro.detach().cpu().numpy(), pts.detach().cpu().numpy())
    if ray_dirs_fake:
        ray_batch[...,3:6] = ray_dirs_fake[0][...,3:6] # TODO remove this this is for ablation of ray dir

    radiance_field = run_network(
        model_coarse,
        pts,
        ray_batch,
        getattr(options.nerf, mode).chunksize,
        encode_position_fn,
        encode_direction_fn,
        expressions,
        latent_code
    )
    # make last RGB values of each ray, the background
    if background_prior is not None:
        radiance_field[:,-1,:3] = background_prior

    (
        rgb_coarse,
        disp_coarse,
        acc_coarse,
        weights,
        depth_coarse,
    ) = volume_render_radiance_field(
        radiance_field,
        z_vals,
        rd,
        radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
        white_background=getattr(options.nerf, mode).white_background,
        background_prior=background_prior
    )


    #Fine network rendering should start here:
    rgb_fine, disp_fine, acc_fine = None, None, None
    if getattr(options.nerf, mode).num_fine > 0:
        # rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid,
            weights[..., 1:-1],
            getattr(options.nerf, mode).num_fine,
            det=(getattr(options.nerf, mode).perturb == 0.0),
        )
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
        # pts -> (N_rays, N_samples + N_importance, 3)
        pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

        radiance_field = run_network(
            model_fine,
            pts,
            ray_batch,
            getattr(options.nerf, mode).chunksize,
            encode_position_fn,
            encode_direction_fn,
            expressions,
            latent_code
        )
        # make last RGB values of each ray, the background
        if background_prior is not None:
            radiance_field[:, -1, :3] = background_prior

        # Uncomment to dump a ply file visualizing camera rays and sampling points
        #dump_rays(ro.detach().cpu().numpy(), pts.detach().cpu().numpy(), radiance_field)

        #dump_rays(ro.detach().cpu().numpy(), pts.detach().cpu().numpy(), torch.softmax(radiance_field[:,:,-1],1).detach().cpu().numpy())

        #rgb_fine, disp_fine, acc_fine, _, depth_fine = volume_render_radiance_field(
        rgb_fine, disp_fine, acc_fine, weights, depth_fine = volume_render_radiance_field( # added use of weights
            radiance_field,
            z_vals,
            rd,
            radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
            white_background=getattr(options.nerf, mode).white_background,
            background_prior=background_prior
        )

    #return rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine, depth_fine #added depth fine
    return rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine, weights[:,-1] #changed last return val to fine_weights


def run_one_iter_of_nerf(
    height,
    width,
    focal_length,           #intrinsics
    model_coarse,
    model_fine,
    ray_origins,
    ray_directions,
    options,                #cfg
    mode="train",           #validation, in rendering
    encode_position_fn=None,
    encode_direction_fn=None,
    expressions = None,
    background_prior=None,
    latent_code = None,
    ray_directions_ablation = None,
    back_propogate = None
):
    is_rad = torch.is_tensor(ray_directions_ablation) #True!!
    viewdirs = None
    if options.nerf.use_viewdirs:  #True!!! in config yaml
        # Provide ray directions as input
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.view((-1, 3))
    # Cache shapes now, for later restoration.
    restore_shapes = [
        ray_directions.shape,
        ray_directions.shape[:-1],
        ray_directions.shape[:-1],
    ]
    if model_fine:
        restore_shapes += restore_shapes
        restore_shapes += [ray_directions.shape[:-1]] # to return fine depth map
    if options.dataset.no_ndc is False:
        #print("calling ndc")
        ro, rd = ndc_rays(height, width, focal_length, 1.0, ray_origins, ray_directions)
        ro = ro.view((-1, 3))
        rd = rd.view((-1, 3))
    else:
        #print("calling ndc")
        #"caling normal rays (not NDC)"
        ro = ray_origins.contiguous().view((-1, 3))
        rd = ray_directions.contiguous().view((-1, 3))
        if is_rad:
            rd_ablations = ray_directions_ablation.view((-1, 3))
    near = options.dataset.near * torch.ones_like(rd[..., :1])
    far = options.dataset.far * torch.ones_like(rd[..., :1])
    rays = torch.cat((ro, rd, near, far), dim=-1)
    if is_rad:
        rays_ablation = torch.cat((ro, rd_ablations, near, far), dim=-1)
    # if options.nerf.use_viewdirs: # TODO uncomment
    #     rays = torch.cat((rays, viewdirs), dim=-1)
    #
    viewdirs = None  # TODO remove this paragraph
    if options.nerf.use_viewdirs:  #True!!! in config yaml
        # Provide ray directions as input
        if is_rad:
            viewdirs = ray_directions_ablation
            viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
            viewdirs = viewdirs.view((-1, 3))


    if is_rad:
        batches_ablation = get_minibatches(rays_ablation, chunksize=getattr(options.nerf, mode).chunksize)
    batches = get_minibatches(rays, chunksize=getattr(options.nerf, mode).chunksize)
    # print(batches)
    """
    [   tensor([[ 0.2222,  0.0497,  0.4856,  ..., -0.9641,  0.2000,  0.8000],
        [ 0.2222,  0.0497,  0.4856,  ..., -0.9639,  0.2000,  0.8000],
        [ 0.2222,  0.0497,  0.4856,  ..., -0.9637,  0.2000,  0.8000],
        ...,
        [ 0.2222,  0.0497,  0.4856,  ..., -0.8639,  0.2000,  0.8000],
        [ 0.2222,  0.0497,  0.4856,  ..., -0.8637,  0.2000,  0.8000],
        [ 0.2222,  0.0497,  0.4856,  ..., -0.8636,  0.2000,  0.8000]],
       device='cuda:0'), 
       
       tensor([[ 0.2222,  0.0497,  0.4856,  ..., -0.9576,  0.2000,  0.8000],
        [ 0.2222,  0.0497,  0.4856,  ..., -0.9575,  0.2000,  0.8000],
        [ 0.2222,  0.0497,  0.4856,  ..., -0.9573,  0.2000,  0.8000],
        ...,
        [ 0.2222,  0.0497,  0.4856,  ..., -0.8575,  0.2000,  0.8000],
        [ 0.2222,  0.0497,  0.4856,  ..., -0.8573,  0.2000,  0.8000],
        [ 0.2222,  0.0497,  0.4856,  ..., -0.8571,  0.2000,  0.8000]],
       device='cuda:0'),
       
       tensor([[ 0.2222,  0.0497,  0.4856,  ..., -0.9512,  0.2000,  0.8000],
        [ 0.2222,  0.0497,  0.4856,  ..., -0.9510,  0.2000,  0.8000],
        [ 0.2222,  0.0497,  0.4856,  ..., -0.9508,  0.2000,  0.8000],
        ...,
        [ 0.2222,  0.0497,  0.4856,  ..., -0.8511,  0.2000,  0.8000],
        [ 0.2222,  0.0497,  0.4856,  ..., -0.8509,  0.2000,  0.8000],
        [ 0.2222,  0.0497,  0.4856,  ..., -0.8507,  0.2000,  0.8000]],
       device='cuda:0'),
       
       tensor([[ 0.2222,  0.0497,  0.4856,  ..., -0.9448,  0.2000,  0.8000],
        [ 0.2222,  0.0497,  0.4856,  ..., -0.9446,  0.2000,  0.8000],
        [ 0.2222,  0.0497,  0.4856,  ..., -0.9444,  0.2000,  0.8000],
        ...,
        [ 0.2222,  0.0497,  0.4856,  ..., -0.8446,  0.2000,  0.8000],
        [ 0.2222,  0.0497,  0.4856,  ..., -0.8444,  0.2000,  0.8000],
        [ 0.2222,  0.0497,  0.4856,  ..., -0.8443,  0.2000,  0.8000]],
       device='cuda:0')]
    """
    assert(batches[0].shape == batches[0].shape)
    background_prior = get_minibatches(background_prior, chunksize=getattr(options.nerf, mode).chunksize) if\
        background_prior is not None else background_prior
    """
    According to the paper:
        For each training image Ii and training iteration, we sample a batch of 2048 viewing rays through the image pixels. 
        We use a bounding box of the head (given by the morphable model) to sample the rays such that 95% of them correspond to pixels within the bounding box and, 
        thus allowing us to reconstruct the face with a high fidelity
    """


    if mode == "train" and not back_propogate:
        print("predicting")
    if is_rad:
        pred = [
            predict_and_render_radiance(
                batch,
                model_coarse,
                model_fine,
                options,
                mode,
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
                expressions = expressions,
                background_prior = background_prior[i] if background_prior is not None else background_prior,
                latent_code = latent_code,
                ray_dirs_fake = batches_ablation
            )
            for i,batch in enumerate(batches)
        ]
    else:
        pred = [
            predict_and_render_radiance(
                batch,
                model_coarse,
                model_fine,
                options,
                mode,
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
                expressions = expressions,
                background_prior = background_prior[i] if background_prior is not None else background_prior,
                latent_code = latent_code,
                ray_dirs_fake = None
            )
            for i,batch in enumerate(batches)
        ]
    if mode == "train" and not back_propogate:
        print("Done prediction")
    # print(type(pred), len(pred))
    """
    pred <---- return rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine, weights[:,-1],
        BUT for i,batch in enumerate(batches) !!!!!!!!!!!!
        each i, batch has the seven return values(tensors)
    return from run_one_iter_of_nerf() from predict_and_render_radiance() from volume_render_radiance_field()
    <class 'list'> 4
    """
    synthesized_images = list(zip(*pred)) #Review the usage and effect of zip() !!!
    synthesized_images = [
        torch.cat(image, dim=0) if image[0] is not None else (None)
        for image in synthesized_images
    ]
    if mode == "validation":
        synthesized_images = [
            image.view(shape) if image is not None else None
            for (image, shape) in zip(synthesized_images, restore_shapes)
        ]

        # Returns rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine
        # (assuming both the coarse and fine networks are used).
        if model_fine:
            return tuple(synthesized_images)
        else:
            # If the fine network is not used, rgb_fine, disp_fine, acc_fine are
            # set to None.
            return tuple(synthesized_images + [None, None, None])

    return tuple(synthesized_images)



def run_one_iter_of_conditional_nerf(
    height,
    width,
    focal_length,
    model_coarse,
    model_fine,
    ray_origins,
    ray_directions,
    expression,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
):
    viewdirs = None
    if options.nerf.use_viewdirs:
        # Provide ray directions as input
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.view((-1, 3))
    # Cache shapes now, for later restoration.
    restore_shapes = [
        ray_directions.shape,
        ray_directions.shape[:-1],
        ray_directions.shape[:-1],
    ]
    if model_fine:
        restore_shapes += restore_shapes
        restore_shapes += ray_directions.shape[:-1] # for fine depth map

    if options.dataset.no_ndc is False:
        ro, rd = ndc_rays(height, width, focal_length, 1.0, ray_origins, ray_directions)
        ro = ro.view((-1, 3))
        rd = rd.view((-1, 3))
    else:
        ro = ray_origins.view((-1, 3))
        rd = ray_directions.view((-1, 3))
    near = options.dataset.near * torch.ones_like(rd[..., :1])
    far = options.dataset.far * torch.ones_like(rd[..., :1])
    rays = torch.cat((ro, rd, near, far), dim=-1)
    if options.nerf.use_viewdirs:
        rays = torch.cat((rays, viewdirs), dim=-1)

    batches = get_minibatches(rays, chunksize=getattr(options.nerf, mode).chunksize)
    pred = [
        predict_and_render_radiance(
            batch,
            model_coarse,
            model_fine,
            options,
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
        )
        for batch in batches
    ]
    synthesized_images = list(zip(*pred))
    synthesized_images = [
        torch.cat(image, dim=0) if image[0] is not None else (None)
        for image in synthesized_images
    ]
    if mode == "validation":
        synthesized_images = [
            image.view(shape) if image is not None else None
            for (image, shape) in zip(synthesized_images, restore_shapes)
        ]

        # Returns rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine
        # (assuming both the coarse and fine networks are used).
        if model_fine:
            return tuple(synthesized_images)
        else:
            # If the fine network is not used, rgb_fine, disp_fine, acc_fine are
            # set to None.
            return tuple(synthesized_images + [None, None, None])

    return tuple(synthesized_images)



import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=5)
