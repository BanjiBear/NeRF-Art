import argparse
import glob
import os
import time
import sys

sys.path.insert(1, './nerf')
os.environ['GPU_DEBUG']='3'
import numpy as np
import torch
import torchvision
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from nerf.load_flame import load_flame_data

from nerf import (CfgNode, get_embedding_function, get_ray_bundle, img2mse,
                  load_llff_data, meshgrid_xy, models,
                  mse2psnr, run_one_iter_of_nerf, dump_rays, GaussianSmoothing)
#from gpu_profile import gpu_profile


#Modifications
import imageio #For tensor to image
import random
from einops import rearrange
#NeRF-Art losses
from criteria.clip_loss import CLIPLoss
from criteria.patchnce_loss import PatchNCELoss
from criteria.contrastive_loss import ContrastiveLoss
from criteria.perp_loss import VGGPerceptualLoss


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint from.",
    )
    parser.add_argument(
        "--savedir", type=str, default='./renders/', help="Save images to this directory, if specified."
    )
    parser.add_argument(
        "--textprompt", type=str, required=True, help="Target style input text prompt."
    )
    configargs = parser.parse_args()

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)

    # If a pre-cached dataset is available, skip the dataloader.
    train_paths, validation_paths = None, None
    images, poses, render_poses, hwf, i_split, expressions = None, None, None, None, None, None
    H, W, focal, i_train, i_val, i_test = None, None, None, None, None, None
    images, poses, render_poses, hwf, expressions = None, None, None, None, None
    if cfg.dataset.type.lower() == "blender":
        images, poses, render_poses, hwf, i_split, expressions, _, bboxs = load_flame_data(
            cfg.dataset.basedir,
            half_res=cfg.dataset.half_res,
            testskip=cfg.dataset.testskip,
            test=True
        )
        # i_train, i_val, i_test = i_split
        i_test = i_split[0]
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]
        if cfg.nerf.train.white_background:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda" #+ ":" + str(cfg.experiment.device)
        print("Using cuda")
    else:
        device = "cpu"
        print("Using cpu")

    encode_position_fn = get_embedding_function(
        num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
        include_input=cfg.models.coarse.include_input_xyz,
        log_sampling=cfg.models.coarse.log_sampling_xyz,
    )

    encode_direction_fn = None
    if cfg.models.coarse.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
            include_input=cfg.models.coarse.include_input_dir,
            log_sampling=cfg.models.coarse.log_sampling_dir,
        )

    # Initialize a coarse-resolution model.
    model_coarse = getattr(models, cfg.models.coarse.type)(
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        include_input_dir=cfg.models.coarse.include_input_dir,
        use_viewdirs=cfg.models.coarse.use_viewdirs,
        num_layers=cfg.models.coarse.num_layers,
        hidden_size=cfg.models.coarse.hidden_size,
        include_expression=True
    )
    model_coarse.to(device)
    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    if hasattr(cfg.models, "fine"):
        model_fine = getattr(models, cfg.models.fine.type)(
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            use_viewdirs=cfg.models.fine.use_viewdirs,
            num_layers = cfg.models.coarse.num_layers,
            hidden_size =cfg.models.coarse.hidden_size,
            include_expression=True
        )
        model_fine.to(device)

    ###################################
    ###################################
    train_background = False
    supervised_train_background = False
    blur_background = False

    train_latent_codes = True
    disable_expressions = False # True to disable expressions
    disable_latent_codes = False # True to disable latent codes
    fixed_background = True # Do False to disable BG
    regularize_latent_codes = True # True to add latent code LOSS, false for most experiments
    ###################################
    ###################################

    replace_background = True
    if replace_background:
        from PIL import Image
        #background = Image.open('./view.png')
        background = Image.open(cfg.dataset.basedir + '/bg/00050.png')
        #background = Image.open("./real_data/andrei_dvp/" + '/bg/00050.png')
        background.thumbnail((H,W))
        background = torch.from_numpy(np.array(background).astype(float)).to(device)
        background = background/255
        print('loaded custom background of shape', background.shape)
    """
    loaded custom background of shape torch.Size([512, 512, 3])
    """

    # Initialize optimizer.
    trainable_parameters = list(model_coarse.parameters())
    if model_fine is not None:
        trainable_parameters += list(model_fine.parameters())
    # if train_latent_codes:
    #     # latent_codes = torch.zeros(len(i_train),32, device=device)
    #     latent_codes = torch.zeros(len(i_test),32, device=device)
    #     print("initialized latent codes with shape %d X %d" % (latent_codes.shape[0], latent_codes.shape[1]))
    #     if not disable_latent_codes:
    #         trainable_parameters.append(latent_codes)
    #         latent_codes.requires_grad = True

    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        [{'params':trainable_parameters},
            {'params': background, 'lr': cfg.optimizer.lr}        ], # this is obsolete but need for continuing training
        lr=cfg.optimizer.lr
    )
    
    
    #######################
    #                     #
    #   LOAD CHECKPOINT   #
    #                     #
    #######################
    checkpoint = torch.load(configargs.load_checkpoint)
    # ['iter', 'model_coarse_state_dict', 'model_fine_state_dict', 'optimizer_state_dict', 'loss', 'psnr', 'background', 'latent_codes']
    model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
    if checkpoint["model_fine_state_dict"]:
        try:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        except:
            print(
                "The checkpoint has a fine-level model, but it could "
                "not be loaded (possibly due to a mismatched config file."
            )
    if "height" in checkpoint.keys():
        hwf[0] = checkpoint["height"]
    if "width" in checkpoint.keys():
        hwf[1] = checkpoint["width"]
    if "focal_length" in checkpoint.keys():
        hwf[2] = checkpoint["focal_length"]
    if "background" in checkpoint.keys():
        background = checkpoint["background"]
        if background is not None:
            print("loaded background from checkpoint with shape", background.shape)
            background.to(device)
    if "latent_codes" in checkpoint.keys():
        latent_codes = checkpoint["latent_codes"]
        if latent_codes is not None:
            latent_codes.to(device)
            print("loading index map for latent codes...")
            idx_map = np.load(cfg.dataset.basedir + "/index_map.npy").astype(int)
            print("loaded latent codes from checkpoint, with shape", latent_codes.shape)
        latent_codes.requires_grad = False
    
    """
    loaded background with shape  torch.Size([512, 512, 3])
    loaded latent codes from checkpoint, with shape torch.Size([5507, 32])
    """
    
    # Setup logging.
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    # Write out config parameters.
    with open(os.path.join(logdir, "config.yml"), "w") as f:
        f.write(cfg.dump())  # cfg, f, default_flow_style=False)
    os.makedirs(configargs.savedir, exist_ok=True)

    # By default, start at iteration 0 (unless a checkpoint is specified).
    start_iter = 0

    # Prepare importance sampling maps
    ray_importance_sampling_maps = []
    p = 0.9
    print("computing boundix boxes probability maps")
    for i in i_test: #for i in i_train:
        bbox = bboxs[i]
        probs = np.zeros((H,W))
        probs.fill(1-p)
        # print(type(probs), type(bbox), type(bbox[0]), type(p))
        #<class 'numpy.ndarray'> <class 'torch.Tensor'> <class 'torch.Tensor'> <class 'float'>
        probs[bbox[0]:bbox[1],bbox[2]:bbox[3]] = p
        probs = (1/probs.sum()) * probs
        ray_importance_sampling_maps.append(probs.reshape(-1))

    contrastive_loss = ContrastiveLoss()
    patchnce_loss = PatchNCELoss([512, 512]).cuda()
    clip_loss = CLIPLoss()
    perp_loss = VGGPerceptualLoss().cuda()
    loss_dict = {'contrastive': contrastive_loss, 'patchnce': patchnce_loss,\
            'clip': clip_loss, 'perceptual': perp_loss}
    
    
    ######################
    #                    #
    #     START LOOP     #
    #                    #
    ######################
    print("Starting loop")
    img_index = 0
    for i in trange(start_iter, cfg.experiment.train_iters):
        """
        Error here, i range will go over both training and testing dataset
        Previous: random sample image for reconstruction!
        """

        model_coarse.train()
        if model_fine:
            model_coarse.train()

        rgb_coarse, rgb_fine = None, None
        target_ray_values = None
        background_ray_values = None
        
        
        #Start the training
        # img_idx = np.random.choice(i_train)
        img_target = images[img_index].to(device)
        pose_target = poses[img_index, :3, :4].to(device)
        if not disable_expressions:
            expression_target = expressions[img_index].to(device) # vector
        else: # zero expr
            expression_target = torch.zeros(76, device=device)
        
        
        # #bbox = bboxs[img_idx]
        # if not disable_latent_codes:
        #     latent_code = latent_codes[i].to(device) if train_latent_codes else None
        # else:
        #     latent_codes = torch.zeros(32, device=device)
        latent_code = latent_codes[idx_map[img_index, 0]].to(device)
        
        ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)
        
        TARGET_TEXT = configargs.textprompt
        # TARGET_TEXT = "van gogh oil painting"
        # TARGET_TEXT = "painting, oil on canvas, Vincent van gogh self-portrait style"
        def create_fine_neg_texts(target_text = TARGET_TEXT):
            path = "criteria/neg_text.txt"
            results = {}
            curr_key = 0
            with open(path, 'r') as fr:
                contents = fr.readlines()
                for item in contents:
                    item = item.strip()
                    if item.startswith("#"):
                        curr_key = item[1:]
                        results[curr_key] = []
                    else:
                        results[curr_key].append(item.split(".")[1])

            all_texts = []
            remove_ids = [] 
            ttext = target_text.lower()
            if 'botero' in ttext or 'monalisa' in ttext or 'portrait' in ttext or 'painting' in ttext:
                remove_ids = ['portrait']
            elif 'zombie' in ttext:
                remove_ids = ['zombie']
            elif 'wolf' in ttext:
                remove_ids = ['wolf']
            elif 'pixlar' in ttext or 'disney' in ttext:
                remove_ids = ['disney']
            elif 'sketch' in ttext:
                remove_ids = ['sketch'] 

            for key in results:
                if key not in remove_ids:
                #if key in remove_ids:
                    all_texts += results[key]
            return all_texts

        def calc_style_loss(rgb:torch.Tensor, rgb_gt:torch.Tensor, H):
            """
            Calculate CLIP-driven style losses

            Input
            -----
            rgb: torch.Tensor, [B, H*W, 3]
            rgb_gt: torch.Tensor, [B, H*W,3 ]
            H: int, height of the image
            """
            loss = 0.0
            rgb_pred = rearrange(rgb, "B (H W) C -> B C H W", H=H)
            rgb_gt = rearrange(rgb_gt, "B (H W) C -> B C H W", H=H)
            s_text = "photo"
            t_text = TARGET_TEXT
            #direct clip loss
            dir_clip_loss = loss_dict["clip"](rgb_gt, s_text, rgb_pred, t_text)
            loss = loss + dir_clip_loss * 1.0
            print("Directional CLIP loss:", dir_clip_loss.data.detach().cpu().numpy()* 1.0)

            #persptual
            perp_loss = loss_dict["perceptual"](rgb_pred, rgb_gt)
            loss = loss + perp_loss * 2.0
            print("Perceptual loss:", perp_loss.data.detach().cpu().numpy()* 2.0)

            #Global contrastive
            s_text_list = neg_texts
            s_text = random.choice(s_text_list)
            loss_contrastive = loss_dict["contrastive"](rgb_gt, s_text, rgb_pred, t_text)
            loss = loss + loss_contrastive * 0.2
            print("Global contrastive loss:", loss_contrastive.data.detach().cpu().numpy()* 0.2)

            #local contrastive
            #stexts = ['Photo', 'Human', 'Human face', 'Real face']
            neg_counts = 8
            s_text_list = random.sample(neg_texts, neg_counts)
            is_full_res = 2 == 1
            loss_patchnce = loss_dict["patchnce"](s_text_list, rgb_pred, t_text, is_full_res)
            loss = loss + loss_patchnce * 0.1
            print("Local contrastive loss:", loss_patchnce.data.detach().cpu().numpy()* 0.1)

            writer.add_scalar("Stylization/dir_clip_loss", dir_clip_loss.data.detach().cpu().numpy()* 1.0, i)
            writer.add_scalar("Stylization/perp_loss", perp_loss.data.detach().cpu().numpy()* 2.0, i)
            writer.add_scalar("Stylization/global_contrastive_loss", loss_contrastive.data.detach().cpu().numpy()* 0.2, i)
            writer.add_scalar("Stylization/local_contrastive_loss", loss_patchnce.data.detach().cpu().numpy()* 0.1, i)
            writer.add_scalar("Stylization/training_loss", loss.data.detach().cpu().numpy(), i)

            return loss
        
        ################
        #              #
        #  Model Fine  #
        #              #
        ################
        with torch.no_grad():
            rgb_coarse, _, _, rgb_fine, _, _, weights = run_one_iter_of_nerf(
                H,
                W,
                focal,
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
                expressions = expression_target,
                background_prior = background.view(-1,3) if (background is not None) else None,     #background NOT None
                latent_code = latent_code,
                # latent_code = latent_code if not disable_latent_codes else torch.zeros(32,device=device),
                ray_directions_ablation = None
            )
            rgb_fine = rgb_fine.unsqueeze(0)
        
        ######################
        #                    #
        #  LOSS CALCULATION  #
        #                    #
        ######################
        neg_texts = None
        if neg_texts is None:
            neg_texts = create_fine_neg_texts()

        rgb_fine.requires_grad_(True)
        img_target = torch.reshape(img_target, (1, -1, 3)).to(device)
        target_rgb = torch.gather(img_target, 1, torch.stack(3*[torch.arange(0, 512*512).unsqueeze(0).to(device)],-1).to(device)).to(device)
        losses = calc_style_loss(rgb_fine, target_rgb, 512)
        del rgb_coarse, target_rgb, _

        print("Total loss: ", losses.data.detach().cpu().numpy())
        writer.add_scalar("Stylization/Total loss", losses.data.detach().cpu().numpy(), i)

        ######################
        #                    #
        #  BACK PROPOGATION  #
        #                    #
        ######################
        print("Back Propogate for iteration: ", i)
        losses.backward()
        gradient_fine = rgb_fine.grad.clone().detach()
        gradient_fine = gradient_fine.squeeze(0)
        optimizer.zero_grad()
        
        coords = torch.stack(
            meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)),
            dim=-1,
        )
        coords = coords.reshape((-1, 2)) #[262144, 2], coordinates, (0, 0) ... (512, 512)

        #Batch Back Propogate for Fine Network
        for batch in range(0, 512*512, 2048):
            """
            #Original: [2048, 2], random sampled 2048 coordinates
            select_inds = np.random.choice(
                coords.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False, p=ray_importance_sampling_maps[img_idx]
            )
            """
            select_inds = list(range(batch, batch + 2048))
            select_inds = coords[select_inds]
            
            #ray_origins --> [512, 512, 3]
            ray_origins_patch = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
            #ray_origins --> [2048, 3]
            
            #ray_directions --> [512, 512, 3]
            ray_directions_patch = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
            #ray_origins --> [2048, 3]
            
            # print("Back Propogate Iteration:", i, i + 2048)
            _, _, _, pred_fine, _, _, _ = run_one_iter_of_nerf(
                H,
                W,
                focal,
                model_coarse,
                model_fine,
                ray_origins_patch,
                ray_directions_patch,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
                expressions = expression_target,
                background_prior = background.view(-1,3) if (background is not None) else None,     #background NOT None
                latent_code = latent_code,
                # latent_code = latent_code if not disable_latent_codes else torch.zeros(32,device=device),
                ray_directions_ablation = None,
                back_propogate = True
            )
            # pred_fine = pred_fine.unsqueeze(0)
            """
            pred_fine.unsqueeze(0).shape: torch.Size([1, 2048, 3])
            gradient_fine.shape: torch.Size([1, 262144, 3])
            gradient_fine[select_inds[:, 0], select_inds[:, 1], :].shape: torch.Size([2048, 3])
            """
            pred_fine.backward(gradient_fine[select_inds[:, 0], :], retain_graph=True)

            del ray_origins_patch, ray_directions_patch, pred_fine, _

        optimizer.step()

        
        ######################
        #                    #
        #     SAVE IMAGE     #
        #                    #
        ######################
        with torch.no_grad():
            rgb_coarse, _, _, rgb_fine, _, _, weights = run_one_iter_of_nerf(
                H,
                W,
                focal,
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="validation",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
                expressions = expression_target,
                background_prior = background.view(-1,3) if (background is not None) else None,     #background NOT None
                latent_code = latent_code,
                # latent_code = latent_code if not disable_latent_codes else torch.zeros(32,device=device),
                ray_directions_ablation = None
            )
            rgb = rgb_fine if rgb_fine is not None else rgb_coarse
        savefile = os.path.join(configargs.savedir, str(i) + ".png")
        imageio.imwrite(
            savefile, cast_to_image(rgb[..., :3], cfg.dataset.type.lower())
        )
        del gradient_fine, rgb_coarse, rgb_fine, rgb, ray_origins, ray_directions, _

        
        ################################
        #                              #
        #     UPDATE LEARNING RATE     #
        #                              #
        ################################
        # Learning rate updates
        writer.add_scalar("Stylization/learning_rate", cfg.optimizer.lr, i)
        
        num_decay_steps = cfg.scheduler.lr_decay * 1000
        lr_new = cfg.optimizer.lr * (cfg.scheduler.lr_decay_factor ** (i / num_decay_steps))
        for param_group in optimizer.param_groups:
            print(param_group["lr"])
            param_group["lr"] = lr_new
            # print(param_group["lr"])
        
        
        
        ###########################
        #                         #
        #     SAVE CHECKPOINT     #
        #                         #
        ###########################
        if i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1:
            checkpoint_dict = {
                "iter": i,
                "model_coarse_state_dict": model_coarse.state_dict(),
                "model_fine_state_dict": None
                if not model_fine
                else model_fine.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": losses,
                "psnr": None,
                "background": None
                if not (train_background or fixed_background)
                else background.data,
                "latent_codes": latent_codes.data
                # "latent_codes": None if not train_latent_codes else latent_codes.data
            }
            torch.save(
                checkpoint_dict,
                os.path.join(logdir, "checkpoint" + str(i).zfill(5) + ".ckpt"),
            )
            tqdm.write("================== Saved Checkpoint =================")
        
        
        img_index = img_index + 1;
        if img_index > 34:
            img_index = 0;
            

    writer.close()
    print("Done!")


# def cast_to_image(tensor):
#     # Input tensor is (H, W, 3). Convert to (3, H, W).
#     tensor = tensor.permute(2, 0, 1)
#     tensor = tensor.clamp(0.0,1.0)
#     # Conver to PIL Image and then np.array (output shape: (H, W, 3))
#     img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
#     # Map back to shape (3, H, W), as tensorboard needs channels first.
#     img = np.moveaxis(img, [-1], [0])
#     return img

def cast_to_image(tensor, dataset_type):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.clamp(0.0,1.0)
    # Convert to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    return img
    # # Map back to shape (3, H, W), as tensorboard needs channels first.
    # return np.moveaxis(img, [-1], [0])

def handle_pdb(sig, frame):
    import pdb
    pdb.Pdb().set_trace(frame)


if __name__ == "__main__":
    import signal

    print("before signal registration")
    signal.signal(signal.SIGTERM, handle_pdb)
    print("after registration")
    #sys.settrace(gpu_profile)

    main()
