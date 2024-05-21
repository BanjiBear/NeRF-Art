import json
import os, sys

import cv2
import imageio
import numpy as np
import torch


def translate_by_t_along_z(t):
    tform = np.eye(4).astype(np.float32)
    tform[2][3] = t
    return tform


def rotate_by_phi_along_x(phi):
    tform = np.eye(4).astype(np.float32)
    tform[1, 1] = tform[2, 2] = np.cos(phi)
    tform[1, 2] = -np.sin(phi)
    tform[2, 1] = -tform[1, 2]
    return tform


def rotate_by_theta_along_y(theta):
    tform = np.eye(4).astype(np.float32)
    tform[0, 0] = tform[2, 2] = np.cos(theta)
    tform[0, 2] = -np.sin(theta)
    tform[2, 0] = -tform[0, 2]
    return tform


def pose_spherical(theta, phi, radius):
    """
    (theta,                             phi,        radius)
    np.linspace(-180, 180, 40 + 1),     -30.0,      4
    """
    c2w = translate_by_t_along_z(radius)
    c2w = rotate_by_phi_along_x(phi / 180.0 * np.pi) @ c2w
    c2w = rotate_by_theta_along_y(theta / 180 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def load_flame_data(basedir, half_res=False, testskip=1, debug=False, expressions=True, load_frontal_faces=False, load_bbox=True, test=False):
    # (cfg.dataset.basedir,  False,          1,                                                                                   test=True)
    print("starting data loading")
    splits = ["train", "val", "test"]
    if test:
        splits = ["test"]
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, f"transforms_{s}.json"), "r") as fp:
            metas[s] = json.load(fp)
    #transform_test.json

    all_frontal_imgs = []
    all_imgs = []
    all_poses = []
    all_expressions = []
    all_bboxs = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        expressions = []
        frontal_imgs = []
        bboxs = []
        if s == "train" or testskip == 0:
            skip = 1
        else:
            skip = testskip
        #skip = 1 in this case, "test"

        # print(len(meta["frames"][::skip]))
        # #1000
        
        counter  = 0
        for frame in meta["frames"][::skip]:
            if counter > 2000:
                break;
            fname = os.path.join(basedir, frame["file_path"] + ".png")
            imgs.append(imageio.imread(fname))
            if load_frontal_faces:
                fname = os.path.join(basedir, frame["file_path"] + "_frontal" + ".png")
                frontal_imgs.append(imageio.imread(fname))
            # import matplotlib.pyplot as plt
            # plt.imshow(imgs[-1])
            # plt.show()
                
            poses.append(np.array(frame["transform_matrix"]))
            expressions.append(np.array(frame["expression"]))
            if load_bbox:
                if "bbox" not in frame.keys():
                    bboxs.append(np.array([0.0,1.0,0.0,1.0]))
                else:
                    bboxs.append(np.array(frame["bbox"]))
            counter = counter + 1

        imgs = (np.array(imgs) / 255.0).astype(np.float32)
        # imgs = [
        #     cv2.resize(imgs[i], dsize=(256, 256), interpolation=cv2.INTER_AREA)
        #     for i in range(imgs.shape[0])
        # ]
        if load_frontal_faces:
            frontal_imgs = (np.array(frontal_imgs) / 255.0).astype(np.float32)
        #load_frontal_faces is False

        poses = np.array(poses).astype(np.float32)
        expressions = np.array(expressions).astype(np.float32)
        bboxs = np.array(bboxs).astype(np.float32)

        counts.append(counts[-1] + imgs.shape[0])
        # print(counts) #[Tommy] Debug
        # [0, 1000]
        all_imgs.append(imgs)
        all_frontal_imgs.append(frontal_imgs)
        all_poses.append(poses)
        all_expressions.append(expressions)
        all_bboxs.append(bboxs)
        # print(len(all_imgs), len(all_frontal_imgs), len(all_poses), len(all_expressions), len(all_bboxs))  #[Tommy] Debug
        # 1 1 1 1 1

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]
    # A list of one array, containing 1 ~ 1000
    
    # print(type(imgs), type(all_imgs))
    # <class 'numpy.ndarray'> <class 'list'>
    imgs = np.concatenate(all_imgs, 0)
    frontal_imgs = np.concatenate(all_frontal_imgs, 0) if load_frontal_faces else None # None!!!
    poses = np.concatenate(all_poses, 0)
    expressions = np.concatenate(all_expressions, 0)
    bboxs = np.concatenate(all_bboxs, 0) #0: in 0 dimension

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
    """
    H       W       focal
    512     512     -2223.21152
    """

    #focals = (meta["focals"])
    intrinsics = meta["intrinsics"] if meta["intrinsics"] else None
    if meta["intrinsics"]:
        intrinsics = np.array(meta["intrinsics"])
    else:
        intrinsics = np.array([focal, focal, 0.5, 0.5]) # fx fy cx cy
    # if type(focals) is list:
    #     focal = np.array([W*focals[0], H*focals[1]]) # fx fy  - x is width
    # else:
    #     focal = np.array([focal, focal])
        
    
    """
    np.linspace(-180, 180, 40 + 1)
    array([-180., -171., -162., -153., -144., -135., -126., -117., -108.,
        -99.,  -90.,  -81.,  -72.,  -63.,  -54.,  -45.,  -36.,  -27.,
        -18.,   -9.,    0.,    9.,   18.,   27.,   36.,   45.,   54.,
        63.,   72.,   81.,   90.,   99.,  108.,  117.,  126.,  135.,
        144.,  153.,  162.,  171.,  180.])
    """
    render_poses = torch.stack(
        [
            torch.from_numpy(pose_spherical(angle, -30.0, 4.0))
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ],
        0,
    )
    """
    tensor([[[ 1.0000e+00,  6.1232e-17, -1.0606e-16, -4.2423e-16],
         [-1.2246e-16,  5.0000e-01, -8.6603e-01, -3.4641e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 9.8769e-01,  7.8217e-02, -1.3548e-01, -5.4190e-01],
         [-1.5643e-01,  4.9384e-01, -8.5536e-01, -3.4215e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 9.5106e-01,  1.5451e-01, -2.6762e-01, -1.0705e+00],
         [-3.0902e-01,  4.7553e-01, -8.2364e-01, -3.2946e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 8.9101e-01,  2.2700e-01, -3.9317e-01, -1.5727e+00],
         [-4.5399e-01,  4.4550e-01, -7.7163e-01, -3.0865e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 8.0902e-01,  2.9389e-01, -5.0904e-01, -2.0361e+00],
         [-5.8779e-01,  4.0451e-01, -7.0063e-01, -2.8025e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 7.0711e-01,  3.5355e-01, -6.1237e-01, -2.4495e+00],
         [-7.0711e-01,  3.5355e-01, -6.1237e-01, -2.4495e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 5.8779e-01,  4.0451e-01, -7.0063e-01, -2.8025e+00],
         [-8.0902e-01,  2.9389e-01, -5.0904e-01, -2.0361e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 4.5399e-01,  4.4550e-01, -7.7163e-01, -3.0865e+00],
         [-8.9101e-01,  2.2700e-01, -3.9317e-01, -1.5727e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 3.0902e-01,  4.7553e-01, -8.2364e-01, -3.2946e+00],
         [-9.5106e-01,  1.5451e-01, -2.6762e-01, -1.0705e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 1.5643e-01,  4.9384e-01, -8.5536e-01, -3.4215e+00],
         [-9.8769e-01,  7.8217e-02, -1.3548e-01, -5.4190e-01],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-6.1232e-17,  5.0000e-01, -8.6603e-01, -3.4641e+00],
         [-1.0000e+00, -3.0616e-17,  5.3029e-17,  2.1212e-16],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-1.5643e-01,  4.9384e-01, -8.5536e-01, -3.4215e+00],
         [-9.8769e-01, -7.8217e-02,  1.3548e-01,  5.4190e-01],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-3.0902e-01,  4.7553e-01, -8.2364e-01, -3.2946e+00],
         [-9.5106e-01, -1.5451e-01,  2.6762e-01,  1.0705e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-4.5399e-01,  4.4550e-01, -7.7163e-01, -3.0865e+00],
         [-8.9101e-01, -2.2700e-01,  3.9317e-01,  1.5727e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-5.8779e-01,  4.0451e-01, -7.0063e-01, -2.8025e+00],
         [-8.0902e-01, -2.9389e-01,  5.0904e-01,  2.0361e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-7.0711e-01,  3.5355e-01, -6.1237e-01, -2.4495e+00],
         [-7.0711e-01, -3.5355e-01,  6.1237e-01,  2.4495e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-8.0902e-01,  2.9389e-01, -5.0904e-01, -2.0361e+00],
         [-5.8779e-01, -4.0451e-01,  7.0063e-01,  2.8025e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-8.9101e-01,  2.2700e-01, -3.9317e-01, -1.5727e+00],
         [-4.5399e-01, -4.4550e-01,  7.7163e-01,  3.0865e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-9.5106e-01,  1.5451e-01, -2.6762e-01, -1.0705e+00],
         [-3.0902e-01, -4.7553e-01,  8.2364e-01,  3.2946e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-9.8769e-01,  7.8217e-02, -1.3548e-01, -5.4190e-01],
         [-1.5643e-01, -4.9384e-01,  8.5536e-01,  3.4215e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-1.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
         [ 0.0000e+00, -5.0000e-01,  8.6603e-01,  3.4641e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-9.8769e-01, -7.8217e-02,  1.3548e-01,  5.4190e-01],
         [ 1.5643e-01, -4.9384e-01,  8.5536e-01,  3.4215e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-9.5106e-01, -1.5451e-01,  2.6762e-01,  1.0705e+00],
         [ 3.0902e-01, -4.7553e-01,  8.2364e-01,  3.2946e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-8.9101e-01, -2.2700e-01,  3.9317e-01,  1.5727e+00],
         [ 4.5399e-01, -4.4550e-01,  7.7163e-01,  3.0865e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-8.0902e-01, -2.9389e-01,  5.0904e-01,  2.0361e+00],
         [ 5.8779e-01, -4.0451e-01,  7.0063e-01,  2.8025e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-7.0711e-01, -3.5355e-01,  6.1237e-01,  2.4495e+00],
         [ 7.0711e-01, -3.5355e-01,  6.1237e-01,  2.4495e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-5.8779e-01, -4.0451e-01,  7.0063e-01,  2.8025e+00],
         [ 8.0902e-01, -2.9389e-01,  5.0904e-01,  2.0361e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-4.5399e-01, -4.4550e-01,  7.7163e-01,  3.0865e+00],
         [ 8.9101e-01, -2.2700e-01,  3.9317e-01,  1.5727e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-3.0902e-01, -4.7553e-01,  8.2364e-01,  3.2946e+00],
         [ 9.5106e-01, -1.5451e-01,  2.6762e-01,  1.0705e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-1.5643e-01, -4.9384e-01,  8.5536e-01,  3.4215e+00],
         [ 9.8769e-01, -7.8217e-02,  1.3548e-01,  5.4190e-01],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[-6.1232e-17, -5.0000e-01,  8.6603e-01,  3.4641e+00],
         [ 1.0000e+00, -3.0616e-17,  5.3029e-17,  2.1212e-16],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 1.5643e-01, -4.9384e-01,  8.5536e-01,  3.4215e+00],
         [ 9.8769e-01,  7.8217e-02, -1.3548e-01, -5.4190e-01],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 3.0902e-01, -4.7553e-01,  8.2364e-01,  3.2946e+00],
         [ 9.5106e-01,  1.5451e-01, -2.6762e-01, -1.0705e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 4.5399e-01, -4.4550e-01,  7.7163e-01,  3.0865e+00],
         [ 8.9101e-01,  2.2700e-01, -3.9317e-01, -1.5727e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 5.8779e-01, -4.0451e-01,  7.0063e-01,  2.8025e+00],
         [ 8.0902e-01,  2.9389e-01, -5.0904e-01, -2.0361e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 7.0711e-01, -3.5355e-01,  6.1237e-01,  2.4495e+00],
         [ 7.0711e-01,  3.5355e-01, -6.1237e-01, -2.4495e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 8.0902e-01, -2.9389e-01,  5.0904e-01,  2.0361e+00],
         [ 5.8779e-01,  4.0451e-01, -7.0063e-01, -2.8025e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 8.9101e-01, -2.2700e-01,  3.9317e-01,  1.5727e+00],
         [ 4.5399e-01,  4.4550e-01, -7.7163e-01, -3.0865e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 9.5106e-01, -1.5451e-01,  2.6762e-01,  1.0705e+00],
         [ 3.0902e-01,  4.7553e-01, -8.2364e-01, -3.2946e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

        [[ 9.8769e-01, -7.8217e-02,  1.3548e-01,  5.4190e-01],
         [ 1.5643e-01,  4.9384e-01, -8.5536e-01, -3.4215e+00],
         [ 0.0000e+00,  8.6603e-01,  5.0000e-01,  2.0000e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]],
       dtype=torch.float64)
    """
    # print(render_poses.shape, render_poses.ndimension())
    # torch.Size([40, 4, 4]) 3

    # In debug mode, return extremely tiny images
    if debug:
        H = H // 32
        W = W // 32
        #focal = focal / 32.0
        intrinsics[:2] = intrinsics[:2] / 32.0
        imgs = [
            torch.from_numpy(
                cv2.resize(imgs[i], dsize=(25, 25), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
        if frontal_imgs:
            frontal_imgs = [
                torch.from_numpy(
                    cv2.resize(frontal_imgs[i], dsize=(25, 25), interpolation=cv2.INTER_AREA)
                )
                for i in range(frontal_imgs.shape[0])
            ]
            frontal_imgs = torch.stack(frontal_imgs, 0)

        poses = torch.from_numpy(poses)

        return imgs, poses, render_poses, [H, W, intrinsics], i_split, frontal_imgs


    if half_res:
        # TODO: resize images using INTER_AREA (cv2)
        H = H // 2
        W = W // 2
        #focal = focal / 2.0
        intrinsics[:2] = intrinsics[:2] * 0.5
        imgs = [
            torch.from_numpy(
                #cv2.resize(imgs[i], dsize=(400, 400), interpolation=cv2.INTER_AREA)
                cv2.resize(imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
        if load_frontal_faces:
            frontal_imgs = [
                torch.from_numpy(
                    #cv2.resize(imgs[i], dsize=(400, 400), interpolation=cv2.INTER_AREA)
                    cv2.resize(frontal_imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
                )
                for i in range(frontal_imgs.shape[0])
            ]
            frontal_imgs = torch.stack(frontal_imgs, 0)

    else:
        imgs = [
            torch.from_numpy(imgs[i]
                # cv2.resize(imgs[i], dsize=(400, 400), interpolation=cv2.INTER_AREA)
                #cv2.resize(imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
        if load_frontal_faces:
            frontal_imgs = [
                torch.from_numpy(frontal_imgs[i]
                                 # cv2.resize(imgs[i], dsize=(400, 400), interpolation=cv2.INTER_AREA)
                                 # cv2.resize(imgs[i], dsize=(H, W), interpolation=cv2.INTER_AREA)
                                 )
                for i in range(frontal_imgs.shape[0])
            ]
            frontal_imgs = torch.stack(frontal_imgs, 0)

    poses = torch.from_numpy(poses)
    expressions = torch.from_numpy(expressions)
    bboxs[:,0:2] *= H
    bboxs[:,2:4] *= W
    bboxs = np.floor(bboxs)
    bboxs = torch.from_numpy(bboxs).int()
    print("Done with data loading")

    return imgs, poses, render_poses, [H, W, intrinsics], i_split, expressions, frontal_imgs, bboxs
