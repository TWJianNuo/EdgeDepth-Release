"""
    Sample Run:
    python test_simple.py --image_path ../sfm_in_video/data/kitti_split1/training/prev_2 --model_path weights/ --num_layers 50
"""

# Copyright Niantic 2020. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the DepthHints licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import time

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist

output_folder = "output"

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Depth Hints models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_path', type=str,
                        help='path to a pretrained model to use', required=True)
    parser.add_argument('--num_layers', help='number of resnet layers in the model',
                        type=int, choices=[18, 50])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="png")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--min_depth",
                        type=float,
                        help="minimum depth",
                        default=0.1)
    parser.add_argument("--max_depth",
                        type=float,
                        help="maximum depth",
                        default=100.0)

    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_path is not None, \
        "You must specify the --model_path parameter"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_path = args.model_path
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder with {} layers".format(args.num_layers))
    encoder = networks.ResnetEncoder(args.num_layers, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = sorted(glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext))))
        output_directory = args.image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    output_directory = os.path.join(output_folder, output_directory)
    output_directory= output_directory.replace("../sfm_in_video/data/", "")

    print("-> Predicting on {:d} test images".format(len(paths)))
    print("Saving to {}".format(output_directory))
    make_dirs(output_directory)

    output_depth_image_directory = output_directory + "_depth_image"
    output_disp_image_directory  = output_directory + "_disp_image"
    make_dirs(output_depth_image_directory)
    make_dirs(output_disp_image_directory)

    start = time.process_time()
    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            try:
                input_image = pil.open(image_path).convert('RGB')
            except:
                print("{} not found!".format(image_path))
                continue
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]

            # Convert disparity to depth in network resolution
            scaled_disp, scaled_depth = disp_to_depth(disp, args.min_depth, args.max_depth)
            scaled_disp_np            = scaled_disp [0,0].cpu().numpy()
            scaled_depth_np           = scaled_depth[0,0].cpu().numpy()

            # Conver disparity in network resolution to disparity in original resolution as image
            disp_original = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Convert disparity original resolution to depth original resolution
            scaled_disp_original, scaled_depth_original = disp_to_depth(disp_original, args.min_depth, args.max_depth)
            scaled_disp_original_np                     = scaled_disp_original [0,0].cpu().numpy()
            scaled_depth_original_np                    = scaled_depth_original[0,0].cpu().numpy()

            # Saving numpy file of depth in network's resolution
            output_name        = os.path.splitext(os.path.basename(image_path))[0]
            # save_numpy(output_directory, output_name, scaled_disp_np , suffix= "disp_network")
            # save_numpy(output_directory, output_name, scaled_depth_np, suffix= "depth_network")
            # save_numpy(output_directory, output_name, scaled_disp_original_np , suffix= "disp")
            save_numpy(output_directory, output_name, scaled_depth_original_np, suffix= "depth")

            # Saving colormapped depth image in original resolution
            save_image(output_disp_image_directory , output_name, scaled_disp_original_np , suffix= "disp")
            save_image(output_depth_image_directory, output_name, scaled_depth_original_np, suffix= "depth")

            if ((idx+1) % 500 == 0 or idx == len(paths)-1):
                print("Time= {:.2f}s   Processed {:d} of {:d} images - saved prediction to {}".format(
                    time.process_time() - start, idx + 1, len(paths), os.path.join(output_directory, output_name)))

    print('-> Done!')

def make_dirs(output_directory):
    if not os.path.exists(output_directory):
        print("Making directory {}".format(output_directory))
        os.makedirs(output_directory)

def save_numpy(output_directory, output_name, np_array, suffix= None):
    if suffix is None:
        name_dest     = os.path.join(output_directory, "{}.npy"   .format(output_name))
    else:
        name_dest     = os.path.join(output_directory, "{}_{}.npy".format(output_name, suffix))

    np.save(name_dest, np_array)

def save_image(output_directory, output_name, np_array, suffix= None):
    vmax = np.percentile(np_array, 95)
    normalizer = mpl.colors.Normalize(vmin=np_array.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(np_array)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)

    if suffix is None:
        name_dest_im = os.path.join(output_directory, "{}.png"   .format(output_name))
    else:
        name_dest_im = os.path.join(output_directory, "{}_{}.png".format(output_name, suffix))

    im.save(name_dest_im)

if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
