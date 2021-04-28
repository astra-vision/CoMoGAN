#!/usr/bin/env python
# coding: utf-8

import pathlib
import torch
import yaml
import sys
import os

from math import pi
from PIL import Image
from munch import Munch
from argparse import ArgumentParser as AP
from torchvision.transforms import ToPILImage

from torchvision.transforms import ToTensor

p_mod = str(pathlib.Path('.').absolute())
sys.path.append(p_mod.replace("/scripts", ""))

from data.base_dataset import get_transform
from networks import create_model

def printProgressBar(i, max, postText):
    n_bar = 20 #size of progress bar
    j= i/max
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}%  {postText}")
    sys.stdout.flush()

def inference(model, opt, A_path, phi):
    t_phi = torch.tensor(phi)
    # Load the image
    A_img = Image.open(A_path).convert('RGB')
    # Apply image transformation
    A = get_transform(opt, convert=False)(A_img)
    # Normalization between -1 and 1
    img_real = (((ToTensor()(A)) * 2) - 1).unsqueeze(0)
    # Forward our image into the model with the specified ɸ
    img_fake = model.forward(img_real.cuda(0), t_phi.cuda(0))

    return ToPILImage()((img_fake[0].cpu() + 1) / 2)

def main(cmdline):

    if cmdline.phi < 0 or cmdline.phi > (2 * pi):
        raise ValueError("Value should be between [0,2𝜋]")

    if cmdline.checkpoint is None:
        # Load names of directories inside /logs
        p = pathlib.Path('./logs')
        list_run_id = [str(x).split('/')[1] for x in p.iterdir() if x.is_dir()]

        RUN_ID = list_run_id[0]
        root_dir = os.path.join('logs', RUN_ID, 'tensorboard', 'default', 'version_0')
        p = pathlib.Path(root_dir + '/checkpoints')
        # Load a list of checkpoints, use the last one by default
        list_checkpoint = [str(x).split('checkpoints/')[1] for x in p.iterdir() if 'iter' in str(x)]
        list_checkpoint.sort(reverse = True, key=lambda x: int(x.split('_')[1].split('.pth')[0]))

        CHECKPOINT = list_checkpoint[0]
    else:
        RUN_ID = (cmdline.checkpoint.split("/tensorboard")[0]).split("/")[-1]
        root_dir = cmdline.checkpoint.split("/checkpoints")[0]
        CHECKPOINT = cmdline.checkpoint.split('checkpoints/')[1]

    print("Load checkpoint {} from {}".format(CHECKPOINT, RUN_ID))

    # Load parameters
    with open(os.path.join(root_dir, 'hparams.yaml')) as cfg_file:
        opt = Munch(yaml.safe_load(cfg_file))

    opt.no_flip = True
    # Load parameters to the model, load the checkpoint
    model = create_model(opt)
    model = model.load_from_checkpoint((os.path.join(root_dir, 'checkpoints/', CHECKPOINT)))
    # Transfer the model to the GPU
    model.to('cuda');
    # Load paths of all files contain in /Day
    p = pathlib.Path(cmdline.load_path)
    dataset_paths = [str(x).replace(cmdline.load_path,'') for x in p.iterdir()]
    dataset_paths.sort()
    # Load only files that contained the given string
    sequence_name = []
    if cmdline.sequence is not None:
        for file in dataset_paths:
            if cmdline.sequence in file:
                sequence_name.append(file)
    else:
        sequence_name = dataset_paths

    # Create repository if it doesn't exist
    os.makedirs(os.path.dirname(cmdline.save_path), exist_ok=True)

    i = 0
    for path_img in sequence_name:
        printProgressBar(i, len(sequence_name), path_img)
        # Forward our image into the model with the specified ɸ
        out_img = inference(model, opt, cmdline.load_path + path_img, cmdline.phi)
        # Saving the generated image
        save_path = cmdline.save_path + path_img.split("/")[-1]
        out_img.save(save_path)
        i += 1

if __name__ == '__main__':
    ap = AP()
    ap.add_argument('--load_path', default='/datasets/waymo_comogan/val/sunny/Day/', type=str, help='Set a path to load the dataset to translate')
    ap.add_argument('--save_path', default='/CoMoGan/images/', type=str, help='Set a path to save the dataset')
    ap.add_argument('--sequence', default=None, type=str, help='Set a sequence, will only use the image that contained the string specified')
    ap.add_argument('--checkpoint', default=None, type=str, help='Set a path to the checkpoint that you want to use')
    ap.add_argument('--phi', default=0.0, type=float, help='Choose the angle of the sun 𝜙 between [0,2𝜋], which maps to a sun elevation ∈ [+30◦,−40◦]')
    main(ap.parse_args())
    print("\n")
