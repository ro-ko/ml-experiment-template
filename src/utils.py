#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch
from loguru import logger
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def set_random_seed(seed, device):
    '''
    for reproducibility (always not guaranteed in pytorch)
    [1] https://pytorch.org/docs/stable/notes/randomness.html
    [2] https://hoya012.github.io/blog/reproducible_pytorch/
    '''

    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def log_param(param):
    for key, value in param.items():
        if type(value) is dict:
            for in_key, in_value in value.items():
                logger.info('{:20}:{:>50}'.format(in_key, '{}'.format(in_value)))
        else:
            logger.info('{:20}:{:>50}'.format(key, '{}'.format(value)))

def show(imgs, labels):
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)

    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0,i].imshow(np.array(img))
        axs[0,i].set_title(labels[i])
        axs[0,i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig('image')

def createDir(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
    except OSError:
        print("Error: Failed to create the directory.")