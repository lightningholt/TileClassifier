import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transform, utils

# For reference:  https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

class AbstractArtData(Dataset):
    """
    Images of abstract art.

    Have to define an __init__ function and a __getitem__ function
    """

    def __init__(self, root_dir, imitation_dir=None, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images
             (if imitation_dir is not specified)
            imitation_dir (string, optional): Directory with imitation images
             (if supplied)
            transform (callable, optional): Transforms to applied on a sample
        """

        self.root_dir = root_dir
        self.image_names = os.listdir(self.root_dir)

        if imitation_dir is None:
            self.imitation_dir = root_dir
        else:
            self.imitation_dir = imitation_dir]
            self.images_names += os.listdir(self.imitation_dir)

        self.transform = transform

    def __len__(self):
        ''' How many images? '''
        return len(self.image_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        img_name = os.path.join(self.root_dir, self.image_names[idx])
        image = io.imread(img_name)
        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """
    Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        
