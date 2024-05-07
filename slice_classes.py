import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# For reference:  https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

class SliceCrop(object):
    """

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
        borders (tuple): desired border position. Should abut on the previous crop
            order is (top border, left border)
    """

    def __init__(self, output_size, borders):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

        if isinstance(borders, int):
            self.borders = (borders, borders)
        else:
            assert isinstance(borders, tuple)
            self.borders = borders

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = self.borders[0]
        left = self.borders[1]


        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image}




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
        self.image_names = [name for name in os.listdir(self.root_dir) if not name.startswith('.')]
        
        
        if imitation_dir is None:
            self.imitation_dir = root_dir
        else:
            self.imitation_dir = imitation_dir
            self.images_names += [name for name in os.listdir(self.imitation_dir) if not name.startswith('.')]
            

        self.transform = transform

    def __len__(self):
        ''' How many images? '''
        return len(self.image_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        img_path = os.path.join(self.root_dir, self.image_names[idx])
        img_name = Path(img_path).stem
        # print('img_path',img_path)
        image = io.imread(img_path)
        sample = {'image': image,'name':img_name, 'path':img_path}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def find_painting_index_by_name(self, desired_name):
        path_names = Path(self.root_dir).parts

        if 'Processed' in path_names:
            if '_cropped' not in desired_name:
                split_name = desired_name.split('.')
                desired_name = split_name[0] + '_cropped.' + split_name[-1]
        if 'Descreened' in path_names:
            if '_descreened' not in desired_name:
                split_name = desired_name.split('.')
                desired_name = split_name[0] + '_descreened.' + split_name[-1]

        desired_index = np.nan
        for ii, name in enumerate(self.image_names):
            if name == desired_name:
                desired_index = ii

        return desired_index
    
    # def listdir_nohidden(path):
    #     for f in os.listdir(path):
    #         if not f.startswith('.'):
    #             yield f



class Rescale(object):
    """
    Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w),preserve_range = True).astype(np.uint8)

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively

        return {'image': img}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image)}
