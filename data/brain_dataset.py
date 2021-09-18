import os
import torch
from data.base_dataset import BaseDataset

class BrainDataset(BaseDataset):
    """A dataset class for brain image dataset.

    It assumes that the directory '/path/to/data/train' contains brain image slices
    in torch.tensor format to speed up the I/O. Otherwise, you can load MRI brain images
    using nibabel package and preprocess the slices during loading period.
    """

    def __init__(self, opt):
        """
        Initialize this dataset class.
        """
        BaseDataset.__init__(self, opt)
        self.sub_list = os.listdir(opt.dataroot)
        self.opt = opt

        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

    def __getitem__(self, index):

        # read a pseudo 3D slice given a random integer index
        self.data = torch.load(self.root + self.sub_list[index])

        brain = self.data['brain']
        lesion = self.data['lesion']

        gt_img = brain.clone()

        # lesion as holes in brain
        brain[lesion == 1] = 0

        return {'brain': brain, 'lesion': lesion, 'gt': gt_img, 'path': self.sub_list[index]}

    def __len__(self):
        return len(self.sub_list)

    def modify_commandline_options(parser, is_train):
        """
        Add any new dataset-specific options, and rewrite default values for existing options.
        """
        return parser
