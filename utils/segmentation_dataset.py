import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms


import torch
from torch.utils.data import Dataset
from utils import ConfigurationUtils, FileUtils

class SoloDataset(Dataset):
    """Dataset class for the SOLO model."""

    def __init__(self, data_paths, mask_index_dict=None):
        """
        Initialize the SOLO dataset.

        Args:
            images (list): List of image data.
            labels (list): List of label data.
            bounding_boxes (list): List of flattened bounding box data.
            masks (list): List of mask data.
            image_transform (callable, optional): Transformation function to be applied to images.
            mask_transform (callable, optional): Transformation function to be applied to masks.
            mask_index_dict (dict, optional): Dictionary containing mask indices.
        """

        # Initialize the dataset with given file paths
        self.images_path, self.masks_path, self.labels_path, self.bboxes_path = data_paths

        # Load data from files
        self.image_data = FileUtils.read_h5_file(self.images_path)
        self.mask_data = FileUtils.read_h5_file(self.masks_path)
        self.label_data = np.load(self.labels_path, allow_pickle=True)
        self.bbox_data = np.load(self.bboxes_path, allow_pickle=True)

        self.image_transform = ConfigurationUtils.load_transform("image_transform")
        self.mask_transform = ConfigurationUtils.load_transform("mask_transform")
        
        # Scaling factors and padding for bounding boxes
        self.x_scaling_factor = 800 / 300
        self.y_scaling_factor = 1066 / 400
        self.x_padding = 11
        self.mask_index_dict = mask_index_dict

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, index):
        """Retrieve a sample from the dataset given an index."""
        
        # Convert image, label, and bounding box data to tensors
        image = (torch.from_numpy(self.images[index].astype(float)) / 255).float()
        labels = torch.from_numpy(self.labels[index]).float()
        bounding_boxes = torch.from_numpy(self.bounding_boxes[index]).float()
      
        # Adjust bounding box coordinates based on scaling and padding
        bounding_boxes[:, 0] *= self.x_scaling_factor
        bounding_boxes[:, 0] += self.x_padding
        bounding_boxes[:, 2] *= self.x_scaling_factor
        bounding_boxes[:, 2] += self.x_padding
        bounding_boxes[:, 1] *= self.y_scaling_factor
        bounding_boxes[:, 3] *= self.y_scaling_factor

        # Retrieve the corresponding mask using the mask index
        num_objects = labels.shape[0]
        mask_start_index = self.mask_index_dict[index]
        masks = torch.from_numpy(self.masks[mask_start_index:mask_start_index + num_objects].astype(float))

        # Apply transformations if provided
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            masks = self.mask_transform(masks)

        return index, image, labels, masks, bounding_boxes