import torch


import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

class SoloDataModule(pl.LightningDataModule):
    """
    Data module for the SOLO model using PyTorch Lightning.
    """

    def __init__(self, dataset, batch_size=64, validation_fraction=0.2):
        """
        Initialize the SoloDataModule.

        Args:
            dataset (Dataset): The dataset containing the data.
            batch_size (int, optional): Batch size for data loading. Defaults to 64.
            validation_fraction (float, optional): Fraction of data to be used for validation. Defaults to 0.2.
        """
        super().__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction

    def setup(self, stage=None):
        """
        Split the dataset into training and validation sets.

        Args:
            stage (str, optional): 'fit' or 'test'. If 'fit', splits the dataset into training and validation sets.
        """

        # Calculate the number of samples for validation
        validation_count = int(self.validation_fraction * len(self.dataset))
        
        # Calculate the number of samples for training
        training_count = len(self.dataset) - validation_count
        
        # Randomly split the dataset into training and validation sets
        self.train_dataset, self.val_dataset = random_split(self.dataset, [training_count, validation_count])

    def collate_fn(self, batch):
        """
        Collate function to format the batch data.

        Args:
            batch (list): List of tuples containing (index, image, labels, masks, bounding_boxes).

        Returns:
            tuple: A tuple containing indices, images, labels, masks, and bounding_boxes.
        """
        
        # Unzip the batch data into separate lists
        indices, images, labels, masks, bounding_boxes = list(zip(*batch))
        
        # Return the collated data
        return indices, torch.stack(images), labels, masks, bounding_boxes

    def train_dataloader(self):
        """
        Create and return the training data loader.

        Returns:
            DataLoader: The training data loader.
        """
        
        return DataLoader(self.train_dataset, collate_fn=self.collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        """
        Create and return the validation data loader.

        Returns:
            DataLoader: The validation data loader.
        """
        
        return DataLoader(self.val_dataset, collate_fn=self.collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=4)