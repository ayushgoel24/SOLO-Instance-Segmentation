from model import SOLO
import pytorch_lightning as pl
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
from utils import ConfigurationUtils, SoloDataset, SoloDataModule
import torch

imgs_path, masks_path, labels_path, bboxes_path = ConfigurationUtils.load_file_paths()
data_paths = [imgs_path, masks_path, labels_path, bboxes_path]

# Create a dictionary to store the cumulative count of labels for each image
process_label_num = {}
j = 0  # Initialize the counter

# Iterate over each image and update the dictionary with the cumulative count of labels
for i in range(len(images)):
    process_label_num[i] = j
    j = j + len(labels[i])

# Initialize the SoloDataset with the given data and transformations
dataset = SoloDataset(data_paths=data_paths, mask_idx_dict=process_label_num)

# Initialize the solo_datamodule with the dataset and a batch size of 2
solo_lightning_module = SoloDataModule(dataset, batch_size=2)

# Check if CUDA is available and set the device accordingly
cuda_available = torch.cuda.is_available()
mps_available = torch.backends.mps.is_available()
device = torch.device("cuda" if cuda_available else "mps" if mps_available else "cpu")

# Load the SOLO model from a checkpoint and move it to the appropriate device
# model = SOLO.load_from_checkpoint("/solo_checkpoints/epoch=14-step=19590.ckpt")
model = SOLO()

# Setup callbacks for training
# Model checkpointing callback to save the model weights during training
checkpoint_callback = pl_callbacks.ModelCheckpoint(dirpath='checkpoints/')

# Monitor the learning rate during training
lr_monitor = LearningRateMonitor(logging_interval='step')

# TensorBoard logger to log training metrics and visualize them in TensorBoard
tb_logger = pl_loggers.TensorBoardLogger("/save_dir", name="solo")

# Initialize the PyTorch Lightning Trainer with the given configurations
trainer = pl.Trainer(
    accelerator='gpu',  # Use GPU for training
    devices='auto',  # Automatically select the available devices
    logger=tb_logger,  # Use the TensorBoard logger
    max_epochs=40,  # Train for a maximum of 40 epochs
    callbacks=[checkpoint_callback, lr_monitor]  # Use the defined callbacks
)

# Start the training process
trainer.fit(model, solo_lightning_module, ckpt_path='/solo_checkpoints/epoch=14-step=19590.ckpt')