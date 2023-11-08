from model import SOLOLoss
import pytorch_lightning as pl
import torch

class SOLOTrainer(pl.LightningModule):

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        
        Args:
            batch (tuple): Current batch of data.
            batch_idx (int): Index of the current batch.
            
        Returns:
            dict: Dictionary containing the training loss and other metrics.
        """
        train_idx, train_images, train_labels, train_masks, train_bboxes = batch
        
        # Generate target categories, masks, and active masks for the current batch
        category_targets, mask_targets, active_targets = self.generate_targets(train_bboxes, train_labels, train_masks)
        
        # Forward pass: Get model predictions for categories and masks
        category_predictions, mask_predictions = self(train_images)
        
        # Compute the solo loss (combined category and mask loss)
        total_loss, category_loss, mask_loss = SOLOLoss.compute_loss(
            category_predictions=category_predictions,
            category_ground_truths=category_targets,
            mask_predictions=mask_predictions,
            mask_ground_truths=mask_targets,
            active_masks=active_targets,
            category_loss_config=self.cate_loss_cfg,
            mask_loss_config=self.mask_loss_cfg)
        
        # Log the losses for monitoring
        self.log("training_dice_loss", mask_loss, on_epoch=True)
        self.log("training_category_loss", category_loss, on_epoch=True)
        self.log("training_total_loss", total_loss, on_epoch=True, prog_bar=True)
        
        # Store the losses for further analysis
        self.training_step_outputs.append({'loss': total_loss, "dice_loss": mask_loss, "cate_loss": category_loss})
        
        return {'loss': total_loss, "dice_loss": mask_loss, "cate_loss": category_loss}
    
    def compute_epoch_loss(self, outputs):
        """
        Compute average loss for an epoch.
        
        Args:
            outputs (list): List of dictionaries containing loss values.
        
        Returns:
            float: Average loss for the epoch.
        """
        return torch.stack([x['loss'] for x in outputs]).mean().item()
    
    def on_train_epoch_end(self, outputs):
        """
        Callback function executed at the end of each training epoch.
        Computes and logs average losses for the epoch.
        
        Args:
            outputs (list): List of dictionaries containing training step outputs.
        """
        average_train_loss = self.compute_epoch_loss(outputs)
        average_dice_loss = self.compute_epoch_loss([x['dice_loss'] for x in outputs])
        average_category_loss = self.compute_epoch_loss([x['cate_loss'] for x in outputs])

        self.train_loss.append(average_train_loss)
        self.train_mask.append(average_dice_loss)
        self.train_cat.append(average_category_loss)

        print('Average Training Loss:', self.train_loss)
        print('Average Training Mask Loss:', self.train_mask)
        print('Average Training Categorical Loss:', self.train_cat)

        self.log("dice_loss", average_dice_loss, on_epoch=True)
        self.log("category_loss", average_category_loss, on_epoch=True)
        self.log("train_loss", average_train_loss, on_epoch=True, prog_bar=True)