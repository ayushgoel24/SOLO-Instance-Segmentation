from utils import DataPreprocessUtils
import torch

class SOLOLoss:
    
    @staticmethod
    def compute_loss(category_predictions, category_ground_truths, mask_predictions, mask_ground_truths, active_masks, category_loss_config, mask_loss_config):
        """
        Compute the SOLO loss given category and mask predictions and their corresponding ground truths.
        
        Args:
            category_predictions (list): Predicted categories for each FPN level. Shape: (batch_size, 3, S, S)
            category_ground_truths (list): Ground truth categories for each FPN level. Shape: (S, S)
            mask_predictions (list): Predicted masks for each FPN level. Shape: (batch_size, S^2, 2*feature_height, 2*feature_width)
            mask_ground_truths (list): Ground truth masks for each FPN level. Shape: (S^2, 2*feature_height, 2*feature_width)
            active_masks (list): Active masks indicating the presence of objects. Shape: (S^2,)
        
        Returns:
            tuple: Total loss, category loss, and mask loss.
        """
        # Ensure consistent shape for predictions and ground truths
        reshaped_category_preds, reshaped_category_targets, reshaped_mask_preds, reshaped_mask_targets, reshaped_active_masks = DataPreprocessUtils.get_consistent_shape(
            category_predictions, category_ground_truths, mask_predictions, mask_ground_truths, active_masks)

        # Compute the focal loss for categories
        focal_loss = -torch.mean(
            reshaped_category_targets * category_loss_config["alpha"] * (1 - reshaped_category_preds) ** category_loss_config["gamma"] * torch.log(reshaped_category_preds + 1e-12)
            + (1 - reshaped_category_targets) * (1 - category_loss_config["alpha"]) * reshaped_category_preds ** category_loss_config["gamma"] * torch.log(1 - reshaped_category_preds + 1e-12)
        )
        
        # Initialize mask loss and counters
        total_mask_loss = 0
        total_active_masks = 0
        total_dice_loss = 0

        # Compute the Dice loss for masks
        for mask_pred, mask_gt, active_mask in zip(reshaped_mask_preds, reshaped_mask_targets, reshaped_active_masks):
            total_active_masks += active_mask.sum()
            
            # Skip if no active masks
            if active_mask.sum() == 0:
                continue
            
            # Compute mask scores for active masks
            active_mask_pred = mask_pred[active_mask.nonzero(as_tuple=True)]
            active_mask_gt = mask_gt[active_mask.nonzero(as_tuple=True)]
            intersection = (active_mask_pred * active_mask_gt).sum()
            union = (active_mask_pred ** 2).sum() + (active_mask_gt ** 2).sum()
            
            # Compute Dice coefficient
            dice_coefficient = torch.divide(2 * intersection, union + 1e-12)
            
            # Compute Dice loss
            dice_loss = 1 - dice_coefficient
            total_dice_loss += dice_loss.sum()

        # Compute average mask loss
        total_mask_loss = total_dice_loss / (total_active_masks + 1e-12)
        
        # Compute the total loss as a weighted sum of category and mask losses
        total_loss = category_loss_config["weight"] * focal_loss + mask_loss_config["weight"] * total_mask_loss

        return total_loss, focal_loss, total_mask_loss
