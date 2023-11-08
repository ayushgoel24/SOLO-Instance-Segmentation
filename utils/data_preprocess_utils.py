import torch

class DataPreprocessUtils:

    @staticmethod
    def get_consistent_shape(category_predictions, category_targets, mask_predictions, mask_targets, active_masks):
        """
        Reshape the data to consistent shapes for further processing.
        
        Args:
            category_predictions (list): Predicted categories for each FPN level. Shape: (batch_size, 3, S, S)
            category_targets (list): Ground truth categories for each FPN level. Shape: (S, S)
            mask_predictions (list): Predicted masks for each FPN level. Shape: (batch_size, S^2, 2*feature_height, 2*feature_width)
            mask_targets (list): Ground truth masks for each FPN level. Shape: (S^2, 2*feature_height, 2*feature_width)
            active_masks (list): Active masks indicating the presence of objects. Shape: (S^2,)
        
        Returns:
            tuple: Reshaped category predictions, category targets, mask predictions, mask targets, and active masks.
        """
        # Reshape category predictions
        reshaped_category_preds = torch.cat([fpn.permute(0, 2, 3, 1).reshape(-1, 3) for fpn in category_predictions])
        reshaped_category_targets = torch.cat([tc.view(-1) for tc in category_targets])
        
        # Reshape mask predictions
        reshaped_mask_predictions = [fpn.view(fpn.size(0), -1, fpn.size(2), fpn.size(3)) for fpn in mask_predictions]
        
        # Reshape mask targets
        reshaped_mask_targets = [mt.view(mt.size(0), -1, mt.size(2), mt.size(3)) for mt in mask_targets]
        
        # Reshape active masks
        reshaped_active_masks = torch.cat(active_masks)
        
        return reshaped_category_preds, reshaped_category_targets, reshaped_mask_predictions, reshaped_mask_targets, reshaped_active_masks
