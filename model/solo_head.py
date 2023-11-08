from backbone import Backbone
from category_branch import CategoryBranch
from mask_branch import MaskBranch
from solo_loss import SOLOLoss

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import pytorch_lightning as pl
import yaml

class SOLO(pl.LightningModule):
    
    with open('config.yaml', 'r') as yaml_file:
        _default_cfg = yaml.safe_load(yaml_file)

    def __init__(self, **kwargs):
        """
        Initialize the SOLO model.

        Args:
            **kwargs (dict): Keyword arguments to customize model configuration.
        """
        
        for k, v in {**self._default_cfg, **kwargs}.items():
            setattr(self, k, v)

        # Initialize backbone
        self.backbone = Backbone()
        
        # Define category branch
        self.category_branch = CategoryBranch(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, padding=1, num_layers=self.stacked_convs)
        
        # Define mask branch
        self.mask_branch = MaskBranch(in_channels=self.in_channels + 2, out_channels=self.in_channels, kernel_size=3, padding=1, num_layers=self.stacked_convs, num_grids=self.num_grids)

        # Initialize loss and metrics lists
        self.training_losses, self.validation_losses = [], []
        self.training_categories, self.validation_categories = [], []
        self.training_masks, self.validation_masks = [], []
        self.training_step_outputs, self.validation_step_outputs = [], []

    def add_channels(self, layer):
        """
        Add coordinate channels to the input tensor.

        Args:
            layer (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with additional coordinate channels.
        """
        x, y = torch.meshgrid(torch.linspace(-1, 1, layer.shape[-2]), torch.linspace(-1, 1, layer.shape[-1]))
        x = x.unsqueeze(0).expand(layer.shape[0], -1, -1, -1)
        y = y.unsqueeze(0).expand(layer.shape[0], -1, -1, -1)
        return torch.cat((layer, x.to(self.device), y.to(self.device)), dim=1)

    def forward_internal(self, layers):
        """
        Forward pass for a single FPN level.

        Args:
            layers (tuple): Tuple containing the FPN layer index and tensor.

        Returns:
            tuple: Tuple containing category and mask predictions.
        """
        layer_idx, layer = layers
        size_layer = self.num_grids[layer_idx]
        align = F.interpolate(layer, size=(size_layer, size_layer), mode="bilinear")
        coord_conv = self.add_channels(layer)
        return self.category_branch(align), self.mask_branch(coord_conv, layer_idx)

    def forward(self, images, eval_mode=True):
        """
        Forward pass of the SOLO model.

        Args:
            images (torch.Tensor): Input images.
            eval_mode (bool): Flag to indicate evaluation mode.

        Returns:
            tuple: Tuple containing category and mask predictions.
        """
        if eval_mode:
            self.eval()
        else:
            self.train()

        feature_pyramid = self.backbone(images)
        outputs = list(map(self.forward_internal, enumerate(feature_pyramid)))
        category_preds = [out[0] for out in outputs]
        mask_preds = [out[1] for out in outputs]

        if eval_mode:
            with torch.no_grad():
                category_preds = [pred.permute(0, 2, 3, 1) for pred in category_preds]
                mask_preds = [F.interpolate(mask, size=(200, 272)) for mask in mask_preds]

        return category_preds, mask_preds

    def grid_idxs_constrained(self, bbox, label, fpn_grid):
        """
        Constrain grid indices based on bounding box and label.

        Args:
            bbox (list): Bounding box coordinates [x1, y1, x2, y2].
            label (int): Label of the object.
            fpn_grid (torch.Tensor): Grid indices.

        Returns:
            tuple: Tuple containing updated grid indices and bounds.
        """
        num_grid = fpn_grid.shape[0]
        h_center, w_center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        h_scale, w_scale = self.epsilon * (bbox[2] - bbox[0]), self.epsilon * (bbox[3] - bbox[1])
        h_half, w_half = h_scale / 2, w_scale / 2

        top_idx = max(0, int((h_center - h_half) / 800 * num_grid))
        bottom_idx = min(num_grid - 1, int((h_center + h_half) / 800 * num_grid))
        left_idx = max(0, int((w_center - w_half) / 1088 * num_grid))
        right_idx = min(num_grid - 1, int((w_center + w_half) / 1088 * num_grid))

        center_h, center_w = int(h_center / 800 * num_grid), int(w_center / 1088 * num_grid)
        top, bottom = max(top_idx, center_h - 1), min(bottom_idx, center_h + 1)
        left, right = max(left_idx, center_w - 1), min(right_idx, center_w + 1)

        fpn_grid[top:bottom + 1, left:right + 1] = label
        return fpn_grid, top, bottom, left, right

    def generate_category_and_mask_targets(self, bounding_boxes, labels, masks):
        """
        Generate category and mask targets.

        Args:
            bounding_boxes (list): List of bounding boxes.
            labels (list): List of labels.
            masks (list): List of masks.

        Returns:
            tuple: Tuple containing category targets, mask targets, and active masks.
        """
        category_targets_list, mask_targets_list, active_masks_list = [], [], []
        image_dimensions = (800, 1088)

        for image_boxes, image_labels, image_mask in zip(bounding_boxes, labels, masks):
            # Calculate box properties: widths, heights, areas, centers
            widths = image_boxes[:, 2] - image_boxes[:, 0]
            heights = image_boxes[:, 3] - image_boxes[:, 1]
            areas = torch.sqrt(widths * heights)
            centers_x = (image_boxes[:, 2] + image_boxes[:, 0]) / 2
            centers_y = (image_boxes[:, 3] + image_boxes[:, 1]) / 2

            # Scaled centers and dimensions
            centers_x_scaled = [centers_x * (grid / image_dimensions[1]) for grid in self.num_grids]
            centers_y_scaled = [centers_y * (grid / image_dimensions[0]) for grid in self.num_grids]
            widths_scaled = [widths * (grid / image_dimensions[1]) for grid in self.num_grids]
            heights_scaled = [heights * (grid / image_dimensions[0]) for grid in self.num_grids]

            levels = [torch.logical_and(self.scale_ranges[i][0] < areas, areas < self.scale_ranges[i][1]).int() for i in range(5)]

            image_category_targets, image_mask_targets, image_active_masks = [], [], []

            for level_id, level in enumerate(levels):
                height_by_stride, width_by_stride = image_dimensions[0] // self.strides[level_id], image_dimensions[1] // self.strides[level_id]
                scaled_mask = F.interpolate(image_mask.unsqueeze(0), size=(2 * height_by_stride, 2 * width_by_stride), mode='nearest').squeeze(0).to('cpu')

                category_target = torch.zeros(self.num_grids[level_id], self.num_grids[level_id]).to('cpu')
                mask_target = torch.zeros(self.num_grids[level_id] ** 2, 2 * height_by_stride, 2 * width_by_stride).to('cpu')
                active_mask = torch.zeros(self.num_grids[level_id] ** 2).to('cpu')

                for label_id, is_active in enumerate(level):
                    if is_active.item() == 0:
                        continue

                    # Scale down the boxes and bound them
                    x_tl, y_tl = max(torch.round(centers_x_scaled[level_id][label_id] - 0.2 * widths_scaled[level_id][label_id] / 2), centers_x_scaled[level_id][label_id] - 1).int().item(), max(torch.round(centers_y_scaled[level_id][label_id] - 0.2 * heights_scaled[level_id][label_id] / 2), centers_y_scaled[level_id][label_id] - 1).int().item()
                    x_br, y_br = min(torch.round(centers_x_scaled[level_id][label_id] + 0.2 * widths_scaled[level_id][label_id] / 2), centers_x_scaled[level_id][label_id] + 1).int().item(), min(torch.round(centers_y_scaled[level_id][label_id] + 0.2 * heights_scaled[level_id][label_id] / 2), centers_y_scaled[level_id][label_id] + 1).int().item()

                    mesh_x, mesh_y = torch.meshgrid(torch.arange(y_tl, y_br + 1, device='cpu'), torch.arange(x_tl, x_br + 1, device='cpu'))

                    category_target[mesh_x, mesh_y] = image_labels[label_id].item()
                    mask_target[mesh_x.flatten() * self.num_grids[level_id] + mesh_y.flatten()] = scaled_mask[label_id]
                    active_mask[mesh_x.flatten() * self.num_grids[level_id] + mesh_y.flatten()] = 1.

                image_category_targets.append(category_target)
                image_mask_targets.append(mask_target)
                image_active_masks.append(active_mask)

            category_targets_list.append(image_category_targets)
            mask_targets_list.append(image_mask_targets)
            active_masks_list.append(image_active_masks)

        return category_targets_list, mask_targets_list, active_masks_list

    def points_nms(self, heat_map, kernel=2):
        """
        Apply point-wise Non-Maximum Suppression (NMS) on the given heat map.

        Args:
            heat_map (torch.Tensor): The input heat map.
            kernel (int, default=2): The size of the max pooling kernel. Must be 2.

        Returns:
            torch.Tensor: The heat map after applying point-wise NMS.
        """
        # Ensure the kernel size is 2
        assert kernel == 2, "Kernel size must be 2"

        # Apply max pooling to the heat map to get the local maxima
        local_maxima = F.max_pool2d(
            heat_map, (kernel, kernel), stride=1, padding=1)

        # Create a mask where the original heat map values are equal to the local maxima
        # This mask will have 1s where the condition is True and 0s elsewhere
        mask = (local_maxima[:, :, :-1, :-1] == heat_map).float()

        # Multiply the original heat map with the mask to suppress non-maximum points
        suppressed_heat_map = heat_map * mask

        return suppressed_heat_map
    
    def reshape_and_flatten_tensors(self, predicted_categories, target_categories, predicted_masks, target_masks, active_masks):
        """
        Reshape and flatten tensors for consistent processing.

        Args:
            predicted_categories (list): List of predicted category tensors.
            target_categories (list): List of target category tensors.
            predicted_masks (list): List of predicted mask tensors.
            target_masks (list): List of target mask tensors.
            active_masks (list): List of active mask tensors indicating the presence of objects.

        Returns:
            tuple: Reshaped and flattened tensors for categories, masks, and active masks.
        """
        # Reshape predicted categories
        reshaped_predicted_categories = torch.cat([p.permute(0, 2, 3, 1).reshape(-1, 3) for p in predicted_categories]).flatten()

        # Reshape target categories
        target_category_flattened = torch.cat([torch.cat(tc).flatten() for tc in zip(*target_categories)])
        target_category_reshaped = F.one_hot(target_category_flattened.type(torch.int64), num_classes=4)[:, 1:].flatten()

        # Flatten masks
        flattened_predicted_masks = [p.flatten(start_dim=0, end_dim=1) for p in predicted_masks]
        concatenated_target_masks = [torch.cat(tm) for tm in zip(*target_masks)]
        concatenated_active_masks = [torch.cat(am) for am in zip(*active_masks)]

        # Cleanup to free up memory
        del predicted_categories, target_categories, predicted_masks, target_masks, active_masks
        gc.collect()

        return reshaped_predicted_categories, target_category_reshaped, flattened_predicted_masks, concatenated_target_masks, concatenated_active_masks

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler for training.
        
        Returns:
            dict: Dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[27, 33], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
