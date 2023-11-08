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
        pretrained_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True)
        self.backbone = pretrained_model.backbone

        # Initialize loss and metrics lists
        self.training_losses, self.validation_losses = [], []
        self.training_categories, self.validation_categories = [], []
        self.training_masks, self.validation_masks = [], []
        self.training_step_outputs, self.validation_step_outputs = [], []

        # Define category branch
        self.category_branch_head = self._make_layers(self.in_channels, self.in_channels, kernel_size=3, padding=1, num_layers=self.stacked_convs)
        self.category_branch_output = nn.Sequential(
            nn.Conv2d(self.in_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Define mask branch
        self.mask_branch_head = self._make_layers(self.in_channels + 2, self.in_channels, kernel_size=3, padding=1, num_layers=self.stacked_convs)
        self.mask_branch_output = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(self.in_channels, grid**2, kernel_size=1), nn.Sigmoid()) for grid in self.num_grids]
        )
        self.mask_branch_output = nn.Sequential(*self.mask_branch_output)

    def _make_layers(self, in_channels, out_channels, kernel_size, padding, num_layers):
        """
        Create a series of convolutional layers with Group Normalization.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            padding (int): Padding for the convolution.
            num_layers (int): Number of layers to stack.

        Returns:
            nn.Sequential: Sequential container of convolutional layers.
        """
        layers = [
            nn.Sequential(
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size, padding=padding, bias=False),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True)
            ) for i in range(num_layers)
        ]
        return nn.Sequential(*layers)

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
        mask_intermediate = self.mask_branch_head(coord_conv)
        mask_intermediate = F.interpolate(mask_intermediate, scale_factor=2, mode="bilinear")
        return self.category_branch_output(self.category_branch_head(align)), self.mask_branch_output[layer_idx](mask_intermediate)

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

        feature_pyramid = [v.detach() for v in self.backbone(images).values()]
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

    def generate_targets(self, bounding_boxes, labels, masks):
        """
        Generate category and mask targets.

        Args:
            bounding_boxes (list): List of bounding boxes.
            labels (list): List of labels.
            masks (list): List of masks.

        Returns:
            tuple: Tuple containing category targets, mask targets, and active masks.
        """
        category_targets, mask_targets, active_masks = [], [], []
        img_dim = (800, 1088)

        for boxes, lbls, mask in zip(bounding_boxes, labels, masks):
            # Calculate width, height, area, and centers
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            areas = torch.sqrt(widths * heights)
            centers_x = (boxes[:, 2] + boxes[:, 0]) / 2
            centers_y = (boxes[:, 3] + boxes[:, 1]) / 2

            # Scaled centers and dimensions
            centers_x_scaled = [centers_x * (grid / img_dim[1]) for grid in self.num_grids]
            centers_y_scaled = [centers_y * (grid / img_dim[0]) for grid in self.num_grids]
            widths_scaled = [widths * (grid / img_dim[1]) for grid in self.num_grids]
            heights_scaled = [heights * (grid / img_dim[0]) for grid in self.num_grids]

            levels = [torch.logical_and(self.scale_ranges[i][0] < areas, areas < self.scale_ranges[i][1]).int() for i in range(5)]

            img_category_targets, img_mask_targets, img_active_masks = [], [], []

            for level_id, level in enumerate(levels):
                height_by_stride, width_by_stride = img_dim[0] // self.strides[level_id], img_dim[1] // self.strides[level_id]
                scaled_mask = F.interpolate(mask.unsqueeze(0), size=(2*height_by_stride, 2*width_by_stride), mode='nearest').squeeze(0).to('cpu')

                cat_target = torch.zeros(self.num_grids[level_id], self.num_grids[level_id]).to('cpu')
                mask_target = torch.zeros(self.num_grids[level_id]**2, 2*height_by_stride, 2*width_by_stride).to('cpu')
                active_mask = torch.zeros(self.num_grids[level_id]**2).to('cpu')

                for label_id, is_active in enumerate(level):
                    if is_active.item() == 0:
                        continue

                    # Scale down the boxes and bound them
                    x_tl, y_tl = max(torch.round(centers_x_scaled[level_id][label_id] - 0.2 * widths_scaled[level_id][label_id] / 2), centers_x_scaled[level_id][label_id] - 1).int().item(), max(torch.round(centers_y_scaled[level_id][label_id] - 0.2 * heights_scaled[level_id][label_id] / 2), centers_y_scaled[level_id][label_id] - 1).int().item()
                    x_br, y_br = min(torch.round(centers_x_scaled[level_id][label_id] + 0.2 * widths_scaled[level_id][label_id] / 2), centers_x_scaled[level_id][label_id] + 1).int().item(), min(torch.round(centers_y_scaled[level_id][label_id] + 0.2 * heights_scaled[level_id][label_id] / 2), centers_y_scaled[level_id][label_id] + 1).int().item()

                    mesh_x, mesh_y = torch.meshgrid(torch.arange(y_tl, y_br+1, device='cpu'), torch.arange(x_tl, x_br+1, device='cpu'))

                    cat_target[mesh_x, mesh_y] = lbls[label_id].item()
                    mask_target[mesh_x.flatten()*self.num_grids[level_id] + mesh_y.flatten()] = scaled_mask[label_id]
                    active_mask[mesh_x.flatten()*self.num_grids[level_id] + mesh_y.flatten()] = 1.

                img_category_targets.append(cat_target)
                img_mask_targets.append(mask_target)
                img_active_masks.append(active_mask)

            category_targets.append(img_category_targets)
            mask_targets.append(img_mask_targets)
            active_masks.append(img_active_masks)

        return category_targets, mask_targets, active_masks

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
    
    def reshape_tensors(self, pred_cats, tgt_cats, pred_masks, tgt_masks, act_masks):
        """
        Reshape and flatten tensors for consistent processing.

        Args:
            pred_cats (list): List of predicted categories.
            tgt_cats (list): List of target categories.
            pred_masks (list): List of predicted masks.
            tgt_masks (list): List of target masks.
            act_masks (list): List of active masks indicating the presence of objects.

        Returns:
            tuple: Reshaped and flattened tensors for categories, masks, and active masks.
        """
        # Reshape predicted categories
        reshaped_cats = torch.cat([p.permute(0, 2, 3, 1).reshape(-1, 3) for p in pred_cats]).flatten()

        # Reshape target categories
        tgt_cat_flattened = torch.cat([torch.cat(tc).flatten() for tc in zip(*tgt_cats)])
        tgt_cat_reshaped = F.one_hot(tgt_cat_flattened.type(torch.int64), num_classes=4)[:, 1:].flatten()

        # Flatten masks
        mask_preds = [p.flatten(start_dim=0, end_dim=1) for p in pred_masks]
        mask_targets = [torch.cat(tm) for tm in zip(*tgt_masks)]
        active_mask_list = [torch.cat(am) for am in zip(*act_masks)]

        # Cleanup to free up memory
        del pred_cats, tgt_cats, pred_masks, tgt_masks, act_masks
        gc.collect()

        return reshaped_cats, tgt_cat_reshaped, mask_preds, mask_targets, active_mask_list

    def get_consistent_shape(self, category_predictions, category_targets, mask_predictions, mask_targets, active_masks):
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

    def solo_loss(self, category_predictions, category_ground_truths, mask_predictions, mask_ground_truths, active_masks):
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
        reshaped_category_preds, reshaped_category_targets, reshaped_mask_preds, reshaped_mask_targets, reshaped_active_masks = self.get_consistent_shape(
            category_predictions, category_ground_truths, mask_predictions, mask_ground_truths, active_masks)
        
        # Retrieve category loss configuration
        category_loss_config = self._default_cfg["cate_loss_cfg"]

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
        total_loss = category_loss_config["weight"] * focal_loss + self._default_cfg["mask_loss_cfg"]["weight"] * total_mask_loss

        return total_loss, focal_loss, total_mask_loss

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
        total_loss, category_loss, mask_loss = self.solo_loss(category_predictions, category_targets, mask_predictions, mask_targets, active_targets)
        
        # Log the losses for monitoring
        self.log("training_dice_loss", mask_loss, on_epoch=True)
        self.log("training_category_loss", category_loss, on_epoch=True)
        self.log("training_total_loss", total_loss, on_epoch=True, prog_bar=True)
        
        # Store the losses for further analysis
        self.training_step_outputs.append({'loss': total_loss, "dice_loss": mask_loss, "cate_loss": category_loss})
        
        return {'loss': total_loss, "dice_loss": mask_loss, "cate_loss": category_loss}

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.
        
        Args:
            batch (tuple): Current batch of validation data.
            batch_idx (int): Index of the current batch.
            
        Returns:
            dict: Dictionary containing the validation loss and other metrics.
        """
        val_idx, val_images, val_labels, val_masks, val_bboxes = batch
        
        # Generate target categories, masks, and active masks for the current batch
        category_targets, mask_targets, active_targets = self.generate_targets(val_bboxes, val_labels, val_masks)
        
        # Forward pass: Get model predictions for categories and masks
        category_predictions, mask_predictions = self(val_images)
        
        # Compute the solo loss (combined category and mask loss) for validation data
        total_val_loss, category_val_loss, mask_val_loss = self.solo_loss(category_predictions, category_targets, mask_predictions, mask_targets, active_targets)
        
        # Log the losses for monitoring
        self.log("validation_total_loss", total_val_loss, on_epoch=True, prog_bar=True)
        self.log("validation_category_loss", category_val_loss, on_epoch=True)
        self.log("validation_dice_loss", mask_val_loss, on_epoch=True)
        
        # Store the losses for further analysis
        self.validation_step_outputs.append({'val_loss': total_val_loss, 'val_cate_loss': category_val_loss, 'val_dice_loss': mask_val_loss})
        
        return {'val_loss': total_val_loss, 'val_cate_loss': category_val_loss, 'val_dice_loss': mask_val_loss}

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
    
    def on_validation_epoch_end(self, outputs):
        """
        Callback function executed at the end of each validation epoch.
        Computes and logs average validation losses for the epoch.
        
        Args:
            outputs (list): List of dictionaries containing validation step outputs.
        """
        average_validation_loss = self.compute_epoch_loss(outputs)
        average_dice_validation_loss = self.compute_epoch_loss([x['val_dice_loss'] for x in outputs])
        average_category_validation_loss = self.compute_epoch_loss([x['val_cate_loss'] for x in outputs])

        self.val_loss.append(average_validation_loss)
        self.val_mask.append(average_dice_validation_loss)
        self.val_cat.append(average_category_validation_loss)

        print('Average Validation Loss:', self.val_loss)
        print('Average Validation Mask Loss:', self.val_mask)
        print('Average Validation Categorical Loss:', self.val_cat)

        self.log("val_loss", average_validation_loss, on_epoch=True, prog_bar=True)
        self.log("val_dice_loss", average_dice_validation_loss, on_epoch=True)
        self.log("val_category_loss", average_category_validation_loss, on_epoch=True)
    
    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler for training.
        
        Returns:
            dict: Dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[27, 33], gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
