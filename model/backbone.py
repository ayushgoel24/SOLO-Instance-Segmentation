import torch.nn as nn
import torchvision.models.detection as tvd

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.pretrained_model = tvd.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True)
        self.backbone = self.pretrained_model.backbone

    def forward(self, images):
        return [v.detach() for v in self.backbone(images).values()]