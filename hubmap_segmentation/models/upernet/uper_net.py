import torch

from torch import nn
from torch.nn import functional as F

from typing import Dict, List, Optional, Any, Tuple

from ..unet.unet_segmentor import create_backbone
from .uper_head import UPerHead
from .aux_fcn_head import FCNHead


class UperNet(nn.Module):
    def __init__(
            self,
            backbone_cfg: Dict[str, Any],
            num_classes: int = 6,
    ):
        super(UperNet, self).__init__()
        self.backbone, _ = create_backbone(backbone_cfg)
        self.decode_head = UPerHead(num_classes=num_classes)
        self.auxiliary_head = FCNHead(num_classes=num_classes)

    def forward(
            self,
            input_x: torch.Tensor,
            additional_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        x = input_x

        bb_feats = self.backbone(input_x)

        logits = self.decode_head(bb_feats)
        logits = F.interpolate(logits, size=input_x.shape[2:], mode='bilinear')
        probs = torch.softmax(logits, dim=1)

        res = {
            "logits": logits,
            "probs": probs
        }
        if self.training:
            aux_logits = self.auxiliary_head(bb_feats)
            aux_logits = F.interpolate(
                aux_logits,
                size=input_x.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            aux_probs = torch.softmax(aux_logits, dim=1)
            res.update({
                "aux_logits": aux_logits,
                "aux_probs": aux_probs
            })
        return res
