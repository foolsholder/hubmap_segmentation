from decoder_modules import *
from typing import List, Dict, Any
import torch.nn.functional as F

class SegFormer(nn.Module):
    def __init__(
        self,
        config: Dict[str, Any],
        in_channels: int,
        widths: List[int],
        depths: List[int],
        all_num_heads: List[int],
        patch_sizes: List[int],
        overlap_sizes: List[int],
        reduction_ratios: List[int],
        mlp_expansions: List[int],
        decoder_channels: int,
        scale_factors: List[int],
        num_classes: int,
        drop_prob: float = 0.0,
    ):

        super().__init__()
        self.encoder = SegFormerEncoder(
            in_channels,
            widths,
            depths,
            all_num_heads,
            patch_sizes,
            overlap_sizes,
            reduction_ratios,
            mlp_expansions,
            drop_prob,
        )
        self.decoder = SegFormerDecoder(decoder_channels, widths[::-1], scale_factors)
        self.head = SegFormerSegmentationHead(
            decoder_channels, num_classes, num_features=len(widths)
        )
        self.num_calsses = num_classes
        self.config = config

    def forward(self, x):
        features = self.encoder(x)
        features = self.decoder(features[::-1])
        logits = self.head(features)
        probs = None
        if self.num_classes == 1:
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=1)
        probs = F.interpolate(
            probs,
            size=logits.shape[2:]*4,
            mode='bilinear',
            align_corners=False
        )
        return dict(
            logits=logits,
            probs=probs
        )