import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from typing import Dict, List, Union, Tuple, Optional, Any

from .holder import ModelHolder


class TTAHolder(ModelHolder):
    def __init__(
            self,
            tta_list: List[Tuple[str, Any]],
            *args,
            **kwargs
    ):
        super(TTAHolder, self).__init__(*args, **kwargs)
        self._stages_names = ['valid']
        self.idx_tta = tta_list

    def _forward(
            self,
            input_x: torch.Tensor,
            additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        only 'probs' is matter
        """
        idx_tta = self.idx_tta
        batch_input = [input_x]
        # [1, 3, H, W]

        #preds['full_probs'] = F.interpolate(preds['probs'], size=self._shape, mode='bicubic')
        for type_aug, args_aug in idx_tta:
            input_y = input_x
            if type_aug == 'flip':
                input_y = torch.flip(input_y, dims=args_aug)
            elif type_aug == 'transpose':
                input_y = torch.transpose(input_y, dim0=2, dim1=3)
            elif type_aug == 'rotate90':
                input_y = torch.rot90(input_y, k=args_aug, dims=[2, 3])
            batch_input += [input_y]

        batch_input = torch.cat(batch_input, dim=0)
        # [T, 3, H, W]

        preds = self._forward_impl(batch_input, additional_info)
        # [T, 1, H, W]

        idx_preds = 1
        for type_aug, args_aug in idx_tta:
            x = preds['probs'][idx_preds:idx_preds+1].clone()
            if type_aug == 'flip':
                x = torch.flip(x, dims=args_aug)
            elif type_aug == 'transpose':
                x = torch.transpose(x, dim0=2, dim1=3)
            elif type_aug == 'rotate90':
                x = torch.rot90(x, k=4-args_aug, dims=[2, 3])
            #x = F.interpolate(x, self._shape, mode='bicubic')
            preds['probs'][idx_preds:idx_preds+1] = x
            idx_preds += 1
        #preds['logits'] = torch.mean(preds['logits'], dim=0, keepdim=True)
        #preds['probs'] = torch.sigmoid(preds['logits'])
        preds['probs'] = torch.mean(preds['probs'], dim=0, keepdim=True)
        # [1, 1, H, W]
        return preds


def _clear_segmentor_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        res = type(state_dict)()
        str_patt = 'segmentor.'
        for k, v in state_dict.items():
            new_k = k[len(str_patt):]
            res[new_k] = v
        return res


class EnsembleHolder(TTAHolder):
    def __init__(
            self,
            ckpt_path_list: List[str],
            weights: Optional["Matrix"] = None,
            *args,
            **kwargs
    ):
        super(EnsembleHolder, self).__init__(*args, **kwargs)
        self._stages_names = ['valid']
        self.state_dicts = []
        if weights is None:
            self._weights = None
        else:
            self._weights = np.array(weights, dtype=np.float32)
        for path in ckpt_path_list:
            state_dict = torch.load(path, map_location='cpu')
            state_dict = state_dict['state_dict']
            state_dict = _clear_segmentor_prefix(state_dict)
            self.state_dicts += [state_dict]

    def _forward_impl(
            self,
            input_x: torch.Tensor,
            additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        probs = []
        if additional_info is not None:
            # batch_size == 1 or same augmented organ
            organ_id = additional_info['organ_id'][0].item()
        else:
            organ_id = -1

        if self._weights is None:
            w = np.ones(len(self.state_dicts), dtype=np.float32) / len(self.state_dicts)
        else:
            w = self._weights[:, organ_id]
        w_sum = np.sum(w)

        for idx, st in enumerate(self.state_dicts):
            self.segmentor.load_state_dict(st, strict=False)
            # [T, 3, H, W]
            preds_tmp = super(EnsembleHolder, self)._forward_impl(input_x, additional_info=additional_info)
            tensor = preds_tmp['probs'] * w[idx]
            probs += [tensor[None]]
        probs = torch.cat(probs, dim=0)
        # [M; T; 1; H, W]
        probs = torch.sum(probs, dim=0, keepdim=False) / w_sum
        preds = {
            'probs': probs,
            #'probs': torch.sigmoid(logits)
        }
        return preds
