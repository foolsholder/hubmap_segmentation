import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from typing import Dict, List, Union, Tuple, Optional, Any

from .holder import ModelHolder


class TilingHolder(ModelHolder):
    def __init__(
            self,
            tiling: bool = False,
            tiling_h: int = 512,
            tiling_w: int = 512,
            *args,
            **kwargs
    ):
        super(TilingHolder, self).__init__(*args, **kwargs)
        self.tiling = tiling
        self.tiling_shape = (tiling_h, tiling_w)

    def _forward_impl(
            self,
            input_x: torch.Tensor,
            additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        preds = self.segmentor(input_x)
        preds.pop('logits')
        return preds

    def _forward(
            self,
            input_x: torch.Tensor,
            additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return self._forward_impl(input_x, additional_info)

    def tiling_forward(
            self,
            input_x: torch.Tensor,
            additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        # only test time
        if 'full_image' in additional_info:
            input_x = additional_info['full_image']
        (th, tw) = self.tiling_shape

        padding_params = (tw // 8, tw // 8, th // 8, th // 8)
        input_x = F.pad(input_x, padding_params, mode='constant', value=0)
        h, w = input_x.shape[2:]

        shift_h = th - th // 4
        shift_w = tw - tw // 4

        weight = torch.zeros((h, w)).to(input_x.device)
        probs = torch.zeros((h, w)).to(input_x.device)

        h_cnt = (h - 1) // shift_h + 1
        w_cnt = (w - 1) // shift_w + 1

        shift_h = (h - 1) // h_cnt + 1
        shift_w = (w - 1) // w_cnt + 1

        for h_idx in range(h_cnt):
            h_right = min(h, shift_h * h_idx + th)
            h_left = h_right - th
            for w_idx in range(w_cnt):
                w_right = min(w, shift_w * w_idx + tw)
                w_left = w_right - tw

                mask = torch.zeros((th, tw)).to(input_x.device)
                mask[th//8:-th//8, tw//8:-tw//8] = 1.

                weight[h_left:h_right, w_left:w_right] += mask
                input_window = input_x[:, :, h_left:h_right, w_left:w_right]
                preds = self._forward(input_window, additional_info)
                window_probs = preds['probs'][0, 0]
                probs[h_left:h_right, w_left:w_right] += window_probs * mask
        probs = probs[th//8:-th//8, tw//8:-tw//8]
        weight = weight[th//8:-th//8, tw//8:-tw//8]
        return {
            "probs": (probs / weight)[None, None]
        }

    def forward(
            self,
            input_x: torch.Tensor,
            additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.tiling:
            return self._forward(input_x, additional_info)
        return self.tiling_forward(input_x, additional_info)


class TTAHolder(TilingHolder):
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
        preds = self._forward_impl(batch_input, additional_info)

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

        preds['probs'] = torch.mean(preds['probs'], dim=0, keepdim=True)
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
            self.segmentor.load_state_dict(st)
            preds_tmp = super(EnsembleHolder, self)._forward_impl(input_x, additional_info=additional_info)
            tensor = preds_tmp['probs'] * w[idx]
            probs += [tensor[None]]
        probs = torch.cat(probs, dim=0)
        probs = torch.sum(probs, dim=0, keepdim=False) / w_sum
        preds = {
            'probs': probs
        }
        return preds
