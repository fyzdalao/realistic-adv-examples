import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.model_wrappers.general_model import MeanStdType, ModelWrapper


class TorchModelWrapper(ModelWrapper):

    def __init__(self,
                 model: nn.Module,
                 n_class: int = 10,
                 im_mean: MeanStdType = None,
                 im_std: MeanStdType = None,
                 take_sigmoid: bool = True):
        super().__init__(n_class, im_mean, im_std, take_sigmoid)
        self._model = model

    def make_model_eval(self):
        self._model.eval()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        if len(image.size()) != 4:
            image = image.unsqueeze(0)
        image = self.preprocess(image)
        logits = self._model(image)
        return logits

    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        processed = image
        if self.im_mean is not None and self.im_std is not None:
            processed = (image - self.im_mean) / self.im_std
        return processed


    def _predict_prob_ori(self, image: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        with torch.no_grad():
            if len(image.size()) != 4:
                image = image.unsqueeze(0)
            image = self.preprocess(image)
            logits = self._model(image)
            self.num_queries += image.size(0)
        return logits


    '''
    trying to add pawn sacrifice defense
    '''
    def _predict_prob(self, image: torch.Tensor, verbose: bool = False) -> torch.Tensor:

        rnd_nu = 0.0

        with torch.no_grad():
            if len(image.size()) != 4:
                image = image.unsqueeze(0)

            # 在preprocess之前进行添加噪音
            # 此时image的值在0-1之间

            noise_in = torch.normal(mean=0, std=1, size=image.size(), device=image.device) * rnd_nu
            image = (image + noise_in).clip(0, 1)

            image = self.preprocess(image)
            logits = self._model(image)
            self.num_queries += image.size(0)

            # do something to logits.
            prob_ori = F.softmax(logits, dim=1)
            value, index_ori = torch.topk(prob_ori, k=2, dim=1)
            margin_ori = value[:, 0] - value[:, 1]

            #pawn_thres = 0.05
            #pawn_thres = 0

            pawn_thres = 0.1

            step = 0.0025

            decision = margin_ori < 0  # All False
            temp = 0
            while temp < pawn_thres:
                temp += step
                decision |= ((margin_ori > temp - step/2) & (margin_ori < temp))
            idx_to_swap = decision


            temp = logits[idx_to_swap, index_ori[idx_to_swap, 0]]
            logits[idx_to_swap, index_ori[idx_to_swap, 0]] = logits[idx_to_swap, index_ori[idx_to_swap, 1]]
            logits[idx_to_swap, index_ori[idx_to_swap, 1]] = temp

        return logits
