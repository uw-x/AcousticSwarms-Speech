import torch
import torch.nn as nn

from sep.training.losses import CompositeLoss, SISDRLoss


class BaseNetwork(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device=torch.device('cpu')

    def set_loss(self, loss: str):
        # Choose loss
        if loss == 'l1':
            print("Using L1 loss")
            self.loss_fn = nn.L1Loss()
        elif loss == 'snr':
            print("Using snr loss")
            self.loss_fn = CompositeLoss()
        elif loss == 'snr_w_scaled_neg':
            print("Using fused loss")
            self.loss_fn = CompositeLoss(r=0, n=500)
        elif loss == 'fused':
            print("Using fused loss")
            self.loss_fn = CompositeLoss(r=0.05)
        elif loss == 'sisdr':
            print("Using sisdr loss")
            self.loss_fn = SISDRLoss()
        else:
            assert 0, 'Loss must be either \'l1\' or \'fused\'.'

    def loss(self, est_signal: torch.Tensor, gt_voice_signal: torch.Tensor):
        """
        Input: (B, C, T)
        """
        return self.loss_fn(est_signal, gt_voice_signal)

    def print_model_info(self):
        total_num = sum(p.numel() for p in self.parameters())
        print("Model has {:.02f}M parameters.".format(total_num/1e6))

    def to(self, device=None, *args, **kwargs):
        if device is not None:
            self.device = device
        return super().to(device, *args, **kwargs)
