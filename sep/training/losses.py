import torch
import torch.nn as nn
from asteroid.losses.sdr import SingleSrcNegSDR


class CompositeLoss(nn.Module):
    def __init__(self, r = 0, n = 1) -> None:
        super().__init__()
        self.l1_loss = nn.L1Loss(reduce=False)
        self.snrloss = SingleSrcNegSDR('snr')
        self.r = r
        self.neg_scale = n
    
    def forward(self, output, gt):
        """
        input: (N, 1, t) (N, 1, t)
        """
        assert gt.shape[1] == 1
        assert output.shape[1] == 1

        # Get output and estimates
        gt = gt[:, 0]
        output = output[:, 0]
        
        # Get mask for samples that are supposed to be 0 (i.e. negative sample)
        # So that we don't use SNR loss on them
        mask = (torch.absolute(gt).max(dim=1)[0] == 0)
        
        # Compute l1 loss
        l1loss = self.l1_loss(output, gt)
        
        # When needed, compute SNR loss
        if self.r < 1:
            snrloss = self.snrloss(output, gt)
        
        comp_loss = 0
        
        # Apply only L1 loss on negative samples
        if any(mask):
            comp_loss += torch.mean(l1loss[mask]) * self.neg_scale
        
        # Apply fused loss on positive samples
        if any((~ mask)):
            comp_loss += torch.mean(l1loss[~ mask]) * self.r + torch.mean(snrloss[~ mask]) * (1 - self.r)

        return comp_loss

class SISDRLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sisdrloss = SingleSrcNegSDR('sisdr')
    
    def forward(self, output, gt):
        """
        input: (N, 1, t) (N, 1, t)
        """
        assert gt.shape[1] == 1
        assert output.shape[1] == 1

        # Get output and estimates
        gt = gt[:, 0]
        output = output[:, 0]

        mask = (torch.absolute(gt).max(dim=1)[0] == 0)
        
        return torch.mean(self.sisdrloss(output[~mask], gt[~mask]))
