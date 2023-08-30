"""
The main training script for training on synthetic data
"""

import argparse
import multiprocessing
import os, time, glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .dataset import Dataset
from .network import Network, normalize_input, unnormalize_input
from asteroid.metrics import get_metrics

from torch.optim.lr_scheduler import StepLR
import re
import sep.helpers.utils as utils


def compute_metrics(orig: torch.Tensor,
                    est: torch.Tensor,
                    gt: torch.Tensor,
                    sr: torch.Tensor):
    """
    input: (N, 1, t) (N, 1, t)
    """
    if gt.shape[1] != 1:
        N, C, t = gt.shape
        gt.reshape(N * C, 1, t)
        est.reshape(N * C, 1, t)

    gt = gt[:, 0].detach().cpu().numpy()
    est = est[:, 0].detach().cpu().numpy()
    orig = orig[:, 0].detach().cpu().numpy() # Take first channel of original input
    
    mask = (np.absolute(gt).max(axis=1) > 0)

    metrics = []

    # Only consider positive samples because of complications with computing SI-SNR
    # If there's at least one positive sample
    if np.sum(mask) > 0:
        gt = gt[mask]
        est = est[mask]
        orig = orig[mask]
    
        for i in range(gt.shape[0]):
            metrics_dict = get_metrics(orig[i], gt[i], est[i], sample_rate=sr, metrics_list=['si_sdr'])
            metrics.append(metrics_dict)

    return metrics

def test_epoch(model: nn.Module, device: torch.device,
               test_loader: torch.utils.data.dataloader.DataLoader,
               sr: int,
               log_interval: int = 20) -> float:
    """
    Evaluate the network.
    """
    model.eval()
    
    test_loss = 0
    metrics = []

    with torch.no_grad():
        for batch_idx, (data,
                        label_voice_signals,
                        n_speakers) in enumerate(test_loader):
            data = data.to(device)
            label_voice_signals = label_voice_signals.to(device)

            # Normalize input, each batch item separately
            normed_data, means, stds = normalize_input(data)
            
            output_signal = model(normed_data, n_speakers)
            
            # Un-normalize
            output_signal = unnormalize_input(output_signal, means, stds)
            output_voices = output_signal

            loss = model.module.loss(output_voices, label_voice_signals, n_speakers)
            test_loss += loss.item()

            # Compute metrics
            m = compute_metrics(data, output_signal, label_voice_signals, sr)
            metrics.extend(m)
            
            if batch_idx % log_interval == 0:
                print("Loss: {}".format(loss.item()))

        return test_loss / len(test_loader), metrics

import os
def train_epoch(model: nn.Module, device: torch.device,
                optimizer: optim.Optimizer,
                train_loader: torch.utils.data.dataloader.DataLoader,
                epoch: int,
                training_params: int,
                log_interval: int = 20) -> float:
    """
    Train a single epoch.
    """
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"    
    
    # Set the model to training.
    model.train()

    # Training loop
    train_loss = 0
    interval_losses = []
    
    t1 = time.time()
    for batch_idx, (data,
                    label_voice_signals,
                    n_speakers) in enumerate(train_loader):
        data = data.to(device)
        label_voice_signals = label_voice_signals.to(device)

        # Reset grad
        optimizer.zero_grad()
        
        # Normalize input, each batch item separately
        normed_data, means, stds = normalize_input(data)
        
        output_signal = model(normed_data, n_speakers)
        
        # Un-normalize
        output_signal = unnormalize_input(output_signal, means, stds)
        output_voices = output_signal
        
        loss = model.module.loss(output_voices, label_voice_signals, n_speakers)

        # Backpropagation
        loss.backward(retain_graph=False)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_params['gradient_clip']) 
        
        # Update the weights
        optimizer.step()

        # Save losses
        loss = loss.detach() 
        interval_losses.append(loss.item())
        train_loss += loss.item()

        # Print the loss
        if batch_idx % log_interval == 0:
            t2 = time.time()
            
            print("Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f} \t Time taken: {:.4f}s ({} examples)".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                np.mean(interval_losses),
                t2 - t1,
                log_interval * output_signal.shape[0] * (batch_idx > 0) + output_signal.shape[0] * (batch_idx == 0)))

            interval_losses = []
            t1 = time.time()

    return train_loss / len(train_loader)
