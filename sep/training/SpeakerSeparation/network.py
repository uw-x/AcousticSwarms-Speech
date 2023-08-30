import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sep.helpers.utils as utils

from speechbrain.lobes.models.transformer.Conformer import ConformerEncoder
from speechbrain.nnet.attention import RelPosEncXL

from sep.training.base_network import BaseNetwork


def rescale_conv(conv, reference):
    """
    Rescale a convolutional module with `reference`.
    """
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale

def rescale_module(module, reference):
    """
    Rescale a module with `reference`.
    """
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)

def normalize_input(data: torch.Tensor):
    """
    Normalizes the input to have mean 0 std 1 for each input
    Inputs:
        data - torch.tensor of size batch x n_mics x n_samples
    """
    data = (data * 2**15).round() / 2**15
    ref = data.mean(1)  # Average across the channels
    means = ref.mean(1).unsqueeze(1).unsqueeze(2)
    stds = ref.std(1).unsqueeze(1).unsqueeze(2)
    data = (data - means) / stds

    return data, means, stds

def unnormalize_input(data: torch.Tensor, means, stds):
    """
    Unnormalizes the step done in the previous function
    """
    data = (data * stds + means)
    return data

class DilatedResidualLayer(nn.Module):
    def __init__(self, nchannels: int, ksize: int, dilation: int = 1):
        super().__init__()
        self.I = nchannels
        self.O = nchannels
        self.D = dilation
        self.K = ksize

        self.conv = nn.Conv1d(self.I, self.O, kernel_size=self.K, dilation=self.D, padding=(self.D * (self.K - 1) + 1) // 2)
        self.norm = nn.LayerNorm(nchannels)
        self.act = nn.ReLU()
    
    def forward(self, x: torch.Tensor):
        y = x
        x = self.conv(x)
        x = self.act(x) + y
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return x

class DilatedResidualSequence(nn.Module):
    def __init__(self, nchannels: int, ksize: int, nlayers: int = 2, dilation_factor: int = 2):
        super().__init__()
        self.I = nchannels
        self.O = nchannels
        self.D = dilation_factor
        self.N = nlayers

        layers = [DilatedResidualLayer(nchannels, ksize, dilation= dilation_factor ** i) for i in range(self.N)]
        self.seq = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.seq(x)

class EncoderBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            residual_layers: int,
            residual_dilation_factor: int
            ):
        super().__init__()

        self.res = DilatedResidualSequence(nchannels=in_channels, ksize=kernel_size, nlayers=residual_layers, dilation_factor=residual_dilation_factor)

        self.conv1 = nn.Conv1d(in_channels, 2 * out_channels, kernel_size, stride, padding=kernel_size//2)
        self.norm1 = nn.GroupNorm(2, 2 * out_channels)
        self.act1 = nn.GLU(dim=1)     

    def forward(self, x: torch.Tensor):        
        x = self.res(x)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        return x

class Encoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            kernel_size: int,
            depth: int,
            stride: list,
            channels: int,
            growth: float,
            residual_layers: int,
            residual_dilation_factor: int
            ):
        super().__init__()
        
        self.module_list = nn.ModuleList()
        
        for i in range(depth):
            block = EncoderBlock(in_channels=in_channels, 
                                 out_channels=channels,
                                 kernel_size=kernel_size,
                                 stride=stride[i],
                                 residual_layers=residual_layers,
                                 residual_dilation_factor=residual_dilation_factor)

            self.module_list.append(block)

            in_channels = channels
            channels = int(growth * channels)

        self.out_channels = in_channels

    def forward(self, mix: torch.Tensor):
        x = mix
        skip_connections = [x]

        # Encoder
        for i, block in enumerate(self.module_list):
            x = block(x)

            skip_connections.append(x)
        
        return x, skip_connections

class UpsamplerBlock(nn.Module):
    def __init__(self, in_channels, out_channels,stride):
        super().__init__()
        self.I = in_channels
        self.O = out_channels
        self.K = stride
        self.S = stride

        self.conv = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=self.K, stride=stride)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels: int, 
            out_channels: int,
            stride: int,
            kernel_size: int,
            residual_layers: int,
            residual_dilation_factor: int
            ):
        
        super().__init__()

        self.upsample = UpsamplerBlock(in_channels, 2 * out_channels, stride)
        self.norm1 = nn.GroupNorm(2, num_channels=2 * out_channels)
        self.act1 = nn.GLU(dim=1)

        self.res = DilatedResidualSequence(nchannels=out_channels, ksize=kernel_size, nlayers=residual_layers, dilation_factor=residual_dilation_factor)

    def forward(self, x, skip):
        x = x + skip
        
        x = self.upsample(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.res(x)

        return x

class Decoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            depth: int,
            channels: int,
            kernel_size: int,
            stride_list: list,
            growth: float,
            residual_layers: int,
            residual_dilation_factor: int,
            ):
        super().__init__()

        self.module_list = nn.ModuleList()
        self.out_channels = in_channels

        for i in range(depth):
            out_channels = in_channels

            block = DecoderBlock(channels, out_channels, stride_list[i], kernel_size, residual_layers, residual_dilation_factor)
            
            self.module_list.insert(0, block)  # Put it at the front, reverse order

            in_channels = channels
            channels = int(growth * channels)

    def forward(self, encoded: torch.Tensor, skip_connections: list):
        x = encoded

        for i, block in enumerate(self.module_list):
            skip = skip_connections.pop(-1)
            x = block(x, skip)
            
        return x

def speakers_to_batches(X: torch.tensor, S: int):
    """
    Reshapes speakers to batch dimension for independant processing
    X: (B, S, C, T)
    S: (B, 1)
    """
    B = X.shape[0]

    X = [X[b, :S[b]] for b in range(B)]
    X = torch.cat(X, 0)

    return X

def batches_to_speakers(X: torch.tensor, S: int):
    """
    Reshapes speakers from batch dimension to speaker dimension
    X: (BS, C, T)
    S: (B, 1)
    """
    B = S.shape[0]

    m = torch.max(S) # Get largest number of speakers

    out = []
    index = 0
    for b in range(B):
        if S[b] < m:
            # Pad m - S[b] zeros in the channel dimension
            out.append(F.pad(X[index: index + S[b]].transpose(0, -1), (0, m-S[b])).transpose(0, -1).unsqueeze(0))
        else:
            out.append(X[index: index + S[b]].unsqueeze(0))
        index += S[b]
    return torch.cat(out, 0)

class BottleNeck(nn.Module):
    def __init__(self,
                 channels: int,
                 nlayers: int,
                 ffw_dim: int,
                 num_head: int,
                 ksize: int,
                 ) -> None:
        super().__init__()
        self.C = channels
        self.L = nlayers
        self.F = ffw_dim
        self.H = num_head
        self.K = ksize

        self.pe_single = RelPosEncXL(self.C)

        self.module_list = nn.ModuleList()
        for i in range(self.L):
            layer = nn.ModuleDict()
            layer['intra'] = ConformerEncoder(num_layers=1, d_model = self.C, d_ffn = self.F, nhead = self.H, kernel_size=self.K)
            tf_encoder = nn.TransformerEncoderLayer(d_model=self.C, nhead=self.H, dim_feedforward=self.F, batch_first=True)
            layer['inter'] = nn.TransformerEncoder(encoder_layer=tf_encoder, num_layers=1)
            self.module_list.append(layer)

    def forward(self, x: torch.Tensor, num_speakers: torch.Tensor):
        """
        x: (N, S, F, T)
        """
        N, S, F, T = x.shape

        for layer in self.module_list:
            # Apply self-attention
            x = speakers_to_batches(x, num_speakers)

            x = x.transpose(1, 2)
            pe = self.pe_single(x)
            x = layer['intra'](x, pos_embs=pe)[0].transpose(1, 2)
            
            x = batches_to_speakers(x, num_speakers)

            # Apply cross-attention
            x = x.permute(0, 3, 2, 1) # (N, T, F, S)
            x = x.reshape(N * T, F, S) # (NT, F, S)
            x = x.transpose(1, 2) # (NT, S, F)
            
            x = layer['inter'](x).transpose(1, 2) # (NT, F, S)
        
            x = x.reshape(N, T, F, S)
            x = x.permute(0, 3, 2, 1) # (N, S, F, T)

        return x # (N, S, F, T)

class Network(BaseNetwork):
    def __init__(
            self,
            device = None,
            n_mics: int=7,
            max_speakers: int = 6,
            kernel_size: int = 5,
            stride_list: list = [2, 2, 4, 4],
            channels: int = 64,
            growth: float = 2,
            encoder_channels: int = 4096,
            encoder_kernel_size: int = 33,
            encoder_stride: int = 16,
            residual_layers: int = 3,
            residual_dilation_factor: int = 2,
            num_head: int = 8,
            ffw_dim: int = 1024,
            bottleneck_layers: int = 3,
            bottleneck_ksize: int = 31,
            rescale: float = 0.1):
        super().__init__()
        self.n_mics = n_mics
        self.max_n_speaker = max_speakers # Only used during training
        self.kernel_size = kernel_size
        self.stride = stride_list
        self.depth = len(stride_list)
        self.channels = channels
        self.growth = growth
        self.rescale = rescale
        self.residual_layers = residual_layers
        self.residual_dilation_factor = residual_dilation_factor
        self.num_head = num_head
        self.ffw_dim = ffw_dim
        self.num_transformer_layers = bottleneck_layers

        in_channels = channels
        
        self.preproc = nn.Conv1d(in_channels=self.n_mics,
                                 out_channels=channels,
                                 kernel_size=1)

        self.encoder = Encoder( in_channels=in_channels,
                                kernel_size=self.kernel_size,
                                depth=self.depth, 
                                stride=self.stride,
                                channels=self.channels,
                                growth=self.growth,
                                residual_layers = self.residual_layers,
                                residual_dilation_factor = self.residual_dilation_factor)

        channels = self.encoder.out_channels
        self.bottleneck = BottleNeck(channels = channels, 
                                     nlayers=bottleneck_layers,
                                     ffw_dim=ffw_dim,
                                     num_head=num_head,
                                     ksize=bottleneck_ksize)

        self.decoder = Decoder(in_channels=in_channels,
                                    depth=self.depth,
                                    channels=self.channels,
                                    kernel_size=self.kernel_size,
                                    stride_list=stride_list,
                                    growth=self.growth,
                                    residual_layers = self.residual_layers,
                                    residual_dilation_factor = self.residual_dilation_factor)

        self.reference_bypass = nn.Conv1d(in_channels=1, 
                                          out_channels=encoder_channels,
                                          kernel_size=encoder_kernel_size,
                                          stride=encoder_stride,
                                          padding=encoder_kernel_size//2)
        self.reference_bypass_relu = nn.ReLU()
        
        self.mask_encoder = nn.Conv1d(in_channels=self.channels,
                                      out_channels=encoder_channels,
                                      kernel_size=encoder_kernel_size,
                                      stride=encoder_stride,
                                      padding=encoder_kernel_size//2)
        self.mask_encoder_relu = nn.ReLU()
        
        self.output_decoder = nn.ConvTranspose1d(in_channels=encoder_channels,
                                                 out_channels=1,
                                                 kernel_size=encoder_kernel_size,
                                                 stride=encoder_kernel_size//2)

        self.stride_product = 1
        for s in self.stride:
            self.stride_product *= s

        rescale_module(self, reference=rescale)

        if device is not None:
            self.to(device)
            self.device = device

    def forward(self, mix: torch.Tensor, num_speakers: torch.Tensor):
        """
        Forward pass.

        Args:
            mix: Aligned input recordings (B, S * M, t)
            num_speakers: Number of speakers S in the mixture

        Output:
            x - Separated output for each source (B, S, t)
        """
        input_length = mix.shape[-1]

        # Pad zeros to the left until length is a multiple of stride product
        T = ((mix.shape[-1] - 1)//self.stride_product + 1) * self.stride_product
        mix = F.pad(mix, (T - input_length, 0))

        # Save reference mic channel
        ref = mix[:, 0].unsqueeze(1)
        x = mix
        
        B, SM, T = x.shape

        # (B * S, M, T)
        x = x.reshape(B, -1, self.n_mics, T)
        x = speakers_to_batches(x, num_speakers)

        # (B * S, C, T)
        x = self.preproc(x)

        # [ENCODER]
        # Pass (B * S, C, T) to encoder
        x, skip_connections = self.encoder(x)

        # (B * S, C, T) -> (B, S, C, T)
        x = batches_to_speakers(x, num_speakers)

        # [BOTTLENECK]

        # Intra- and inter-channel attention layers
        x = self.bottleneck(x, num_speakers) # (N, S, F, T)

        B, S, C, T = x.shape
        x = speakers_to_batches(x, num_speakers) # Pass (B, C, T) to decoder
        
        # [DECODER]
        x = self.decoder(x, skip_connections)

        # [MASKING]
        
        # Generate a latent space representation for the reference channel input
        y = self.reference_bypass_relu(self.reference_bypass(ref)).unsqueeze(3) # (N, C, T, 1)
        
        # Generate mask from U-Net output
        mask = self.mask_encoder_relu(self.mask_encoder(x)) # (N * S, F, T)
        BS, C, T = mask.shape
        
        mask = batches_to_speakers(mask, num_speakers) # (N, C, T, S)
        mask = mask.permute(0, 2, 3, 1) # (N, C, T, S)

        # Apply mask 
        y = y * mask # (N, F, T, S)
        y = y.permute(0, 3, 1, 2)
        B, S, C, T = y.shape
        y = y.reshape(B * S, C, T) # (N, F, T)
        
        x = self.output_decoder(y).reshape(B, S, -1) # (N, S, T) 
        x = x[...,9:-8] # Trim extra samples from convtr (Note: This should be changed if you want to use different kernel sizes)

        if x.shape[1] < self.max_n_speaker:
            x = F.pad(x, (0, 0, 0, self.max_n_speaker - x.shape[1]))

        return x[..., -input_length:]
            
    def infer(self, input_channels, patch_list) -> np.ndarray:
        sample_list = [p.sample_offset for p in patch_list]
        return self.infer_sample(input_channels, sample_list)

    def infer_sample(self, input_channels, sample_list) -> np.ndarray:
        """
        input_channels: (M x T)
        sample_list: (S x (M-1))
        """
        self.eval()
        with torch.no_grad():
            mix = input_channels

            data = []
            
            # For each speaker sample shift in the list of samples
            for sample_shifts in sample_list:
                # Round samples shifts to nearest integer
                sample_shifts_rounded = np.round(sample_shifts).astype(int)
                
                # Shift channels to each speaker and combine to form (M x T) mixture tensor
                mixture = [mix[0]]
                for c in range(1, self.n_mics):
                        shift_samples = -sample_shifts_rounded[c-1]
                        channel = torch.roll(mix[c], shift_samples)
                        
                        if shift_samples > 0:
                            channel[:shift_samples] = 0
                        elif shift_samples < 0:
                            channel[shift_samples:] = 0
                        mixture.append(channel)
                
                shifted_mixture = torch.stack(mixture, 0)
                data.append(shifted_mixture)
            
            # Concatenate all shifted samples to (S * M x T) array
            data = torch.vstack(data).float().to(self.device)
            
            # Add batch dimension
            data = data.unsqueeze(0)
            
            # Normalize input
            data, means, stds = normalize_input(data)

            # Initialize speaker number as tensor
            sample_tensor = torch.tensor([[len(sample_list)]], device=self.device)

            # Run through the model
            output_signal = self(data, sample_tensor)
            
            # De-normalize
            output_signal = unnormalize_input(output_signal, means, stds)

            # Return numpy array
            results = output_signal[0].cpu().numpy()[:len(sample_list)]
            
            return results

    def loss(self, voice_signal, gt_voice_signal, n_speakers):
        """Simple L1 loss between voice and gt"""
        N, S, t = voice_signal.shape
        return self.loss_fn(voice_signal.reshape(N*S, 1, t), gt_voice_signal.reshape(N*S, 1, t))