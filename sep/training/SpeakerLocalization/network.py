import torch
import torch.nn as nn
import torch.nn.functional as F

import sep.helpers.utils as utils

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

def normalize_input(data):
    """
    Normalizes the input to have mean 0 std 1 for each input
    Inputs:
        data - torch.tensor of size batch x n_mics x n_samples
    """
    data = (data * 2**15).round() / 2**15
    ref = data.mean(1)  # Average across the n microphones
    means = ref.mean(1).unsqueeze(1).unsqueeze(2)
    stds = ref.std(1).unsqueeze(1).unsqueeze(2)
    data = (data - means) / stds

    return data, means, stds#, ref

def unnormalize_input(data, means, stds):
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

        self.embed1 = nn.Conv1d(2, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, window_embedding: torch.Tensor):
        x = self.res(x)
        
        x = self.embed1(window_embedding.unsqueeze(2)) * x
        
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

    def forward(self, mix: torch.Tensor, window_embedding:torch.Tensor):
        x = mix
        skip_connections = [x]

        # Encoder
        for i, block in enumerate(self.module_list):
            x = block(x, window_embedding)

            skip_connections.append(x)
        
        return x, skip_connections

class UpsamplerBlock(nn.Module):
    def __init__(self, in_channels, out_channels,stride):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=stride, stride=stride)

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

        self.embed1 = nn.Conv1d(2, 2 * out_channels, kernel_size=1)

    def forward(self, x, skip, window_embedding):
        x = x + skip
        
        x = self.upsample(x)
        
        x = self.embed1(window_embedding.unsqueeze(2)) * x

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

    def forward(self, encoded: torch.Tensor, skip_connections: list, window_embedding:torch.Tensor):
        x = encoded

        # Source decoder
        for i, block in enumerate(self.module_list):
            skip = skip_connections[-(i+1)]
            
            x = block(x, skip, window_embedding)
        
        return x

class BottleNeck(nn.Module):
    def __init__(self,
                 channels: int,
                 nhead: int,
                 ffw_dim: int,
                 nlayers: int,
                 ) -> None:
        super().__init__()

        self.H = nhead
        self.C = channels
        self.L = nlayers
        self.D = ffw_dim
        
        transf_layer = nn.TransformerEncoderLayer(d_model=self.C, nhead=self.H, dim_feedforward=self.D)
        self.transf = nn.TransformerEncoder(transf_layer, num_layers=self.L)

    def forward(self, x: torch.Tensor):
        """
        x: (N, F, T)
        """
        x = x.permute(2, 0, 1) # (T, N, F)
        x = self.transf(x) 
        x = x.permute(1, 2, 0) # (N, F, T)

        return x


class Network(BaseNetwork):
    def __init__(
            self,
            device=None,
            n_mics: int=7,
            kernel_size: int = 7,
            stride_list: list = [2, 2, 4, 4, 4],
            channels: int = 64,
            growth: float = 2,
            encoder_channels: int = 2048,
            encoder_kernel_size: int = 33,
            encoder_stride: int = 16,
            rescale: float = 0.1,
            residual_layers: int = 3,
            residual_dilation_factor: int = 7,
            num_head: int = 8,
            ffw_dim: int = 1024,
            num_transformer_layers: int = 2):  # pylint: disable=redefined-outer-name
        
        super().__init__()
        self.n_mics = n_mics
        self.n_audio_channels = n_mics #* n_mics
        self.kernel_size = kernel_size
        self.stride = stride_list
        self.depth = len(stride_list)
        self.channels = channels
        self.growth = growth
        self.rescale = rescale
        self.output_kernel_size = 1
        self.residual_layers = residual_layers
        self.residual_dilation_factor = residual_dilation_factor
        self.num_head = num_head
        self.ffw_dim = ffw_dim
        self.num_transformer_layers = num_transformer_layers

        in_channels = channels #self.n_audio_channels
        
        self.preproc = nn.Conv1d(in_channels = self.n_audio_channels, 
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


        # TF encoder bottleneck layer
        channels = self.encoder.out_channels
        self.bottleneck = BottleNeck(channels = channels, nhead = num_head, ffw_dim= ffw_dim, nlayers=num_transformer_layers)

        # Compute product of strides
        self.stride_product = 1
        for s in self.stride:
            self.stride_product *= s

        rescale_module(self, reference=rescale)

        # TODO: Remove
        if device is not None:
            self.to(device)
            self.device = device

    def forward(self, mix: torch.Tensor, window_embedding: torch.Tensor):
        """
        Forward pass.

        Args:
            mix (torch.Tensor) - Aligned input recordings (B, M, t)
            window_embedding (torch.Tensor) - Patch size (B, 2)

        Output:
            x - Single channel source separation output (B, 1, t)
        """
        input_length = mix.shape[-1]
        
        # Pad zeros to the left until length is a multiple of stride product
        T = ((mix.shape[-1] - 1)//self.stride_product + 1) * self.stride_product
        mix = F.pad(mix, (T - input_length, 0))

        # Save reference mic channel
        ref = mix[:, 0].unsqueeze(1)
        x = mix

        # Convert M channels to initial input channel number C
        x = self.preproc(x)

        # Pass (B, C, T) to encoder
        x, skip_connections = self.encoder(x, window_embedding)

        # Transformer layers
        x = self.bottleneck(x)

        # Pass (B, C, T) to decoder
        x = self.decoder(x, skip_connections, window_embedding)
        
        # Generate a latent space representation for the reference channel input
        y = self.reference_bypass_relu(self.reference_bypass(ref))
        
        # Generate mask from U-Net output
        mask = self.mask_encoder_relu(self.mask_encoder(x))
        
        # Apply mask & decode
        x = self.output_decoder(y * mask)
        x = x[...,9:-8] # Trim extra samples from convtr (Note: This should be changed if you want to use different kernel sizes)

        return x[..., -input_length:]