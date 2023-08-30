import opuslib
import torch
import numpy as np


class OpusCodec():
    """
    Runs Opus compression with the same parameters used on each of the robots
    """
    def __init__(self, channels, sr, frame_width=0.02) -> None:
        self.channels = channels
        
        # Initialize encoder
        self.encoder = opuslib.api.encoder.create_state(sr, channels, opuslib.APPLICATION_RESTRICTED_LOWDELAY)
        
        # Parameters used on robots to do compression
        opuslib.api.encoder.encoder_ctl(
                    self.encoder, opuslib.api.ctl.set_bitrate, 32000)
        opuslib.api.encoder.encoder_ctl(
                    self.encoder, opuslib.api.ctl.set_complexity, 0)
        opuslib.api.encoder.encoder_ctl(
                    self.encoder, opuslib.api.ctl.set_inband_fec, 0)
        opuslib.api.encoder.encoder_ctl(
                    self.encoder, opuslib.api.ctl.set_packet_loss_perc, 0)
        opuslib.api.encoder.encoder_ctl(
                    self.encoder, opuslib.api.ctl.set_dtx, 0)
        opuslib.api.encoder.encoder_ctl(
                    self.encoder, opuslib.api.ctl.set_lsb_depth, 16)
        
        # Create decoder
        self.decoder = opuslib.api.decoder.create_state(sr, channels)

        # Initialize frame size
        self.frame_size = int(round(sr * frame_width))

    def apply(self, audio: torch.FloatTensor) -> torch.Tensor:
        # Reset encoder state
        opuslib.api.encoder.encoder_ctl(self.encoder, opuslib.api.ctl.reset_state)
        
        # Reset decoder state
        opuslib.api.decoder.decoder_ctl(self.decoder, opuslib.api.ctl.reset_state)
        
        # Convert float tensor to int16 and then to bytes
        audio = (audio * (2 ** 15 - 1)).short().numpy().tobytes()

        # Go over and encoder byte array into frames and add encoded frames to list
        bchunks = []
        for i in range(0, len(audio), 2 * self.frame_size):
            encoded = opuslib.api.encoder.encode(self.encoder,
                                                 audio[i: i + 2 * self.frame_size],
                                                 self.frame_size,
                                                 2 * self.frame_size)
            bchunks.append(encoded)
        
        # Decode chunks into audio waveform as bytes
        output = b''
        for bchunk in bchunks:
            dec = opuslib.api.decoder.decode(self.decoder,
                                             bchunk,
                                             len(bchunk),
                                             1500,
                                             False,
                                             self.channels)
            output += dec

        # Convert bytes back to int16 and then back to float
        output = torch.from_numpy(np.frombuffer(output, dtype=np.int16) / (2 ** 15 - 1))
        return output