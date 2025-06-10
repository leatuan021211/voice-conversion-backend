import base64
from io import BytesIO
from typing import BinaryIO

import numpy as np
import torch
import librosa
import soundfile as sf
import struct
import webrtcvad
from scipy.ndimage import binary_dilation


from core.settings import AIModelSettings, AudioSettings
from core.constants import DENOISER, VOCODER

class AudioService:

    @staticmethod
    def load_audio(audio: BinaryIO) -> np.ndarray:
        waveform, _ = librosa.load(audio, sr=AudioSettings.SAMPLE_RATE, mono=True)
        return waveform
    
    @staticmethod
    def trim_long_silences(wav: np.ndarray) -> np.ndarray:
        """
        Trims long silences from the audio using WebRTC VAD.
        Args:
            wav (np.ndarray): Input audio waveform.
        Returns:
            np.ndarray: Trimmed audio waveform.
        """
        samples_per_window = (AudioSettings.VAD_WINDOW_LENGTH * AudioSettings.SAMPLE_RATE) // 1000
        
        wav = wav[:len(wav) - (len(wav) % samples_per_window)]
        
        int16_max = (2 ** 15) - 1
        pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
        
        voice_flags = []
        vad = webrtcvad.Vad(mode=3)
        for window_start in range(0, len(wav), samples_per_window):
            window_end = window_start + samples_per_window
            voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                            sample_rate=AudioSettings.SAMPLE_RATE))
        voice_flags = np.array(voice_flags)

        def moving_average(array, width):
            array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
            ret = np.cumsum(array_padded, dtype=float)
            ret[width:] = ret[width:] - ret[:-width]
            return ret[width - 1:] / width
        
        audio_mask = moving_average(voice_flags, AudioSettings.VAD_MOVING_AVERAGE_WIDTH)
        audio_mask = np.round(audio_mask).astype(np.bool)
        
        audio_mask = binary_dilation(audio_mask, np.ones(AudioSettings.VAD_MAX_SILENCE_LENGTH + 1))
        audio_mask = np.repeat(audio_mask, samples_per_window)
        
        return wav[audio_mask == True]

    @staticmethod
    def convert_waveform_to_base64(waveform: torch.Tensor) -> str:
        """
        Converts a torch waveform tensor to a base64 string.
        Args:
            waveform (torch.Tensor): Input audio waveform.
        Returns:
            str: Base64 encoded string of the waveform.
        """
        if waveform.is_cuda:
            waveform = waveform.cpu()

        if len(waveform.shape) > 1:
            waveform = waveform.squeeze()

        waveform_np = waveform.detach().numpy()
        buffer = BytesIO()
        sf.write(buffer, waveform_np, 16000, format='WAV')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    @staticmethod
    def generate_waveform_from_mel_db(mel_spec: torch.Tensor) -> torch.Tensor:
        mel_spec = torch.exp(mel_spec)
        audio = VOCODER(mel_spec.to(AIModelSettings.DEVICE)).squeeze(0)
        for _ in range(100):
            audio = DENOISER(audio)
        return audio.detach().cpu()