from io import BytesIO
import torch
import numpy as np

from core.settings import AIModelSettings
from core.constants import VOICE_CONVERSION, VOICES
from services.audio_service import AudioService

import matplotlib.pyplot as plt
import torch

def plot_mel_spectrogram(mel: torch.Tensor, title: str = "Mel Spectrogram", save_path: str = None):
    """
    Plot and optionally save a mel spectrogram image.

    Args:
        mel (Tensor): Mel spectrogram of shape (n_mels, time)
        title (str): Plot title
        save_path (str): If provided, saves the image to this path
    """
    if mel.dim() == 3:  # remove batch dimension if needed
        mel = mel.squeeze(0)
    if mel.dim() != 2:
        raise ValueError("Expected mel shape (n_mels, time)")

    plt.figure(figsize=(10, 4))
    plt.imshow(mel.detach().numpy(), aspect='auto', origin='lower', interpolation='none')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Mel bins")
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


class VoiceConversionService:
    
    @staticmethod
    def converse_voice(source: BytesIO, voice_name: str) -> list:
        waveform = AudioService.load_audio(source)
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform.to(AIModelSettings.DEVICE)
        
        voice_feature_vector = VoiceConversionService.get_voice_feature_vector(voice_name)
        voice_feature_vector = voice_feature_vector.to(AIModelSettings.DEVICE)
        mel, _ = VOICE_CONVERSION(waveform, voice_feature_vector)
        audio = AudioService.generate_waveform_from_mel_db(mel.squeeze(0).T)
        audio = AudioService.convert_waveform_to_base64(audio)
        return audio

    @staticmethod
    def get_available_voices() -> list:
        """
        Returns a list of available voices.
        """
        return list(VOICES.keys())
    
    @staticmethod
    def get_voice_feature_vector(voice_name: str) -> torch.Tensor:
        """
        Returns the feature vector for a given voice.
        """
        if voice_name not in VOICES:
            raise ValueError(f"Voice '{voice_name}' not found.")
        
        voice_path = VOICES[voice_name]
        feature_vector = np.load(voice_path)
        feature_vector = torch.from_numpy(feature_vector).unsqueeze(0)
        feature_vector = feature_vector.to(AIModelSettings.DEVICE)
        
        return feature_vector
