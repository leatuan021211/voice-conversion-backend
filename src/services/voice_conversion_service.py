from typing import BinaryIO
import wave
import torch
import numpy as np

from core.settings import AIModelSettings, AudioSettings
from core.constants import VOICE_CONVERSION, VOICES
from services.audio_service import AudioService
from utils.normalize import denormalize_tensor

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
    def converse_voice(source: BinaryIO, voice_id: str) -> str:
        waveform = AudioService.load_audio(source)
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform.to(AIModelSettings.DEVICE)
    
        voice_feature_vector = VoiceConversionService.get_voice_feature_vector(voice_id)
        voice_feature_vector = voice_feature_vector.to(AIModelSettings.DEVICE)
        
        # Pass the attention mask to the model
        result = VOICE_CONVERSION.synthesis(waveform, voice_feature_vector)
        pred_log_mel = denormalize_tensor(result, mean=AudioSettings.LOG_MEL_MEAN, std=AudioSettings.LOG_MEL_STD)
        pred_log_mel = pred_log_mel.permute(0, 2, 1)
        audio = AudioService.generate_waveform_from_mel_db(pred_log_mel)
        audio = audio.squeeze(0)
        audio = AudioService.convert_waveform_to_base64(audio)
        return audio

    @staticmethod
    def get_available_voices() -> list:
        """
        Returns a list of available voices.
        """
        return [
            {
                "id": voice_info["id"],
                "name": voice_info["name"],
                "description": voice_info["description"],
            } for voice_info in VOICES.values()
        ]
    
    @staticmethod
    def get_voice_feature_vector(voice_id: str) -> torch.Tensor:
        """
        Returns the feature vector for a given voice.
        """
        if voice_id not in VOICES:
            raise ValueError(f"Voice '{voice_id}' not found.")
        
        voice_path = VOICES[voice_id]['path']
        feature_vector = torch.load(voice_path).unsqueeze(0).to(AIModelSettings.DEVICE)
        
        return feature_vector
