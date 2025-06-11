from typing import BinaryIO
import torch
import numpy as np
import librosa

from core.settings import AIModelSettings, AudioSettings
from core.constants import VOICE_ENCODER
from services.audio_service import AudioService


class VoiceEncoderService:

    @staticmethod
    def encode(audio: BinaryIO):
        waveform = AudioService.load_audio(audio)
        waveform = AudioService.trim_long_silences(waveform)
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform.to(AIModelSettings.DEVICE)
        feature_vector: torch.Tensor = VOICE_ENCODER(waveform)
        return feature_vector
    
    @staticmethod
    def encode_from_path(audio_path: str):
        """
        Encode audio from a file path directly.
        
        Args:
            audio_path (str): Path to the audio file.
            
        Returns:
            torch.Tensor: Feature vector for the audio.
        """
        try:
            # Use librosa directly for loading from path
            waveform, _ = librosa.load(audio_path, sr=AudioSettings.SAMPLE_RATE, mono=True)
            waveform = torch.from_numpy(waveform).unsqueeze(0)
            waveform = waveform.to(AIModelSettings.DEVICE)
            feature_vector: torch.Tensor = VOICE_ENCODER(waveform)
            return feature_vector
        except Exception as e:
            # Fallback to soundfile if librosa fails
            import soundfile as sf
            try:
                waveform, sr = sf.read(audio_path)
                if waveform.ndim > 1:
                    waveform = np.mean(waveform, axis=1)  # Convert to mono
                if sr != AudioSettings.SAMPLE_RATE:
                    # Resample if needed
                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=AudioSettings.SAMPLE_RATE)
                waveform = torch.from_numpy(waveform).unsqueeze(0)
                waveform = waveform.to(AIModelSettings.DEVICE)
                feature_vector: torch.Tensor = VOICE_ENCODER(waveform)
                return feature_vector
            except Exception as sf_error:
                raise ValueError(f"Could not process audio file: {str(sf_error)}")
