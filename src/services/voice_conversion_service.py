from io import BytesIO
import torch
import numpy as np

from core.settings import AIModelSettings, AudioSettings
from core.constants import VOICE_CONVERSION, TOKENIZER, VOICE_ENCODER
from services.audio_service import AudioService
from services.voice_encoder_service import VoiceEncoderService

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
    def converse_voice(source: BytesIO, target: BytesIO) -> list:
        feature_vector2 = VoiceEncoderService.encode(target)
        waveform = AudioService.load_audio(source)
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform.to(AIModelSettings.DEVICE)
        mel, content_feature, _ = VOICE_CONVERSION(waveform, feature_vector2)
        mel = AudioService.denormalize(mel, AudioSettings.MEL_MEAN, AudioSettings.MEL_STD)
        plot_mel_spectrogram(mel, title="Mel Spectrogram", save_path="mel_spectrogram.png")
        audio = AudioService.generate_waveform_from_mel_db(mel.squeeze(0).T)
        audio = AudioService.convert_waveform_to_base64(audio)
        predicted_ids = torch.argmax(content_feature, dim=-1)
        return audio, TOKENIZER.batch_decode(predicted_ids)
    
    @staticmethod
    def converse_voice_from_paths(source_path: str, target_path: str) -> list:
        """
        Convert voice from one speaker to another using file paths.
        
        Args:
            source_path (str): Path to the source audio file.
            target_path (str): Path to the target audio file (voice to convert to).
            
        Returns:
            tuple: (audio_base64, text) - The converted audio and extracted text.
        """
        try:
            # Load target audio for voice encoding
            import librosa
            target_waveform, _ = librosa.load(target_path, sr=AudioSettings.SAMPLE_RATE, mono=True)
            target_waveform = torch.from_numpy(target_waveform).unsqueeze(0)
            target_waveform = target_waveform.to(AIModelSettings.DEVICE)
            feature_vector2 = VOICE_ENCODER(target_waveform)
            
            # Load source audio
            source_waveform, _ = librosa.load(source_path, sr=AudioSettings.SAMPLE_RATE, mono=True)
            source_waveform = torch.from_numpy(source_waveform).unsqueeze(0)
            source_waveform = source_waveform.to(AIModelSettings.DEVICE)
            
            # Perform voice conversion
            mel, content_feature, _ = VOICE_CONVERSION(source_waveform, feature_vector2)
            mel = AudioService.denormalize(mel, AudioSettings.MEL_MEAN, AudioSettings.MEL_STD)
            plot_mel_spectrogram(mel, title="Mel Spectrogram", save_path="mel_spectrogram.png")
            audio = AudioService.generate_waveform_from_mel_db(mel.squeeze(0).T)
            audio = AudioService.convert_waveform_to_base64(audio)
            predicted_ids = torch.argmax(content_feature, dim=-1)
            return audio, TOKENIZER.batch_decode(predicted_ids)
        except Exception as e:
            # If librosa fails, try with soundfile
            import soundfile as sf
            import numpy as np
            
            try:
                target_waveform, sr = sf.read(target_path)
                if target_waveform.ndim > 1:
                    target_waveform = np.mean(target_waveform, axis=1)  # Convert to mono
                if sr != AudioSettings.SAMPLE_RATE:
                    target_waveform = librosa.resample(target_waveform, orig_sr=sr, target_sr=AudioSettings.SAMPLE_RATE)
                
                source_waveform, sr = sf.read(source_path)
                if source_waveform.ndim > 1:
                    source_waveform = np.mean(source_waveform, axis=1)  # Convert to mono
                if sr != AudioSettings.SAMPLE_RATE:
                    source_waveform = librosa.resample(source_waveform, orig_sr=sr, target_sr=AudioSettings.SAMPLE_RATE)
                
                target_waveform = torch.from_numpy(target_waveform).unsqueeze(0)
                source_waveform = torch.from_numpy(source_waveform).unsqueeze(0)
                
                target_waveform = target_waveform.to(AIModelSettings.DEVICE)
                source_waveform = source_waveform.to(AIModelSettings.DEVICE)
                
                feature_vector2 = VOICE_ENCODER(target_waveform)
                mel, content_feature, _ = VOICE_CONVERSION(source_waveform, feature_vector2)
                mel = AudioService.denormalize(mel, AudioSettings.MEL_MEAN, AudioSettings.MEL_STD)
                plot_mel_spectrogram(mel, title="Mel Spectrogram", save_path="mel_spectrogram.png")
                audio = AudioService.generate_waveform_from_mel_db(mel.squeeze(0).T)
                audio = AudioService.convert_waveform_to_base64(audio)
                predicted_ids = torch.argmax(content_feature, dim=-1)
                return audio, TOKENIZER.batch_decode(predicted_ids)
            except Exception as sf_error:
                raise ValueError(f"Could not process audio files: {str(sf_error)}")
