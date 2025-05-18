from io import BytesIO
import torch

from core.settings import AIModelSettings
from core.constants import VOICE_ENCODER
from services.audio_service import AudioService


class VoiceEncoderService:

    @staticmethod
    def encode(audio: BytesIO):
        waveform = AudioService.load_audio(audio)
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform.to(AIModelSettings.DEVICE)
        feature_vector: torch.Tensor = VOICE_ENCODER(waveform)
        return feature_vector
