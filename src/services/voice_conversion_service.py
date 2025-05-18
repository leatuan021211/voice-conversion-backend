from io import BytesIO
import torch
import numpy as np

from core.settings import AIModelSettings
from core.constants import VOICE_CONVERSION
from services.audio_service import AudioService
from services.voice_encoder_service import VoiceEncoderService


class VoiceConversionService:
    
    @staticmethod
    def converse_voice(source: BytesIO, target: BytesIO) -> list:
        feature_vector2 = VoiceEncoderService.encode(target)
        waveform = AudioService.load_audio(source)
        print("MINIMUM SOURCE VALUE:", min(waveform))
        print("MAXIMUM SOURCE VALUE:", max(waveform))
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        waveform = waveform.to(AIModelSettings.DEVICE)
        mel, _ = VOICE_CONVERSION(waveform, feature_vector2)
        audio = AudioService.generate_waveform_from_mel_db(mel.squeeze(0).T)
        audio = AudioService.convert_waveform_to_base64(audio)
        return audio
