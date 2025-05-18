from io import BytesIO
import torch

from services.voice_encoder_service import VoiceEncoderService


class SpeakerVerificationService:
    
    @staticmethod
    def cal_cosine_similarity(audio1: BytesIO, audio2: BytesIO) -> float:
        feature_vector1 = VoiceEncoderService.encode(audio1)
        feature_vector2 = VoiceEncoderService.encode(audio2)
        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(feature_vector1, feature_vector2)
        return similarity.item()
