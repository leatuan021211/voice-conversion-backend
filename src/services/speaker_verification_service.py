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
    
    @staticmethod
    def cal_cosine_similarity_from_paths(audio1_path: str, audio2_path: str) -> float:
        """
        Calculate cosine similarity between two audio files using their file paths.
        
        Args:
            audio1_path (str): Path to the first audio file.
            audio2_path (str): Path to the second audio file.
            
        Returns:
            float: Cosine similarity between the two audio files.
        """
        feature_vector1 = VoiceEncoderService.encode_from_path(audio1_path)
        feature_vector2 = VoiceEncoderService.encode_from_path(audio2_path)
        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(feature_vector1, feature_vector2)
        return similarity.item()
