from transformers import AutoTokenizer

from ai.models.voice_encoder.hubert_voice_encoder import HuBERTVoiceEncoder
from ai.models.voice_conversion.voice_conversion import VoiceConversion


VOICE_ENCODER = HuBERTVoiceEncoder.load_from_checkpoint("ai/checkpoints/voice_encoder.pt")
VOICE_CONVERSION = VoiceConversion.load_from_checkpoint("ai/checkpoints/voice_conversion_epoch57.pt")
TOKENIZER = AutoTokenizer.from_pretrained("nguyenvulebinh/wav2vec2-base-vietnamese-250h")