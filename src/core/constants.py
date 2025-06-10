from utils.json_utils import load_json
from ai.models.voice_encoder.hubert_voice_encoder import HuBERTVoiceEncoder
from ai.models.voice_conversion.voice_conversion import VoiceConversion
from ai.models.hifigan.hifigan import Generator as HiFiGANGenerator
from ai.utils.denoiser import Denoiser

VOICE_ENCODER = HuBERTVoiceEncoder.load_from_checkpoint("static/checkpoints/voice_encoder.pt")
VOICE_CONVERSION = VoiceConversion.load_from_checkpoint("static/checkpoints/voice_conversion-old.pt")
HIFIGAN_VOCODER_CONFIG = load_json("static/vocoder_config.json")
VOCODER = HiFiGANGenerator.load_from_checkpoint("static/checkpoints/vocoder9600", config=HIFIGAN_VOCODER_CONFIG)
VOICES = load_json("static/voices.json")
DENOISER = Denoiser(VOCODER, filter_length=1280, n_overlap=4, win_length=1280, mode="normal")