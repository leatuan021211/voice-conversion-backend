from utils.ai_utils import get_device

class AudioSettings:
    """Audio settings for the model."""

    # Common settings
    SAMPLE_RATE = 16000
    N_FFT = 1280
    N_MELS = 80
    HOP_LENGTH = 320
    WIN_LENGTH = 1280
    F_MIN = 0
    F_MAX = 8000
    POWER = 2.0
    CENTER = False
    
    # VAD TRIMMING
    VAD_WINDOW_LENGTH = 30
    VAD_MOVING_AVERAGE_WIDTH = 8
    VAD_MAX_SILENCE_LENGTH = 6

    # Normalization
    LOG_MEL_MEAN = -14.719878446839134
    LOG_MEL_STD = 19.178215519514534
    
    # Griffin-Lim settings
    GRIFFIN_LIM_N_ITER = 32


class SpeakerVerificationSettings:
    
    THRESHOLD = 0.8


class AIModelSettings:
    """Settings for the AI model."""

    DEVICE = get_device()


class LoggerSettings:
    """Logger settings."""

    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = "app.log"
    LOG_DIR = "logs"


class VoiceSettings:
    
    VOICE_FOLDER = "core/voices"
    VOICE_LIST_FILE = "voice_list.json"