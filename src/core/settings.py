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
    CENTER = False
    
    # VAD TRIMMING
    VAD_WINDOW_LENGTH = 30
    VAD_MOVING_AVERAGE_WIDTH = 8
    VAD_MAX_SILENCE_LENGTH = 6

    # Normalization
    MEL_MEAN = -4.51203894828719
    MEL_STD = 2.0027185876269336


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