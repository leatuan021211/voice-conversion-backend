import os
import torch
from torch.utils.data import Dataset

from services.audio_service import AudioService
from ai.modules.content_encoder import ContentEncoder

class PrepareDataset(Dataset):
    """
    Custom dataset class for preparing data for training or evaluation.
    This class is used to load and preprocess the data.
    """

    def __init__(self, path: str):
        self.path = path
        self.files = [f for f in os.listdir(path) if f.endswith('.wav')]
        self.speech_to_text = ContentEncoder()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.path, self.files[idx])
        waveform = AudioService.load_audio(file_path)
        trimmed_audio = AudioService.trim_long_silences(waveform)
        input_tensor = torch.tensor(trimmed_audio, dtype=torch.float32).unsqueeze(0)
        tokens = self.speech_to_text(input_tensor) 
        return trimmed_audio, tokens
    
    