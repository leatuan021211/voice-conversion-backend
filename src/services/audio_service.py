import base64
from io import BytesIO
import tempfile

import numpy as np
import torch
import torchaudio
import librosa
from librosa.filters import mel as librosa_mel_fn
import soundfile as sf
import struct
import webrtcvad
from scipy.ndimage import binary_dilation


from core.settings import AudioSettings


class AudioService:

    @staticmethod
    def load_audio(audio: BytesIO) -> tuple[np.ndarray, int]:
        waveform, _ = librosa.load(audio.file, sr=AudioSettings.SAMPLE_RATE, mono=True)
        return waveform
    
    @staticmethod
    def trim_long_silences(wav: np.ndarray) -> np.ndarray:
        """
        Trims long silences from the audio using WebRTC VAD.
        Args:
            wav (np.ndarray): Input audio waveform.
        Returns:
            np.ndarray: Trimmed audio waveform.
        """
        samples_per_window = (AudioSettings.VAD_WINDOW_LENGTH * AudioSettings.SAMPLE_RATE) // 1000
        
        wav = wav[:len(wav) - (len(wav) % samples_per_window)]
        
        int16_max = (2 ** 15) - 1
        pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))
        
        voice_flags = []
        vad = webrtcvad.Vad(mode=3)
        for window_start in range(0, len(wav), samples_per_window):
            window_end = window_start + samples_per_window
            voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                            sample_rate=AudioSettings.SAMPLE_RATE))
        voice_flags = np.array(voice_flags)

        def moving_average(array, width):
            array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
            ret = np.cumsum(array_padded, dtype=float)
            ret[width:] = ret[width:] - ret[:-width]
            return ret[width - 1:] / width
        
        audio_mask = moving_average(voice_flags, AudioSettings.VAD_MOVING_AVERAGE_WIDTH)
        audio_mask = np.round(audio_mask).astype(np.bool)
        
        audio_mask = binary_dilation(audio_mask, np.ones(AudioSettings.VAD_MAX_SILENCE_LENGTH + 1))
        audio_mask = np.repeat(audio_mask, samples_per_window)
        
        return wav[audio_mask == True]

    @staticmethod
    def convert_waveform_to_base64(waveform: np.ndarray) -> str:
        """
        Converts a waveform to a base64 string.
        Args:
            waveform (np.ndarray): Input audio waveform.
        Returns:
            str: Base64 encoded string of the waveform.
        """

        buffer = BytesIO()
        sf.write(buffer, waveform, 16000, format='WAV')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    @staticmethod
    def griffin_lim(mel: np.ndarray, n_iter: int = 60) -> np.ndarray:
        """
        Applies Griffin-Lim algorithm to reconstruct the waveform from a mel spectrogram.
        Args:
            mel (np.ndarray): Input mel spectrogram.
            n_iter (int): Number of iterations for Griffin-Lim algorithm.
        Returns:
            np.ndarray: Reconstructed waveform.
        """
        mel = librosa.feature.inverse.mel_to_stft(mel, sr=AudioSettings.SAMPLE_RATE)
        waveform = librosa.griffinlim(mel, n_iter=n_iter, hop_length=AudioSettings.HOP_LENGTH)
        return waveform

    @staticmethod
    def denormalize(data: torch.Tensor, mu: float, std: float) -> torch.Tensor:
        if not isinstance(mu, float):
            if isinstance(mu, list):
                mu = torch.tensor(mu, dtype=data.dtype, device=data.device)
            elif isinstance(mu, torch.Tensor):
                mu = mu.to(data.device)
            elif isinstance(mu, np.ndarray):
                mu = torch.from_numpy(mu).to(data.device)
            mu = mu.unsqueeze(-1)
        if not isinstance(std, float):
            if isinstance(std, list):
                std = torch.tensor(std, dtype=data.dtype, device=data.device)
            elif isinstance(std, torch.Tensor):
                std = std.to(data.device)
            elif isinstance(std, np.ndarray):
                std = torch.from_numpy(std).to(data.device)
            std = std.unsqueeze(-1)
        return data * std + mu

    @staticmethod
    def generate_waveform_from_mel_db(mel_spec: torch.Tensor) -> np.ndarray:
        mel_spec = AudioService.denormalize(mel_spec, AudioSettings.MEL_MEAN, AudioSettings.MEL_STD)
        mel_spec = mel_spec.clone()
        mel_spec = torch.clamp(mel_spec, min=1e-5)
        mel_spec = torch.log(mel_spec)

        # Step 2: Invert Mel basis to get linear spectrogram
        mel_basis = librosa_mel_fn(
            sr=AudioSettings.SAMPLE_RATE, n_fft=AudioSettings.N_FFT, n_mels=AudioSettings.N_MELS, fmin=AudioSettings.F_MIN, fmax=AudioSettings.F_MAX
        )  # shape: [n_mels, n_fft//2 + 1]
        mel_basis_torch = torch.from_numpy(mel_basis).float().to(mel_spec.device)
        inv_mel_basis = torch.pinverse(mel_basis_torch)  # Pseudo-inverse
        linear_spec = torch.matmul(inv_mel_basis, mel_spec)  # shape: [n_fft//2+1, time]

        # Step 3: Griffin-Lim to estimate waveform
        griffinlim = torchaudio.transforms.GriffinLim(
            n_fft=AudioSettings.N_FFT,
            win_length=AudioSettings.WIN_LENGTH,
            hop_length=AudioSettings.HOP_LENGTH,
            window_fn=torch.hann_window,
            power=2.0,
            n_iter=60,
        ).to(mel_spec.device)

        # GriffinLim expects shape [freq, time]
        waveform = griffinlim(linear_spec)
        waveform = waveform.squeeze(0).detach().numpy().tolist()
        return waveform