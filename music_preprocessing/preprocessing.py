import os
import torch
import torchaudio
import numpy as np

from fastai.vision.all import Image


def load_audio_and_cut_length(filename: str) -> torch.Tensor:
    wave, sample_rate = torchaudio.load(filename)
    cut_wave = wave[:, 0:sample_rate*30] #Audio length 30s
    return cut_wave


def slice_audio(wave: torch.Tensor, num_slices: int) -> list[torch.Tensor]:
    if num_slices == 1:
        return [wave]

    possible_even_slices = [3, 5, 6, 10]

    if num_slices not in possible_even_slices:
        raise ValueError("num_slices must one of valid number: 2,6,10")

    output = []
    hop = wave.shape[1] // num_slices # (1, 661500)

    for i in range(num_slices):
        output.append(wave[:, i*hop:(i+1)*hop])

    return output


def create_spectrogram_image(mel_spec_transformer, wave: torch.Tensor) -> np.ndarray:
    spectrogram = mel_spec_transformer(wave)
    spectrogram = spectrogram.squeeze().numpy()
    spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min()) * 255;
    spectrogram = spectrogram.astype('uint8')

    return spectrogram


def save_spectrogram_as_image(spectrogram: np.ndarray, filename: str):
    Image.fromarray(spectrogram).save(filename)


def create_file_name(filename: str, output_dir: str, apply_idx: bool, idx: int) -> str:
    genre, song_number, ext = filename.split(".")
    if apply_idx:
        song_number = song_number + f"_{idx}"
    ext = "png"

    return os.path.join(output_dir, f"{genre}.{song_number}.{ext}")
