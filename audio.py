import librosa
import numpy as np
import torch
import subprocess
import tempfile
import os

SAMPLE_RATE = 16000
MAX_LEN = 64600

MIN_DURATION = 0.5
MIN_RMS = 0.001


def load_audio(path):
    """
    Try librosa first.
    If it fails, use FFmpeg to convert the audio to WAV.
    """

    try:
        audio, _ = librosa.load(
            path,
            sr=SAMPLE_RATE,
            mono=True
        )
        return audio

    except Exception:
        print("Librosa failed. Trying FFmpeg...")

    temp_wav = None

    try:
        temp = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        )

        temp_wav = temp.name
        temp.close()

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", path,
                "-ac", "1",
                "-ar", str(SAMPLE_RATE),
                temp_wav
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )

        audio, _ = librosa.load(
            temp_wav,
            sr=SAMPLE_RATE,
            mono=True
        )

        return audio

    except Exception as e:
        print("FFmpeg error:", e)
        raise ValueError("Could not read or convert the audio file")

    finally:
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)


def process_audio(path, device):

    # Load audio
    audio = load_audio(path)

    # Empty audio
    if len(audio) == 0:
        raise ValueError("Audio file is empty")

    # Check duration
    duration = len(audio) / SAMPLE_RATE

    print(f"Audio duration: {duration:.2f} seconds")

    if duration < MIN_DURATION:
        raise ValueError(
            "Audio is too short. Please provide at least 0.5 seconds."
        )

    # Calculate loudness
    rms = np.sqrt(np.mean(audio ** 2))

    print(f"Audio RMS: {rms:.6f}")

    # Detect silence
    if rms < MIN_RMS:
        raise ValueError(
            "Audio is too quiet or contains mostly silence."
        )

    # Crop long audio
    if len(audio) >= MAX_LEN:
        audio = audio[:MAX_LEN]

    # Repeat short audio
    else:
        repeats = (MAX_LEN // len(audio)) + 1
        audio = np.tile(audio, repeats)[:MAX_LEN]

    # Convert to PyTorch tensor
    audio = torch.tensor(
        audio,
        dtype=torch.float32
    )

    # Add batch dimension
    audio = audio.unsqueeze(0)

    return audio.to(device)