import sys
import torch
import torch.nn.functional as F
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
import math
from models.AASIST import Model


DEVICE = torch.device("cpu")
MODEL_PATH = "./models/weights/AASIST.pth"

MODEL_CONFIG = {
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0]
}

NB_SAMP = 64600


def load_model():
    model = Model(MODEL_CONFIG)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


def load_audio(audio_path):

    audio, sample_rate = sf.read(audio_path)

    # Stereo → Mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    print(f"Sample rate: {sample_rate}")
    print(f"Original samples: {len(audio)}")



# Resample audio to 16 kHz
    if sample_rate != 16000:

        print(f"Resampling {sample_rate} Hz → 16000 Hz...")

        gcd = math.gcd(sample_rate, 16000)

        up = 16000 // gcd
        down = sample_rate // gcd

        audio = resample_poly(audio, up, down)

        sample_rate = 16000

    # Crop long audio
    if len(audio) > NB_SAMP:
        audio = audio[:NB_SAMP]

    # Repeat short audio
    elif len(audio) < NB_SAMP:
        repeat = NB_SAMP // len(audio) + 1
        audio = np.tile(audio, repeat)
        audio = audio[:NB_SAMP]

    audio_tensor = torch.tensor(
        audio,
        dtype=torch.float32
    ).unsqueeze(0)

    return audio_tensor


def predict(audio_path, model):

    audio = load_audio(audio_path).to(DEVICE)

    with torch.no_grad():

        _, output = model(audio)

        probabilities = F.softmax(output, dim=1)

    probabilities = probabilities.cpu().numpy()[0]

    predicted_class = int(np.argmax(probabilities))
    confidence = probabilities[predicted_class] * 100
    if predicted_class == 1:
        result = "GENUINE HUMAN VOICE"
    else:
        result = "SPOOF / AI-GENERATED VOICE"

    print("\n========== RESULT ==========")
    print(f"Result: {result}")
    print(f"Genuine probability: {probabilities[1] * 100:.2f}%")
    print(f"Spoof probability: {probabilities[0] * 100:.2f}%")
    print(f"Confidence: {confidence:.2f}%")
    print("============================\n")

    print("\n========== RESULT ==========")
    print(f"Raw probabilities: {probabilities}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print("============================\n")

    return predicted_class, probabilities


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage:")
        print("python predict.py your_audio.wav")
        sys.exit(1)

    print("Loading AASIST model...")

    model = load_model()

    print("Model loaded successfully!")

    predict(sys.argv[1], model)