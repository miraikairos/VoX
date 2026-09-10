import torch
from transformers import pipeline

# ==========================================
# MODEL
# ==========================================

MODEL_NAME = "Shanmugapriya6/voice-fake-detector-v1"

print("Loading model...")

device = 0 if torch.cuda.is_available() else -1

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")

classifier = pipeline(
    "audio-classification",
    model=MODEL_NAME,
    device=device
)

print("Model loaded successfully!")


# ==========================================
# AUDIO FILE
# ==========================================

audio_file = r"D:\sih\aasist\test.wav"

print("\nAnalyzing:", audio_file)

results = classifier(audio_file)


# ==========================================
# SHOW RAW PROBABILITIES
# ==========================================

print("\nRaw results:")

for result in results:
    print(f"{result['label']}: {result['score'] * 100:.4f}%")


# ==========================================
# GET BEST PREDICTION
# ==========================================

best = max(results, key=lambda x: x["score"])

label = best["label"]
confidence = best["score"] * 100


# ==========================================
# LABEL MAPPING
# LABEL_0 = GENUINE
# LABEL_1 = FAKE
# ==========================================

if label == "LABEL_0":
    prediction = "GENUINE"
else:
    prediction = "FAKE"


# ==========================================
# FINAL RESULT
# ==========================================

print("\n" + "=" * 45)
print("        VOICE CLONE CHECKER")
print("=" * 45)

print(f"Prediction : {prediction}")
print(f"Confidence : {confidence:.2f}%")

print("=" * 45)