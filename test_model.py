from transformers import pipeline

MODEL_NAME = "Shanmugapriya6/voice-fake-detector-v1"

print("Loading model...")
classifier = pipeline(
    "audio-classification",
    model=MODEL_NAME
)

print("Model loaded!")

audio_file = "test.wav"

print(f"\nTesting: {audio_file}")

result = classifier(audio_file)

print("\nRESULT:")
print(result)