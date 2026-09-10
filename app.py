import os
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline


# ==========================================
# FASTAPI
# ==========================================

app = FastAPI(
    title="Voice Deepfake Detection API"
)


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# DEVICE
# ==========================================

DEVICE = 0 if torch.cuda.is_available() else -1

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU")


# ==========================================
# LOAD MODEL ONCE
# ==========================================

MODEL_NAME = "Shanmugapriya6/voice-fake-detector-v1"

print("Loading voice detection model...")

classifier = pipeline(
    "audio-classification",
    model=MODEL_NAME,
    device=DEVICE
)

print("Model loaded successfully!")


# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():
    return {
        "message": "Voice Deepfake Detection API running",
        "model": MODEL_NAME
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():
    return {
        "status": "online",
        "model_loaded": True,
        "device": "cuda" if DEVICE == 0 else "cpu"
    }


# ==========================================
# PREDICT
# ==========================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    allowed_extensions = [
        ".wav",
        ".mp3",
        ".flac",
        ".m4a",
        ".ogg",
        ".webm"
    ]

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided"
        )

    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported audio format. "
                "Allowed: WAV, MP3, FLAC, M4A, OGG, WEBM"
            )
        )

    temp_path = None

    try:

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp:

            content = await file.read()

            temp.write(content)

            temp_path = temp.name


        # ==========================================
        # RUN MODEL
        # ==========================================

        results = classifier(temp_path)


        # Get best prediction
        best = max(
            results,
            key=lambda x: x["score"]
        )

        label = best["label"]
        confidence = float(best["score"]) * 100


        # ==========================================
        # LABEL MAPPING
        # ==========================================

        if label == "LABEL_0":
            prediction = "genuine"
        elif label == "LABEL_1":
            prediction = "spoof"
        else:
            prediction = label.lower()


        # Get both probabilities safely
        genuine_probability = 0
        spoof_probability = 0

        for result in results:

            score = float(result["score"]) * 100

            if result["label"] == "LABEL_0":
                genuine_probability = score

            elif result["label"] == "LABEL_1":
                spoof_probability = score


        # ==========================================
        # RESPONSE
        # ==========================================

        return {
            "success": True,
            "filename": file.filename,

            "prediction": prediction,

            "confidence": round(confidence, 2),

            "spoof_probability": round(
                spoof_probability,
                2
            ),

            "genuine_probability": round(
                genuine_probability,
                2
            )
        }


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


    finally:

        if temp_path and os.path.exists(temp_path):

            os.remove(temp_path)