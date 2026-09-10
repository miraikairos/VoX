# VOX — AI Voice Deepfake Detection

VOX is an AI-powered voice authenticity detection platform designed to identify whether an audio sample is **genuine human speech or potentially AI-generated/voice-cloned speech**.

## Features

- 🎙️ **Browser Voice Recording** — Record audio directly from the browser.
- 📁 **Audio Upload** — Upload an existing audio file for analysis.
- 🤖 **AI-Based Detection** — Uses a deep-learning audio classification model to analyze voice authenticity.
- 📊 **Confidence Scores** — Displays genuine and spoof probabilities.
- ⚡ **Fast API-Based Inference** — Audio is processed through a FastAPI backend.
- 🔒 **Temporary Audio Processing** — Uploaded audio is temporarily stored for inference and removed afterward.
- 🌐 **Web-Based Interface** — Simple interface accessible through a browser.

## How It Works

```text
User
 │
 ├── Upload Audio
 │       OR
 └── Record Voice
          │
          ▼
     VOX Frontend
          │
          ▼
     FastAPI Backend
          │
          ▼
   Deepfake Detection Model
          │
          ▼
 Genuine / Spoof Prediction
          │
          ▼
    Confidence Scores
          │
          ▼
     VOX Frontend
```

## AI Model

VOX integrates the **`Shanmugapriya6/voice-fake-detector-v1`** deep-learning model.

The model is based on:

```text
facebook/wav2vec2-xls-r-300m
```

It performs binary audio classification to distinguish between genuine and spoofed/AI-generated speech.

The model is loaded through the Hugging Face Transformers library and performs inference locally through the Python backend.

## Technology Stack

### Frontend

- HTML
- CSS
- JavaScript
- Browser MediaRecorder API

### Backend

- Python
- FastAPI
- Uvicorn
- PyTorch
- Hugging Face Transformers

### Model

- Wav2Vec2 XLS-R
- Audio Classification
- Voice Deepfake Detection

### Deployment / Testing

- GitHub
- Netlify
- Cloudflare Tunnel

## Project Structure

```text
VOX/
│
├── backend/
│   ├── ai-service/
│   │   ├── app.py
│   │   ├── audio.py
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── .gitignore
│   │
│   ├── app.js
│   ├── package.json
│   └── package-lock.json
│
├── index.html
├── dd.html
│
├── main.py
├── predict.py
├── train.py
├── evaluation.py
├── data_utils.py
├── utils.py
│
├── test_model.py
├── test_voice_detector.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

## Running the AI Backend Locally

Navigate to the AI service:

```bash
cd backend/ai-service
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI server:

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

## API

### Health Check

```http
GET /health
```

### Voice Prediction

```http
POST /predict
```

The endpoint accepts an audio file and returns a prediction containing:

```json
{
  "success": true,
  "filename": "audio.wav",
  "prediction": "genuine",
  "confidence": 99.99,
  "spoof_probability": 0.01,
  "genuine_probability": 99.99
}
```

## Privacy

Audio files are temporarily stored during processing and deleted after inference. VOX does not need to permanently store uploaded voice recordings for the detection process.

## Limitations

Voice deepfake detection performance can vary depending on:

- Audio quality
- Background noise
- Recording conditions
- Language and accent
- Compression and audio format
- Type of voice-generation technology used

The system should therefore be treated as a **detection aid**, not as definitive forensic or legal evidence.

## Future Scope

- Real-time voice deepfake detection
- Improved multilingual detection
- Detection of newer voice-cloning techniques
- Advanced audio preprocessing
- Model fine-tuning with larger and more diverse datasets
- Cloud-based scalable inference
- Browser and mobile integration

## Disclaimer

VOX is an experimental AI-based voice authenticity detection system developed for educational, research, and demonstration purposes. Detection results should not be considered definitive proof of authenticity or manipulation.

## License

This project is intended for educational and research purposes.
