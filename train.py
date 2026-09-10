"""
train.py

Fine-tunes AASIST on a custom genuine/spoof dataset (NOT the ASVspoof2019
protocol structure).

Expected dataset layout:

    VoiceDeepfakeDataset/
        genuine/   -> real human voice audio  (label 0)
        spoof/     -> AI-generated / cloned / fake voice audio  (label 1)

Run with:
    python train.py
"""

import os
import random

import numpy as np
import librosa

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

from models.AASIST import Model


# ============================================================
# CONFIGURATION
# ============================================================

# CHANGE THIS TO YOUR DATASET LOCATION
DATASET_ROOT = r"D:\sih\aasist\VoiceDeepfakeDataset"

# Training parameters
BATCH_SIZE = 8
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Fraction of data held out for validation
VAL_FRACTION = 0.2

# AASIST expects 64600 samples at 16 kHz
NUM_SAMPLES = 64600
SAMPLE_RATE = 16000

# Label convention (must match dataset folders below)
LABEL_GENUINE = 0
LABEL_SPOOF = 1

# Formats librosa/audioread can read. Some (m4a, ogg, webm) require
# ffmpeg to be installed and on PATH — if a file fails to load, it is
# skipped with a warning rather than crashing the whole run.
AUDIO_EXTENSIONS = (
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".ogg",
    ".webm",
)

# Output directory
OUTPUT_DIR = "trained_models"

# Random seed
SEED = 42

# Optionally start from a pretrained checkpoint (fine-tuning) instead of
# training from scratch. Set to None to train from random init.
PRETRAINED_WEIGHTS = os.path.join("models", "weights", "AASIST.pth")


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("AASIST TRAINING")
print("=" * 60)
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("=" * 60)


# ============================================================
# AASIST MODEL CONFIGURATION
# (unchanged — must match models/AASIST.py and any pretrained checkpoint)
# ============================================================

MODEL_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [
        70,
        [1, 32],
        [32, 32],
        [32, 64],
        [64, 64]
    ],
    "gat_dims": [
        64,
        32
    ],
    "pool_ratios": [
        0.5,
        0.7,
        0.5,
        0.5
    ],
    "temperatures": [
        2.0,
        2.0,
        100.0,
        100.0
    ]
}


# ============================================================
# LOAD DATASET (genuine/ + spoof/ folder structure)
# ============================================================

if not os.path.exists(DATASET_ROOT):
    raise FileNotFoundError(
        f"Dataset directory not found:\n{DATASET_ROOT}"
    )


def load_dataset():

    samples = []

    genuine_dir = os.path.join(DATASET_ROOT, "genuine")
    spoof_dir = os.path.join(DATASET_ROOT, "spoof")

    if not os.path.exists(genuine_dir):
        raise FileNotFoundError(f"Genuine folder not found: {genuine_dir}")

    if not os.path.exists(spoof_dir):
        raise FileNotFoundError(f"Spoof folder not found: {spoof_dir}")

    # Genuine = 0
    for root, _, files in os.walk(genuine_dir):
        for file in files:
            if file.lower().endswith(AUDIO_EXTENSIONS):
                samples.append(
                    (os.path.join(root, file), LABEL_GENUINE)
                )

    # Spoof = 1
    for root, _, files in os.walk(spoof_dir):
        for file in files:
            if file.lower().endswith(AUDIO_EXTENSIONS):
                samples.append(
                    (os.path.join(root, file), LABEL_SPOOF)
                )

    return samples


all_samples = load_dataset()

num_genuine = sum(1 for _, label in all_samples if label == LABEL_GENUINE)
num_spoof = sum(1 for _, label in all_samples if label == LABEL_SPOOF)

print("\n" + "=" * 60)
print("DATASET LOADED")
print("=" * 60)
print(f"Total samples:   {len(all_samples)}")
print(f"Genuine samples: {num_genuine}")
print(f"Spoof samples:   {num_spoof}")

if num_genuine == 0 or num_spoof == 0:
    raise RuntimeError(
        "Need at least one file in both 'genuine' and 'spoof' folders."
    )


# ============================================================
# TRAIN / VALIDATION SPLIT (stratified, so class balance is preserved)
# ============================================================

paths = [p for p, _ in all_samples]
labels = [l for _, l in all_samples]

train_paths, dev_paths, train_labels, dev_labels = train_test_split(
    paths,
    labels,
    test_size=VAL_FRACTION,
    random_state=SEED,
    stratify=labels,
)

train_samples = list(zip(train_paths, train_labels))
dev_samples = list(zip(dev_paths, dev_labels))

print(f"Training samples:   {len(train_samples)}")
print(f"Validation samples: {len(dev_samples)}")
print("=" * 60)


# ============================================================
# DATASET
# ============================================================

class VoiceDeepfakeDataset(Dataset):
    """
    Loads audio of any supported format, converts to mono, resamples to
    16 kHz, and crops/pads to exactly NUM_SAMPLES. Random crop during
    training, center crop during validation.
    """

    def __init__(self, samples, training=False):
        self.samples = samples
        self.training = training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        try:
            # librosa.load handles: format decoding (via soundfile/audioread,
            # ffmpeg backend for mp3/m4a/ogg/webm if installed), stereo->mono
            # (mono=True), and resampling to SAMPLE_RATE — all in one call.
            audio, _ = librosa.load(
                path,
                sr=SAMPLE_RATE,
                mono=True,
            )

            if audio.size == 0:
                raise ValueError("Empty audio file")

            audio = torch.tensor(audio, dtype=torch.float32)

            # ------------------------------------------------
            # RANDOM CROP DURING TRAINING / CENTER CROP FOR VAL
            # ------------------------------------------------

            if len(audio) > NUM_SAMPLES:

                if self.training:
                    start = random.randint(0, len(audio) - NUM_SAMPLES)
                else:
                    start = (len(audio) - NUM_SAMPLES) // 2

                audio = audio[start:start + NUM_SAMPLES]

            # ------------------------------------------------
            # PAD SHORT AUDIO (repeat-pad, consistent with AASIST's
            # own training convention rather than zero-padding)
            # ------------------------------------------------

            elif len(audio) < NUM_SAMPLES:
                num_repeats = (NUM_SAMPLES // len(audio)) + 1
                audio = audio.repeat(num_repeats)[:NUM_SAMPLES]

            return audio, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"\n[warn] Failed to load {path}: {e}")
            # Return a silent placeholder rather than crashing the whole
            # epoch on one bad file. It keeps its true label, so a handful
            # of corrupt files won't skew training in a fixed direction.
            audio = torch.zeros(NUM_SAMPLES, dtype=torch.float32)
            return audio, torch.tensor(label, dtype=torch.long)


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = VoiceDeepfakeDataset(train_samples, training=True)
dev_dataset = VoiceDeepfakeDataset(dev_samples, training=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

print("\nDataset:")
print("Training samples:", len(train_dataset))
print("Development samples:", len(dev_dataset))


# ============================================================
# MODEL
# ============================================================

model = Model(MODEL_CONFIG)

if PRETRAINED_WEIGHTS and os.path.exists(PRETRAINED_WEIGHTS):
    print(f"\nLoading pretrained weights from: {PRETRAINED_WEIGHTS}")
    state = torch.load(PRETRAINED_WEIGHTS, map_location="cpu")
    # Some checkpoints save a raw state_dict, others wrap it in a dict
    # (e.g. {"model_state_dict": ...}) like this script's own saves do.
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    print("Pretrained weights loaded — fine-tuning from AASIST.pth")
else:
    print("\nNo pretrained weights loaded — training from random init.")

model = model.to(DEVICE)

print("\nModel:")
print(model)

number_parameters = sum(p.numel() for p in model.parameters())
print(f"\nNumber of parameters: {number_parameters:,}")


# ============================================================
# LOSS / OPTIMIZER / SCHEDULER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS,
    eta_min=5e-6
)


def calculate_eer(labels, scores):
    """
    scores = model's predicted probability of the SPOOF class (label 1).
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=LABEL_SPOOF)
    fnr = 1 - tpr

    index = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[index] + fnr[index]) / 2

    return eer


# ============================================================
# TRAINING LOOP
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

best_eer = float("inf")
best_path = os.path.join(OUTPUT_DIR, "best_model.pth")
last_path = os.path.join(OUTPUT_DIR, "last_model.pth")

for epoch in range(1, NUM_EPOCHS + 1):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")

    for audio, labels in progress:

        audio = audio.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        # AASIST's forward() returns (last_hidden, output) — output is the
        # (batch, 2) classification logits used for the loss.
        _, output = model(audio)

        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * audio.size(0)

        predictions = torch.argmax(output, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        progress.set_postfix(loss=loss.item())

    train_loss = running_loss / total
    train_accuracy = correct / total

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    model.eval()

    dev_loss = 0.0
    dev_total = 0
    dev_correct = 0

    all_labels = []
    all_scores = []

    with torch.no_grad():

        for audio, labels in tqdm(dev_loader, desc="Validation"):

            audio = audio.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            _, output = model(audio)

            loss = criterion(output, labels)

            dev_loss += loss.item() * audio.size(0)
            dev_total += labels.size(0)

            predictions = torch.argmax(output, dim=1)
            dev_correct += (predictions == labels).sum().item()

            # Probability of the SPOOF class (label 1), used for EER.
            scores = torch.softmax(output, dim=1)[:, LABEL_SPOOF]

            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    dev_loss = dev_loss / dev_total
    dev_accuracy = dev_correct / dev_total
    dev_eer = calculate_eer(all_labels, all_scores)

    scheduler.step()
    current_lr = optimizer.param_groups[0]["lr"]

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print(f"Epoch: {epoch}/{NUM_EPOCHS}")
    print(f"Train Loss: {train_loss:.6f}")
    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Dev Loss: {dev_loss:.6f}")
    print(f"Dev Accuracy: {dev_accuracy * 100:.2f}%")
    print(f"Dev EER: {dev_eer * 100:.4f}%")
    print(f"Learning Rate: {current_lr:.8f}")
    print("=" * 60)

    # --------------------------------------------------------
    # SAVE BEST MODEL (by validation EER)
    # --------------------------------------------------------

    if dev_eer < best_eer:

        best_eer = dev_eer

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_eer": best_eer,
                "dev_accuracy": dev_accuracy,
                "model_config": MODEL_CONFIG,
                "label_convention": {"genuine": LABEL_GENUINE, "spoof": LABEL_SPOOF},
            },
            best_path
        )

        print(f"\nNew best model saved:\n{best_path}")

    # --------------------------------------------------------
    # SAVE LAST CHECKPOINT (every epoch, for resuming)
    # --------------------------------------------------------

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_eer": best_eer,
            "dev_accuracy": dev_accuracy,
            "model_config": MODEL_CONFIG,
            "label_convention": {"genuine": LABEL_GENUINE, "spoof": LABEL_SPOOF},
        },
        last_path
    )


print("\n")
print("=" * 60)
print("TRAINING FINISHED")
print("=" * 60)
print(f"Best Dev EER: {best_eer * 100:.4f}%")
print("Best model:", best_path)
print("Last checkpoint:", last_path)
print("=" * 60)
