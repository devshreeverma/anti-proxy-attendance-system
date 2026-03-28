import pickle
import numpy as np
import sounddevice as sd
import librosa

print("🔄 Loading model...")

with open("model/voice_model.pkl", "rb") as f:
    model, scaler, label_map = pickle.load(f)

print("✅ Model loaded")
print("🎤 Speak now...")

fs = 44100
seconds = 3  # MUST match training

audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()

print("✅ Recording complete")

audio = audio.flatten()

# ================= PREPROCESS =================
audio, _ = librosa.effects.trim(audio)

if np.max(np.abs(audio)) != 0:
    audio = audio / np.max(np.abs(audio))

# FIX LENGTH
max_len = 3 * fs
if len(audio) < max_len:
    audio = np.pad(audio, (0, max_len - len(audio)))
else:
    audio = audio[:max_len]

# ================= FEATURES =================

# MFCC
mfcc = librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=20)
mfcc_mean = np.mean(mfcc.T, axis=0)
mfcc_std = np.std(mfcc.T, axis=0)

# Chroma
chroma = librosa.feature.chroma_stft(y=audio, sr=fs)
chroma_mean = np.mean(chroma.T, axis=0)

# Spectral Contrast
spectral = librosa.feature.spectral_contrast(y=audio, sr=fs)
spectral_mean = np.mean(spectral.T, axis=0)

# ZCR + Energy
zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
energy = np.mean(audio ** 2)

# Final feature vector
feature_vector = np.concatenate((
    mfcc_mean,
    mfcc_std,
    chroma_mean,
    spectral_mean,
    [zcr, energy]
)).reshape(1, -1)

# ================= SCALE =================
feature_vector = scaler.transform(feature_vector)

print("🔍 Predicting...")

probs = model.predict_proba(feature_vector)
confidence = np.max(probs)

prediction = model.predict(feature_vector)[0]
person = label_map[str(prediction)] if isinstance(list(label_map.keys())[0], str) else label_map[prediction]

print(f"Confidence: {confidence:.2f}")

if confidence < 0.6:
    print("⚠️ Low confidence — Possible proxy / try again")
else:
    print(f"✅ Recognized: {person}")
