import os
import librosa
import numpy as np
import pandas as pd
import json

DATASET_PATH = "C:/Users/Admin/Desktop/NIT SIKKIM/voice_attendance_project/voice_dataset"

def extract_features():
    features = []
    labels = []
    label_map = {}

    people = sorted(os.listdir(DATASET_PATH))

    for label, person in enumerate(people):
        person_path = os.path.join(DATASET_PATH, person)

        if not os.path.isdir(person_path):
            continue

        label_map[label] = person

        for file in os.listdir(person_path):
            if not file.endswith(".wav"):
                continue

            file_path = os.path.join(person_path, file)

            try:
                # ================= LOAD AUDIO =================
                audio, sr = librosa.load(file_path, sr=None)

                # Remove silence
                audio, _ = librosa.effects.trim(audio)

                # Normalize audio
                if np.max(np.abs(audio)) != 0:
                    audio = audio / np.max(np.abs(audio))

                # ================= FIX LENGTH =================
                max_len = 3 * sr  # 3 seconds

                if len(audio) < max_len:
                    audio = np.pad(audio, (0, max_len - len(audio)))
                else:
                    audio = audio[:max_len]

                # ================= MFCC =================
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)

                # Delta & Delta-Delta
                delta = librosa.feature.delta(mfcc)
                delta2 = librosa.feature.delta(mfcc, order=2)

                # MFCC stats
                mfcc_mean = np.mean(mfcc, axis=1)
                mfcc_std = np.std(mfcc, axis=1)

                delta_mean = np.mean(delta, axis=1)
                delta_std = np.std(delta, axis=1)

                delta2_mean = np.mean(delta2, axis=1)
                delta2_std = np.std(delta2, axis=1)

                # ================= ADDITIONAL FEATURES =================

                # Zero Crossing Rate
                zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

                # Chroma
                chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
                chroma_mean = np.mean(chroma, axis=1)

                # Spectral Centroid
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))

                # Energy
                energy = np.mean(audio ** 2)

                # ================= FINAL FEATURE VECTOR =================
                feature_vector = np.hstack((
                    mfcc_mean, mfcc_std,
                    delta_mean, delta_std,
                    delta2_mean, delta2_std,
                    chroma_mean,
                    [zcr, spectral_centroid, energy]
                ))

                features.append(feature_vector)
                labels.append(label)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return np.array(features), np.array(labels), label_map


# ================= MAIN PIPELINE =================
if __name__ == "__main__":

    # Step 1: Extract features
    X, y, label_map = extract_features()

    print("Feature shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Label map:", label_map)

    # Step 2: Save dataset (CSV)
    df = pd.DataFrame(X)
    df['label'] = y
    df.to_csv("voice_features.csv", index=False)

    print("✅ Features saved to voice_features.csv")

    # Step 3: Save label map
    with open("label_map.json", "w") as f:
        json.dump(label_map, f)

    print("✅ Label map saved to label_map.json")