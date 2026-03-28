import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import librosa
import os
import time

fs = 16000
seconds = 4
samples = 40

dataset_folder = "voice_dataset"
os.makedirs(dataset_folder, exist_ok=True)

print("Speak clearly and naturally\n")

while True:

    name = input("\nEnter student name (or type exit): ")

    if name.lower() == "exit":
        break

    student_folder = os.path.join(dataset_folder, name)
    os.makedirs(student_folder, exist_ok=True)

    print(f"\nRecording {samples} samples for {name}")

    for i in range(samples):

        print(f"\nSample {i+1}")
        print("Speak after countdown...")

        for c in range(3, 0, -1):
            print(c)
            time.sleep(1)

        print("Recording...")

        recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()

        # Convert to 1D array
        audio = recording.flatten()

        # 🔥 Normalize
        if np.max(np.abs(audio)) != 0:
            audio = audio / np.max(np.abs(audio))

        # 🔥 Trim silence
        audio, _ = librosa.effects.trim(audio)

        # 🔥 Fix length (important for consistency)
        max_len = fs * 3  # 3 seconds

        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)))
        else:
            audio = audio[:max_len]

        filename = os.path.join(student_folder, f"{name}_{i+1}.wav")

        # Convert back to int16 for saving
        write(filename, fs, (audio * 32767).astype(np.int16))

        print("Saved:", filename)

print("\nDataset collection finished.")