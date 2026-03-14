import sounddevice as sd
from scipy.io.wavfile import write
import os
import time

fs = 16000
seconds = 4
samples = 8

dataset_folder = "voice_dataset"
os.makedirs(dataset_folder, exist_ok=True)

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

        for c in range(3,0,-1):
            print(c)
            time.sleep(1)

        print("Recording...")

        recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()

        filename = os.path.join(student_folder, f"{name}_{i+1}.wav")
        write(filename, fs, recording)

        print("Saved:", filename)

print("\nDataset collection finished.")