import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load audio
y, sr = librosa.load('input.wav')

# Perform Short-Time Fourier Transform (STFT)
D = np.abs(librosa.stft(y))

# Convert to decibels
DB = librosa.amplitude_to_db(D, ref=np.max)

# Plot frequency distribution (Spectrogram)
plt.figure(figsize=(12, 6))
librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format="%+2.0f dB")
plt.title("Frequency Distribution (Spectrogram)")
plt.xlabel("Time")
plt.ylabel("Frequency (Hz)")
plt.show()
