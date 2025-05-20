import librosa
import numpy as np

# Load audio
y, sr = librosa.load("input.wav")  # Ensure "input.wav" exists in the same directory or provide the full path

# Convert amplitude to decibels
S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

# dB range
db_min = np.min(S)
db_max = np.max(S)
print(f"dB range: {db_min:.2f} dB to {db_max:.2f} dB")

# Calculate the dB range
db_range = db_max - db_min

# Print the dB range
print(f"The dB range of the audio is: {db_range:.2f} dB")
