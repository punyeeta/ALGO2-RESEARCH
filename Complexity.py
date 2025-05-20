import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load audio
y, sr = librosa.load('44Hz_1.wav')

# Perform Short-Time Fourier Transform (STFT)
D = np.abs(librosa.stft(y))

# Convert to decibels
DB = librosa.amplitude_to_db(D, ref=np.max)

# Calculate spectral centroid
spectral_centroid = librosa.feature.spectral_centroid(S=D, sr=sr)[0]
mean_centroid = np.mean(spectral_centroid)

# Calculate spectral bandwidth
spectral_bandwidth = librosa.feature.spectral_bandwidth(S=D, sr=sr)[0]
mean_bandwidth = np.mean(spectral_bandwidth)

# Calculate power spectrum
power_spectrum = np.sum(D ** 2, axis=0)

# Normalize power spectrum
normalized_power = power_spectrum / np.sum(power_spectrum)

# Calculate entropy manually
entropy = -np.sum(normalized_power * np.log2(normalized_power + 1e-10))

# ---------- STRUCTURAL COMPLEXITY CLASSIFICATION ----------

def classify_complexity(centroid, bandwidth, entropy):
    score = 0
    
    # Centroid classification
    if centroid > 2000:
        score += 1
    
    # Bandwidth classification
    if bandwidth > 3000:
        score += 1
    
    # Entropy classification
    if entropy > 5.0:
        score += 1

    if score == 3:
        return "High"
    elif score == 2:
        return "Moderate"
    else:
        return "Low"

complexity = classify_complexity(mean_centroid, mean_bandwidth, entropy)

# ---------- PRINT RESULTS ----------

print(f"Sampling Rate: {sr} Hz")
print(f"Mean Spectral Centroid: {mean_centroid:.2f} Hz")
print(f"Mean Spectral Bandwidth: {mean_bandwidth:.2f} Hz")
print(f"Spectral Entropy: {entropy:.2f}")
print(f"Structural Complexity: {complexity}")

# ---------- PLOTS ----------

# Spectrogram
plt.figure(figsize=(12, 6))
librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format="%+2.0f dB")
plt.title("Frequency Distribution (Spectrogram)")
plt.xlabel("Time")
plt.ylabel("Frequency (Hz)")
plt.show()

# Spectral centroid
plt.figure(figsize=(12, 4))
plt.plot(spectral_centroid, color='r')
plt.title("Spectral Centroid")
plt.xlabel("Time")
plt.ylabel("Frequency (Hz)")
plt.show()

# Spectral bandwidth
plt.figure(figsize=(12, 4))
plt.plot(spectral_bandwidth, color='g')
plt.title("Spectral Bandwidth")
plt.xlabel("Time")
plt.ylabel("Bandwidth (Hz)")
plt.show()