from pesq import pesq
from scipy.io import wavfile

# Load original and compressed audio
sr1, ref = wavfile.read("48Hz_3.wav")
sr2, deg = wavfile.read("48Hz_3.2.wav")

# Both must be same sampling rate (8 or 16 kHz)
score = pesq(sr1, ref, deg, 'wb')  # 'wb' = wideband, 'nb' = narrowband
print("PESQ Score:", score)
