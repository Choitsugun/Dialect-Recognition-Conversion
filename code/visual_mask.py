from utils import*

y, sr = librosa.load("../dataset/chunk/chunk_158.mp3", sr=16_000)
y_tm = time_masking(y, T=100)
S = librosa.feature.melspectrogram(y=y_tm, sr=sr, n_fft=512, hop_length=128, n_mels=128)
S_fm = frequency_masking(S, F=10, nu=128)
reconstructed_audio = librosa.feature.inverse.mel_to_audio(S_fm, sr=sr, n_fft=512, hop_length=128, n_iter=32)
#sf.write('../save_load/visual/output_path.mp3', reconstructed_audio, sr)

plt.figure(figsize=(10, 4))
log_S = librosa.power_to_db(S_fm, ref=np.max)
plt.imshow(log_S, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar(format='%+2.0f dB')
plt.ylabel('Mel Bands')
plt.xlabel('Time (sec)')
plt.title('Log Mel Spectrogram')
plt.tight_layout()
plt.savefig('../save_load/visual/Masked_spec.png')