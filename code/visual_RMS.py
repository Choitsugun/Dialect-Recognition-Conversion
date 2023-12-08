from utils import*

def visualize_rms(audio_path):
    y, sr = librosa.load(audio_path, sr=16_000)
    rms_value = cal_rms(y)

    return rms_value

rms = []
args = set_args()
data = pd.read_csv(args.train_csv_path, encoding='CP932')
audio_path = "/".join(args.train_csv_path.split("/")[:-1])

for id in tqdm(data["id"]):
    rms.append(visualize_rms(audio_path + "/chunk/" + f"{id}.mp3"))

plt.hist(rms, bins=50, edgecolor="k", alpha=0.7)
plt.title("Distribution of RMS")
plt.xlabel("RMS Value")
plt.ylabel("Number")
plt.savefig('../save_load/visual/RMS_his.png')

plt.bar(range(len(data["id"])), rms)
plt.ylabel('RMS Value')
plt.xlabel('Audio Files')
plt.title('RMS Values of Audio Files')
plt.tight_layout()
plt.savefig('../save_load/visual/RMS_bar.png')