from utils import*

class Collater_base():
    def __init__(self, processor, device):
        self.processor = processor
        self.device = device

    def __call__(self, batch):
        Tokyo, speech = process_base(batch, self.processor, self.device)
        return Tokyo, speech


class Collater_frsq():
    def __init__(self, processor, device, **kwargs):
        self.processor = processor
        self.device = device
        self.kwargs = kwargs

    def __call__(self, batch):
        Toyama, speech = process_frsq(batch, self.processor, self.device, self.kwargs)
        return Toyama, speech


class Collater_desq():
    def __init__(self, processor, device, **kwargs):
        self.processor = processor
        self.device = device
        self.kwargs = kwargs

    def __call__(self, batch):
        Tokyo, Toyama, speech = process_desq(batch, self.processor, self.device, self.kwargs)
        return Tokyo, Toyama, speech


class Collater_sqsq():
    def __init__(self, processor, device):
        self.processor = processor
        self.device = device

    def __call__(self, batch):
        pair = process_sqsq(batch, self.processor, self.device)
        return pair


class Dataset_base(Dataset):
    def __init__(self, data):
        self.Tokyo = data["Tokyo"]
        self.speech = data["speech"]

    def __getitem__(self, index):
        Tokyo = self.Tokyo[index]
        speech = self.speech[index]

        return Tokyo, speech

    def __len__(self):
        return len(self.speech)


class Dataset_frsq(Dataset):
    def __init__(self, data):
        self.Toyama = data["Toyama"]
        self.speech = data["speech"]

    def __getitem__(self, index):
        Toyama = self.Toyama[index]
        speech = self.speech[index]

        return Toyama, speech

    def __len__(self):
        return len(self.speech)


class Dataset_desq(Dataset):
    def __init__(self, data):
        self.Tokyo = data["Tokyo"]
        self.Toyama = data["Toyama"]
        self.speech = data["speech"]

    def __getitem__(self, index):
        Tokyo = self.Tokyo[index]
        Toyama = self.Toyama[index]
        speech = self.speech[index]

        return Tokyo, Toyama, speech

    def __len__(self):
        return len(self.speech)


class Dataset_sqsq(Dataset):
    def __init__(self, pair):
        self.pair = pair

    def __getitem__(self, index):
        pair = self.pair[index]
        return pair

    def __len__(self):
        return len(self.pair)


def load_train_data(m_code, csv_path, num_valid, d_rms, logger):
    data = pd.read_csv(csv_path, encoding='CP932')

    if m_code in ["base", "frsq", "desq"]:
        logger.info("Starting speech array process")
        audio_path = "/".join(csv_path.split("/")[:-1])
        speeches = []

        for id in tqdm(data["id"]):
            speech_array, sampling_rate = librosa.load(audio_path+"/chunk/"+f"{id}.mp3", sr=16_000)

            if d_rms == 0:
                speeches.append(speech_array)
            else:
                c_rms = cal_rms(speech_array)
                scal = d_rms / c_rms
                speeches.append(speech_array*scal)

        data["speech"] = speeches
        train_dataset = globals().get("Dataset_" + m_code)(data[num_valid:].reset_index(drop=True))

        if num_valid != 0:
            valid_dataset = globals().get("Dataset_" + m_code)(data[:num_valid].reset_index(drop=True))
            logger.info("Using {} samples for validation".format(num_valid))
        else:
            valid_dataset = None
            logger.info("Not validation")

    elif m_code is "sqsq":
        pair = []
        logger.info("Starting prompt creation")

        for i in range(len(data["id"])):
            input_text = "「{}」を標準語に変換してください。".format(data["Toyama"][i].strip())
            prompt = f"ユーザー: {input_text}\nシステム: "
            pair.append(prompt + data["Tokyo"][i].strip())

        train_dataset = Dataset_sqsq(pair[num_valid:])

        if num_valid != 0:
            valid_dataset = Dataset_sqsq(pair[:num_valid])
            logger.info("Using {} samples for validation".format(num_valid))
        else:
            valid_dataset = None
            logger.info("Not validation")

    else:
        logger.info("Please check if m_code is set correctly!!")
        sys.exit()

    logger.info("Training Dataset format process are finshed")
    return train_dataset, valid_dataset


def load_test_data(m_code, csv_path, d_rms, logger):
    data = pd.read_csv(csv_path, encoding='CP932')

    if m_code in ["base", "dtd1", "dtd2", "rule"]:
        logger.info("Starting speech array process")
        audio_path = "/".join(csv_path.split("/")[:-1])
        speeches = []

        for id in tqdm(data["id"]):
            speech_array, sampling_rate = librosa.load(audio_path+"/chunk/"+f"{id}.mp3", sr=16_000)

            if d_rms == 0:
                speeches.append(speech_array)
            else:
                c_rms = cal_rms(speech_array)
                scal = d_rms / c_rms
                speeches.append(speech_array*scal)

        data["speech"] = speeches
        test_dataset = data

        """
    elif m_code is "sqsq":
        pair = []
        logger.info("Starting prompt creation")

        for i in range(len(data["id"])):
            input_text = "「{}」を標準語に変換してください。".format(data["Toyama"][i].strip())
            prompt = f"ユーザー: {input_text}\nシステム:"
            pair.append(prompt)

        data["Toyama"] = pair
        test_dataset = data
        """

    else:
        logger.info("Please check if m_code is set correctly!!")
        sys.exit()

    logger.info("Testing dataset format process are finshed")
    return test_dataset


def format_desp(Tokyo, text, tokenizer, device):
    pair = []

    for index, t in enumerate(text):
        input_text = "「{}」を標準語に変換してください。".format(t.strip())
        prompt = f"ユーザー: {input_text}\nシステム: "
        pair.append(prompt + Tokyo[index].strip())

    pair = tokenizer(pair, padding=True, return_tensors="pt").to(device)

    return pair


def process_base(batch, processor, device):
    Tokyo, speech = zip(*batch)
    Tokyo = processor(text=Tokyo, return_tensors="pt", padding=True).to(device)
    speech = processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True).to(device)

    return Tokyo, speech


def process_frsq(batch, processor, device, kwargs):
    Toyama, speech = zip(*batch)
    Toyama = processor(text=Toyama, return_tensors="pt", padding=True).to(device)

    if kwargs:
        speech = [apply_augment(y.copy(), kwargs) for y in speech]

    speech = processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True).to(device)

    return Toyama, speech


def process_desq(batch, processor, device, kwargs):
    Tokyo, Toyama, speech = zip(*batch)

    speech = [time_stretch(y.copy(), kwargs) for y in speech]
    speech = processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True).to(device)

    return Tokyo, Toyama, speech


def process_sqsq(batch, processor, device):
    pair = batch
    pair = processor(pair, padding=True, return_tensors="pt").to(device)

    return pair