from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoModelForCausalLM, AutoTokenizer, pipeline
from torch.utils.data import DataLoader, Dataset
from hyperparams import set_args
import torch.nn.functional as F
import matplotlib.pyplot as plt
import soundfile as sf
from tqdm import tqdm
import pandas as pd
import transformers
import numpy as np
import logging
import librosa
import codecs
import torch
import sys
import os

def device_assign(args, logger):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    torch.multiprocessing.set_start_method("spawn")
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device:{}".format(args.device))


def to_device(device, logger, *params):
    if len(params) == 1:
        logger.info("One model is discovered")
        model = params[0]
        model.to(device)
        logger.info("Using {} to train/eval it".format(device))

        return model

    elif len(params) == 2:
        logger.info("Two models are discovered")
        model1, model2 = params
        model1.to(device)
        model2.to(device)
        logger.info("Using {} to train/eval them".format(device))

        return model1, model2

    else:
        logger.info("Invalid number of models, please check the argument")
        sys.exit()


def create_logger(args):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def batch_buil_base(batch_data):
    text, speech = batch_data
    inputs = speech
    inputs["labels"] = torch.where(text.attention_mask==1, text.input_ids, -100)

    return inputs


def batch_buil_frsq(batch_data):
    return batch_buil_base(batch_data)


def batch_buil_desq(batch_data, step):
    if step is "wav":
        Tokyo, Toyama, inputs = batch_data
        return Tokyo, Toyama, inputs

    if step is "llm":
        return batch_buil_sqsq(batch_data)


def batch_buil_sqsq(batch_data):
    def make_mask(inputs, target_value, point=1):
        batch_size, max_length = inputs.shape
        bl = (inputs == target_value)
        nonzero = torch.nonzero(bl)
        count = torch.count_nonzero(bl, dim=1)
        cumsum = torch.cumsum(count, dim=0)
        indices = cumsum - point
        l_rep = nonzero[:, 1][indices].view(-1, 1).repeat(1, max_length)
        mask = torch.arange(max_length).repeat(batch_size, 1).to(inputs.device)

        return l_rep < mask

    inputs = batch_data
    mask = make_mask(inputs.input_ids, 272)  # 272 -> :
    inputs["labels"] = torch.where(mask & (inputs["attention_mask"]==1), inputs["input_ids"], -100)

    return inputs


def cal_rms(y):
    rms_value = np.sqrt(np.mean(y**2))

    return rms_value


def time_masking(y, T=100):
    t = np.random.randint(0, T)
    t0 = np.random.randint(0, len(y)-t)
    y[t0:t0+t] = 0

    return y


def frequency_masking(mel_spectrogram, F=20, nu=128):
    f = np.random.randint(0, F)
    f0 = np.random.randint(0, nu-f)
    mel_spectrogram[f0:f0+f, :] = 0

    return mel_spectrogram


def apply_augment(y, kwargs):
    if np.random.rand() < 0.2:
        if kwargs["a_code"] == "both":
            # time mask
            y_tm = time_masking(y, T=kwargs["num_frame"])

            # frequency_masking
            S = librosa.feature.melspectrogram\
            (y=y_tm, sr=16_000, n_fft=kwargs["n_fft"], hop_length=kwargs["hop_length"], n_mels=kwargs["n_mels"])

            S_fm = frequency_masking(S, F=kwargs["num_frequ"], nu=kwargs["n_mels"])

            y = librosa.feature.inverse.mel_to_audio\
            (M=S_fm, sr=16_000, n_fft=kwargs["n_fft"], hop_length=kwargs["hop_length"], n_iter=kwargs["n_iter"])

        elif kwargs["a_code"] == "eith":
            if np.random.rand() < 0.5:
                # time mask
                y = time_masking(y, T=kwargs["num_frame"])

            else:
                # frequency_masking
                S = librosa.feature.melspectrogram\
                (y=y, sr=16_000, n_fft=kwargs["n_fft"], hop_length=kwargs["hop_length"], n_mels=kwargs["n_mels"])

                S_fm = frequency_masking(S, F=kwargs["num_frequ"], nu=kwargs["n_mels"])

                y = librosa.feature.inverse.mel_to_audio\
                (M=S_fm, sr=16_000, n_fft=kwargs["n_fft"], hop_length=kwargs["hop_length"], n_iter=kwargs["n_iter"])

        elif kwargs["a_code"] == "time":
            # time mask
            y = time_masking(y, T=kwargs["num_frame"])

        elif kwargs["a_code"] == "freq":
            # frequency_masking
            S = librosa.feature.melspectrogram\
            (y=y, sr=16_000, n_fft=kwargs["n_fft"], hop_length=kwargs["hop_length"], n_mels=kwargs["n_mels"])

            S_fm = frequency_masking(S, F=kwargs["num_frequ"], nu=kwargs["n_mels"])

            y = librosa.feature.inverse.mel_to_audio\
            (M=S_fm, sr=16_000, n_fft=kwargs["n_fft"], hop_length=kwargs["hop_length"], n_iter=kwargs["n_iter"])

        else:
            print("Please check if a_code is set correctly!!")
            sys.exit()

    return y


def make_kwargs(args):
    kwargs = {"num_frame": args.num_frame, "num_frequ": args.num_frequ, "hop_length": args.hop_length,
              "n_fft": args.n_fft, "n_mels": args.n_mels, "n_iter": args.n_iter, "a_code": args.a_code,
              "d_rms": args.d_rms, "time_scale": args.time_scale}

    return kwargs


def param_freeze(model):
    for param in model.parameters():
        param.requires_grad = False

    return model


def time_stretch(y, kwargs):
    if np.random.rand() < 0.2:
        rate = np.random.uniform(1, kwargs["time_scale"])
        y = librosa.effects.time_stretch(y, n_fft=kwargs["n_fft"], hop_length=kwargs["hop_length"], rate=rate)
        c_rms = cal_rms(y)
        scal = kwargs["d_rms"] / c_rms
        y = y * scal

    return y