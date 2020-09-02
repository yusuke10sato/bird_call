"""
usage:
     python3 train.py settings/test_baseline_1epoch.yaml
"""

import gc
import os
import random
import shutil
import time
import typing as tp
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import List

import numpy as np
import pytorch_pfn_extras as ppe
import resnest.torch as resnest_torch
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import yaml
from joblib import Parallel, delayed
from pytorch_pfn_extras.training import extensions as ppe_extensions

import audioread
import click
import cv2
import librosa
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import cv2
import audioread
import logging
import os
import random
import time
import warnings

import librosa
import librosa.display as display
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from contextlib import contextmanager
from IPython.display import Audio
from pathlib import Path
from typing import Optional, List

from catalyst.dl import SupervisedRunner, State, CallbackOrder, Callback, CheckpointCallback
from fastprogress import progress_bar
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, average_precision_score


from aug import noise, shifting_time, speed, pitch

BIRD_CODE = {
    'aldfly': 0, 'ameavo': 1, 'amebit': 2, 'amecro': 3, 'amegfi': 4,
    'amekes': 5, 'amepip': 6, 'amered': 7, 'amerob': 8, 'amewig': 9,
    'amewoo': 10, 'amtspa': 11, 'annhum': 12, 'astfly': 13, 'baisan': 14,
    'baleag': 15, 'balori': 16, 'banswa': 17, 'barswa': 18, 'bawwar': 19,
    'belkin1': 20, 'belspa2': 21, 'bewwre': 22, 'bkbcuc': 23, 'bkbmag1': 24,
    'bkbwar': 25, 'bkcchi': 26, 'bkchum': 27, 'bkhgro': 28, 'bkpwar': 29,
    'bktspa': 30, 'blkpho': 31, 'blugrb1': 32, 'blujay': 33, 'bnhcow': 34,
    'boboli': 35, 'bongul': 36, 'brdowl': 37, 'brebla': 38, 'brespa': 39,
    'brncre': 40, 'brnthr': 41, 'brthum': 42, 'brwhaw': 43, 'btbwar': 44,
    'btnwar': 45, 'btywar': 46, 'buffle': 47, 'buggna': 48, 'buhvir': 49,
    'bulori': 50, 'bushti': 51, 'buwtea': 52, 'buwwar': 53, 'cacwre': 54,
    'calgul': 55, 'calqua': 56, 'camwar': 57, 'cangoo': 58, 'canwar': 59,
    'canwre': 60, 'carwre': 61, 'casfin': 62, 'caster1': 63, 'casvir': 64,
    'cedwax': 65, 'chispa': 66, 'chiswi': 67, 'chswar': 68, 'chukar': 69,
    'clanut': 70, 'cliswa': 71, 'comgol': 72, 'comgra': 73, 'comloo': 74,
    'commer': 75, 'comnig': 76, 'comrav': 77, 'comred': 78, 'comter': 79,
    'comyel': 80, 'coohaw': 81, 'coshum': 82, 'cowscj1': 83, 'daejun': 84,
    'doccor': 85, 'dowwoo': 86, 'dusfly': 87, 'eargre': 88, 'easblu': 89,
    'easkin': 90, 'easmea': 91, 'easpho': 92, 'eastow': 93, 'eawpew': 94,
    'eucdov': 95, 'eursta': 96, 'evegro': 97, 'fiespa': 98, 'fiscro': 99,
    'foxspa': 100, 'gadwal': 101, 'gcrfin': 102, 'gnttow': 103, 'gnwtea': 104,
    'gockin': 105, 'gocspa': 106, 'goleag': 107, 'grbher3': 108, 'grcfly': 109,
    'greegr': 110, 'greroa': 111, 'greyel': 112, 'grhowl': 113, 'grnher': 114,
    'grtgra': 115, 'grycat': 116, 'gryfly': 117, 'haiwoo': 118, 'hamfly': 119,
    'hergul': 120, 'herthr': 121, 'hoomer': 122, 'hoowar': 123, 'horgre': 124,
    'horlar': 125, 'houfin': 126, 'houspa': 127, 'houwre': 128, 'indbun': 129,
    'juntit1': 130, 'killde': 131, 'labwoo': 132, 'larspa': 133, 'lazbun': 134,
    'leabit': 135, 'leafly': 136, 'leasan': 137, 'lecthr': 138, 'lesgol': 139,
    'lesnig': 140, 'lesyel': 141, 'lewwoo': 142, 'linspa': 143, 'lobcur': 144,
    'lobdow': 145, 'logshr': 146, 'lotduc': 147, 'louwat': 148, 'macwar': 149,
    'magwar': 150, 'mallar3': 151, 'marwre': 152, 'merlin': 153, 'moublu': 154,
    'mouchi': 155, 'moudov': 156, 'norcar': 157, 'norfli': 158, 'norhar2': 159,
    'normoc': 160, 'norpar': 161, 'norpin': 162, 'norsho': 163, 'norwat': 164,
    'nrwswa': 165, 'nutwoo': 166, 'olsfly': 167, 'orcwar': 168, 'osprey': 169,
    'ovenbi1': 170, 'palwar': 171, 'pasfly': 172, 'pecsan': 173, 'perfal': 174,
    'phaino': 175, 'pibgre': 176, 'pilwoo': 177, 'pingro': 178, 'pinjay': 179,
    'pinsis': 180, 'pinwar': 181, 'plsvir': 182, 'prawar': 183, 'purfin': 184,
    'pygnut': 185, 'rebmer': 186, 'rebnut': 187, 'rebsap': 188, 'rebwoo': 189,
    'redcro': 190, 'redhea': 191, 'reevir1': 192, 'renpha': 193, 'reshaw': 194,
    'rethaw': 195, 'rewbla': 196, 'ribgul': 197, 'rinduc': 198, 'robgro': 199,
    'rocpig': 200, 'rocwre': 201, 'rthhum': 202, 'ruckin': 203, 'rudduc': 204,
    'rufgro': 205, 'rufhum': 206, 'rusbla': 207, 'sagspa1': 208, 'sagthr': 209,
    'savspa': 210, 'saypho': 211, 'scatan': 212, 'scoori': 213, 'semplo': 214,
    'semsan': 215, 'sheowl': 216, 'shshaw': 217, 'snobun': 218, 'snogoo': 219,
    'solsan': 220, 'sonspa': 221, 'sora': 222, 'sposan': 223, 'spotow': 224,
    'stejay': 225, 'swahaw': 226, 'swaspa': 227, 'swathr': 228, 'treswa': 229,
    'truswa': 230, 'tuftit': 231, 'tunswa': 232, 'veery': 233, 'vesspa': 234,
    'vigswa': 235, 'warvir': 236, 'wesblu': 237, 'wesgre': 238, 'weskin': 239,
    'wesmea': 240, 'wessan': 241, 'westan': 242, 'wewpew': 243, 'whbnut': 244,
    'whcspa': 245, 'whfibi': 246, 'whtspa': 247, 'whtswi': 248, 'wilfly': 249,
    'wilsni1': 250, 'wiltur': 251, 'winwre3': 252, 'wlswar': 253, 'wooduc': 254,
    'wooscj2': 255, 'woothr': 256, 'y00475': 257, 'yebfly': 258, 'yebsap': 259,
    'yehbla': 260, 'yelwar': 261, 'yerwar': 262, 'yetvir': 263
}
INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    #torch.backends.cudnn.deterministic = True  # type: ignore
    #torch.backends.cudnn.benchmark = True  # type: ignore
    
@contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None) -> None:
    """Timer Util"""
    t0 = time.time()
    msg = f"[{name}] start"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)
    yield
    msg = f"[{name}] done in {time.time() - t0:.2f} s"
    if logger is None:
        print(msg)
    else:
        logger.info(msg)

class DFTBase(nn.Module):
    def __init__(self):
        """Base class for DFT and IDFT matrix"""
        super(DFTBase, self).__init__()

    def dft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(-2 * np.pi * 1j / n)
        W = np.power(omega, x * y)
        return W

    def idft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(2 * np.pi * 1j / n)
        W = np.power(omega, x * y)
        return W

class STFT(DFTBase):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None, 
        window='hann', center=True, pad_mode='reflect', freeze_parameters=True):
        """Implementation of STFT with Conv1d. The function has the same output 
        of librosa.core.stft
        """
        super(STFT, self).__init__()

        assert pad_mode in ['constant', 'reflect']

        self.n_fft = n_fft
        self.center = center
        self.pad_mode = pad_mode

        # By default, use the entire frame
        if win_length is None:
            win_length = n_fft

        # Set the default hop, if it's not already specified
        if hop_length is None:
            hop_length = int(win_length // 4)

        fft_window = librosa.filters.get_window(window, win_length, fftbins=True)

        # Pad the window out to n_fft size
        fft_window = librosa.util.pad_center(fft_window, n_fft)

        # DFT & IDFT matrix
        self.W = self.dft_matrix(n_fft)

        out_channels = n_fft // 2 + 1

        self.conv_real = nn.Conv1d(in_channels=1, out_channels=out_channels, 
            kernel_size=n_fft, stride=hop_length, padding=0, dilation=1, 
            groups=1, bias=False)

        self.conv_imag = nn.Conv1d(in_channels=1, out_channels=out_channels, 
            kernel_size=n_fft, stride=hop_length, padding=0, dilation=1, 
            groups=1, bias=False)

        self.conv_real.weight.data = torch.Tensor(
            np.real(self.W[:, 0 : out_channels] * fft_window[:, None]).T)[:, None, :]
        # (n_fft // 2 + 1, 1, n_fft)

        self.conv_imag.weight.data = torch.Tensor(
            np.imag(self.W[:, 0 : out_channels] * fft_window[:, None]).T)[:, None, :]
        # (n_fft // 2 + 1, 1, n_fft)

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        """input: (batch_size, data_length)
        Returns:
          real: (batch_size, n_fft // 2 + 1, time_steps)
          imag: (batch_size, n_fft // 2 + 1, time_steps)
        """

        x = input[:, None, :]   # (batch_size, channels_num, data_length)

        if self.center:
            x = F.pad(x, pad=(self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)

        real = self.conv_real(x)
        imag = self.conv_imag(x)
        # (batch_size, n_fft // 2 + 1, time_steps)

        real = real[:, None, :, :].transpose(2, 3)
        imag = imag[:, None, :, :].transpose(2, 3)
        # (batch_size, 1, time_steps, n_fft // 2 + 1)

        return real, imag

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x

class AttBlock(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear",
                 temperature=1.0):
        super().__init__()

        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.bn_att = nn.BatchNorm1d(out_features)
        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)

class PANNsCNN14Att(nn.Module):
    def __init__(self, sample_rate: int, window_size: int, hop_size: int,
                 mel_bins: int, fmin: int, fmax: int, classes_num: int):
        super().__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.interpolate_ratio = 32  # Downsampled ratio

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.att_block = AttBlock(2048, classes_num, activation='sigmoid')

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        
    def cnn_feature_extractor(self, x):
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        return x
    
    def preprocess(self, input, mixup_lambda=None):
        # t1 = time.time()
        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        return x, frames_num
        

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""
        x, frames_num = self.preprocess(input, mixup_lambda=mixup_lambda)

        # Output shape (batch size, channels, time, frequency)
        x = self.cnn_feature_extractor(x)
        
        # Aggregate in frequency axis
        x = torch.mean(x, dim=3)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        # Get framewise output
        framewise_output = interpolate(segmentwise_output,
                                       self.interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        output_dict = {
            'framewise_output': framewise_output,
            'clipwise_output': clipwise_output
        }

        return output_dict

class mAPCallback(Callback):
    def __init__(self,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 model_output_key: str = "clipwise_output",
                 prefix: str = "mAP"):
        super().__init__(CallbackOrder.Metric)
        self.input_key = input_key
        self.output_key = output_key
        self.model_output_key = model_output_key
        self.prefix = prefix

    def on_loader_start(self, state: State):
        self.prediction: List[np.ndarray] = []
        self.target: List[np.ndarray] = []

    def on_batch_end(self, state: State):
        targ = state.input[self.input_key].detach().cpu().numpy()
        out = state.output[self.output_key]

        clipwise_output = out[self.model_output_key].detach().cpu().numpy()

        self.prediction.append(clipwise_output)
        self.target.append(targ)

        score = average_precision_score(targ, clipwise_output, average=None)
        score = np.nan_to_num(score).mean()
        state.batch_metrics[self.prefix] = score

    def on_loader_end(self, state: State):
        y_pred = np.concatenate(self.prediction, axis=0)
        y_true = np.concatenate(self.target, axis=0)
        score = average_precision_score(y_true, y_pred, average=None)
        score = np.nan_to_num(score).mean()
        state.loader_metrics[self.prefix] = score
        if state.is_valid_loader:
            state.epoch_metrics[state.valid_loader + "_epoch_" +
                                self.prefix] = score
        else:
            state.epoch_metrics["train_epoch_" + self.prefix] = score

# Criterion
class PANNsLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = nn.BCELoss()

    def forward(self, input, target):
        input_ = input["clipwise_output"]
        input_ = torch.where(torch.isnan(input_),
                             torch.zeros_like(input_),
                             input_)
        input_ = torch.where(torch.isinf(input_),
                             torch.zeros_like(input_),
                             input_)

        target = target.float()

        return self.bce(input_, target)

@click.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.option('--short_mode', is_flag=True)
def train(config_path, short_mode):
    f = open(config_path, 'r')
    settings = yaml.safe_load(f)

    ROOT = Path(settings['ROOT'])
    INPUT_ROOT = ROOT / settings['INPUT_ROOT']
    RAW_DATA = INPUT_ROOT / settings['RAW_DATA']
    TRAIN_AUDIO_DIR = RAW_DATA / settings['TRAIN_AUDIO_DIR']
    TRAIN_RESAMPLED_AUDIO_DIRS = RAW_DATA / settings['TRAIN_RESAMPLED_AUDIO_DIRS']
    TEST_AUDIO_DIR = RAW_DATA / settings['TEST_AUDIO_DIR']
    
    if not TEST_AUDIO_DIR.exists():
        TEST_AUDIO_DIR = INPUT_ROOT / "birdcall-check" / "test_audio"
        test = pd.read_csv(INPUT_ROOT / "birdcall-check" / "test.csv")
    else:
        test = pd.read_csv(RAW_DATA / "test.csv")

    train = pd.read_csv(TRAIN_RESAMPLED_AUDIO_DIRS / "train_mod.csv")

    BIRD_CODE = settings['BIRD_CODE']

    print('... train_start')
    start_time = time.time()
    
    tmp_list = []
    for ebird_d in TRAIN_RESAMPLED_AUDIO_DIRS.iterdir():
        if ebird_d.is_file():
            continue
        for wav_f in ebird_d.iterdir():
            tmp_list.append([ebird_d.name, wav_f.name, wav_f.as_posix()])
    train_wav_path_exist = pd.DataFrame(
        tmp_list, columns=["ebird_code", "resampled_filename", "file_path"])

    del tmp_list
    train_all = pd.merge(
        train, train_wav_path_exist, on=["ebird_code", "resampled_filename"], how="inner")

    skf = StratifiedKFold(**settings["split"]["params"])
    train_all["fold"] = -1
    for fold_id, (train_index, val_index) in enumerate(skf.split(train_all, train_all["ebird_code"])):
        train_all.iloc[val_index, -1] = fold_id
    
    use_fold = settings["globals"]["use_fold"]
    train_file_list = train_all.query("fold != @use_fold")[["file_path", "ebird_code"]].values.tolist()
    val_file_list = train_all.query("fold == @use_fold")[["file_path", "ebird_code"]].values.tolist()
    if short_mode:
        train_file_list = train_file_list[:100]
        val_file_list = train_file_list[:100]
    print("[fold {}] train: {}, val: {}".format(use_fold, len(train_file_list), len(val_file_list)))

    set_seed(settings["globals"]["seed"])
    device = torch.device(settings["globals"]["device"])
    output_dir = Path(settings["globals"]["output_dir"])

    # get loader
    train_loader, val_loader = get_loaders_for_training(
        settings["dataset"]["params"], settings["loader"], 
        train_file_list, val_file_list, BIRD_CODE)
    # get model
    melspectrogram_parameters = settings["dataset"]["params"]["melspectrogram_parameters"]
    model_config = {
        "sample_rate": settings["globals"]["target_sr"],
        "window_size": melspectrogram_parameters["window_size"],
        "hop_size": melspectrogram_parameters["hop_size"],
        "mel_bins": melspectrogram_parameters["n_mels"],
        "fmin": melspectrogram_parameters["fmin"],
        "fmax": melspectrogram_parameters["fmax"],
        "classes_num": settings["model"]["params"]["n_classes"]
    }

    # loaders
    loaders = {
        "train": train_loader,
        "valid": val_loader
    }

    # model
    model_config["classes_num"] = 527
    model = PANNsCNN14Att(**model_config)
    weights = torch.load(INPUT_ROOT / "pannscnn14-decisionlevelatt-weight/Cnn14_DecisionLevelAtt_mAP0.425.pth")
    # Fixed in V3   
    model.load_state_dict(weights["model"])
    model.att_block = AttBlock(2048, settings["model"]["params"]["n_classes"], activation='sigmoid')
    model.att_block.init_weights()
    model.to(device)
    model = model.to(device)
    # get optimizer
    optimizer = getattr(
        torch.optim, settings["optimizer"]["name"]
        )(model.parameters(), **settings["optimizer"]["params"])
    # get scheduler
    scheduler = getattr(
        torch.optim.lr_scheduler, settings["scheduler"]["name"]
        )(optimizer, **settings["scheduler"]["params"])
    # get loss
    loss_func = PANNsLoss().to(device)#getattr(nn, settings["loss"]["name"])(**settings["loss"]["params"])
    # Loss
    #criterion = PANNsLoss().to(device)

    # callbacks
    callbacks = [
        F1Callback(input_key="targets", output_key="logits", prefix="f1"),
        mAPCallback(input_key="targets", output_key="logits", prefix="mAP"),
        CheckpointCallback(save_n_best=0)
    ]
    
    # eval func
    eval_func_dict = {
        'f1_score': f1_loss
    }

    warnings.simplefilter("ignore")

    # # check the propotion
    fold_proportion = pd.pivot_table(train_all, index="ebird_code", columns="fold", values="xc_id", aggfunc=len)
    print(fold_proportion.shape)



    # create training manager
    trigger = None
    manager = ppe.training.ExtensionsManager(
        model, optimizer, settings["globals"]["num_epochs"],
        iters_per_epoch=len(train_loader),
        stop_trigger=trigger,
        out_dir=output_dir
        )
    # set manager extensions
    manager = set_extensions(
        manager, settings, model, device,
        loaders["valid"], optimizer, loss_func, eval_func_dict
        )

    train_loop(
        manager, settings, model, device,
        loaders["train"], optimizer, scheduler, loss_func)

    log = pd.read_json(output_dir / "log")
    best_epoch = log["val/loss"].idxmin() + 1
    print('... best epoch')
    print(log.iloc[[best_epoch - 1],])

    shutil.copy(output_dir / "snapshot_epoch_{}.pth".format(best_epoch), output_dir / "best_model.pth")
    
    model_config["classes_num"] = settings["model"]["params"]["n_classes"]
    m = get_model(model_config, output_dir / 'best_model.pth')
    state_dict = torch.load(output_dir / 'best_model.pth')
    print(m.load_state_dict(state_dict))

    # f1ベスト書き出し
    best_epoch = log["val/f1_score"].idxmax() + 1
    print('... best epoch')
    print(log.iloc[[best_epoch - 1],])

    shutil.copy(output_dir / "snapshot_epoch_f1_{}.pth".format(best_epoch), output_dir / "f1_best_model.pth")

    print('... all well done')
    end_time = time.time()
    print('... elapsed time : {} minutes'.format((end_time - start_time)/60))

# Callbacks
class F1Callback(Callback):
    def __init__(self,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 model_output_key: str = "clipwise_output",
                 prefix: str = "f1"):
        super().__init__(CallbackOrder.Metric)

        self.input_key = input_key
        self.output_key = output_key
        self.model_output_key = model_output_key
        self.prefix = prefix

    def on_loader_start(self, state: State):
        self.prediction: List[np.ndarray] = []
        self.target: List[np.ndarray] = []

    def on_batch_end(self, state: State):
        targ = state.input[self.input_key].detach().cpu().numpy()
        out = state.output[self.output_key]

        clipwise_output = out[self.model_output_key].detach().cpu().numpy()

        self.prediction.append(clipwise_output)
        self.target.append(targ)

        y_pred = clipwise_output.argmax(axis=1)
        y_true = targ.argmax(axis=1)

        score = f1_score(y_true, y_pred, average="macro")
        state.batch_metrics[self.prefix] = score

    def on_loader_end(self, state: State):
        y_pred = np.concatenate(self.prediction, axis=0).argmax(axis=1)
        y_true = np.concatenate(self.target, axis=0).argmax(axis=1)
        score = f1_score(y_true, y_pred, average="macro")
        state.loader_metrics[self.prefix] = score
        if state.is_valid_loader:
            state.epoch_metrics[state.valid_loader + "_epoch_" +
                                self.prefix] = score
        else:
            state.epoch_metrics["train_epoch_" + self.prefix] = score


PERIOD = 5


class Spectrogram(nn.Module):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None, 
        window='hann', center=True, pad_mode='reflect', power=2.0, 
        freeze_parameters=True):
        """Calculate spectrogram using pytorch. The STFT is implemented with 
        Conv1d. The function has the same output of librosa.core.stft
        """
        super(Spectrogram, self).__init__()

        self.power = power

        self.stft = STFT(n_fft=n_fft, hop_length=hop_length, 
            win_length=win_length, window=window, center=center, 
            pad_mode=pad_mode, freeze_parameters=True)

    def forward(self, input):
        """input: (batch_size, 1, time_steps, n_fft // 2 + 1)
        Returns:
          spectrogram: (batch_size, 1, time_steps, n_fft // 2 + 1)
        """

        (real, imag) = self.stft.forward(input)
        # (batch_size, n_fft // 2 + 1, time_steps)

        spectrogram = real ** 2 + imag ** 2

        if self.power == 2.0:
            pass
        else:
            spectrogram = spectrogram ** (power / 2.0)

        return spectrogram

class LogmelFilterBank(nn.Module):
    def __init__(self, sr=32000, n_fft=2048, n_mels=64, fmin=50, fmax=14000, is_log=True, 
        ref=1.0, amin=1e-10, top_db=80.0, freeze_parameters=True):
        """Calculate logmel spectrogram using pytorch. The mel filter bank is 
        the pytorch implementation of as librosa.filters.mel 
        """
        super(LogmelFilterBank, self).__init__()

        self.is_log = is_log
        self.ref = ref
        self.amin = amin
        self.top_db = top_db

        self.melW = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
            fmin=fmin, fmax=fmax).T
        # (n_fft // 2 + 1, mel_bins)

        self.melW = nn.Parameter(torch.Tensor(self.melW))

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        """input: (batch_size, channels, time_steps)
        
        Output: (batch_size, time_steps, mel_bins)
        """

        # Mel spectrogram
        mel_spectrogram = torch.matmul(input, self.melW)

        # Logmel spectrogram
        if self.is_log:
            output = self.power_to_db(mel_spectrogram)
        else:
            output = mel_spectrogram

        return output


    def power_to_db(self, input):
        """Power to db, this function is the pytorch implementation of 
        librosa.core.power_to_lb
        """
        ref_value = self.ref
        log_spec = 10.0 * torch.log10(torch.clamp(input, min=self.amin, max=np.inf))
        log_spec -= 10.0 * np.log10(np.maximum(self.amin, ref_value))

        if self.top_db is not None:
            if self.top_db < 0:
                raise ParameterError('top_db must be non-negative')
            log_spec = torch.clamp(log_spec, min=log_spec.max().item() - self.top_db, max=np.inf)

        return log_spec

class SpectrogramDataset(data.Dataset):
    def __init__(
        self,
        file_list: tp.List[tp.List[str]], bird_code: tp.Dict, train=True, img_size=224,
        waveform_transforms=None, spectrogram_transforms=None, melspectrogram_parameters={},
        aug={}
        ):
        self.file_list = file_list  # list of list: [file_path, ebird_code]
        self.bird_code = bird_code
        self.train = train
        self.img_size = img_size
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters
        self.aug = aug

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        wav_path, ebird_code = self.file_list[idx]

        y, sr = sf.read(wav_path)
        if self.train:
            if len(self.aug) > 0:
                if self.aug.get('noise', False):
                    prob = random.random()
                    if self.aug['noise_prob'] >= prob:
                        y = noise(y, 0.02)
                if self.aug.get('shifting_time', False):
                    prob = random.random()
                    if self.aug['shifting_time_prob'] >= prob:
                        y = shifting_time(y, sr, 2, 'right')
                if self.aug.get('speed', False):
                    prob = random.random()
                    if self.aug['speed_prob'] >= prob:
                        y = shifting_time(y, sr, 2, 'right')
                if self.aug.get('pitch', False):
                    prob = random.random()
                    if self.aug['pitch_prob'] >= prob:
                        y = pitch(y, sr, 2)

        if self.waveform_transforms:
            y = self.waveform_transforms(y)
        else:
            len_y = len(y)
            effective_length = sr * PERIOD
            if len_y < effective_length:
                new_y = np.zeros(effective_length, dtype=y.dtype)
                start = np.random.randint(effective_length - len_y)
                new_y[start:start + len_y] = y
                y = new_y.astype(np.float32)
            elif len_y > effective_length:
                start = np.random.randint(len_y - effective_length)
                y = y[start:start + effective_length].astype(np.float32)
            else:
                y = y.astype(np.float32)

        spec_mode = self.melspectrogram_parameters['mode']
        assert spec_mode == 'mel' or spec_mode =='linear'

        labels = np.zeros(len(self.bird_code), dtype="f")
        labels[self.bird_code[ebird_code]] = 1

        return y, labels

class SpecAugmentation(nn.Module):
    def __init__(self, time_drop_width, time_stripes_num, freq_drop_width, 
        freq_stripes_num):
        """Spec augmetation. 
        [ref] Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D. 
        and Le, Q.V., 2019. Specaugment: A simple data augmentation method 
        for automatic speech recognition. arXiv preprint arXiv:1904.08779.
        Args:
          time_drop_width: int
          time_stripes_num: int
          freq_drop_width: int
          freq_stripes_num: int
        """

        super(SpecAugmentation, self).__init__()

        self.time_dropper = DropStripes(dim=2, drop_width=time_drop_width, 
            stripes_num=time_stripes_num)

        self.freq_dropper = DropStripes(dim=3, drop_width=freq_drop_width, 
            stripes_num=freq_stripes_num)

    def forward(self, input):
        x = self.time_dropper(input)
        x = self.freq_dropper(x)
        return x


def get_loaders_for_training(
    args_dataset: tp.Dict, args_loader: tp.Dict,
    train_file_list: tp.List[str], val_file_list: tp.List[str], bird_code: tp.Dict
    ):
    # # make dataset
    train_dataset = SpectrogramDataset(train_file_list, bird_code, train=True, **args_dataset)
    val_dataset = SpectrogramDataset(val_file_list, bird_code, train=False, **args_dataset)
    # # make dataloader
    train_loader = data.DataLoader(train_dataset, **args_loader["train"])
    val_loader = data.DataLoader(val_dataset, **args_loader["val"])
    
    return train_loader, val_loader


def get_model(config: dict, weights_path: str):
    model = PANNsCNN14Att(**config)
    #checkpoint = torch.load(weights_path)
    #model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    return model

class DropStripes(nn.Module):
    def __init__(self, dim, drop_width, stripes_num):
        """Drop stripes. 
        Args:
          dim: int, dimension along which to drop
          drop_width: int, maximum width of stripes to drop
          stripes_num: int, how many stripes to drop
        """
        super(DropStripes, self).__init__()

        assert dim in [2, 3]    # dim 2: time; dim 3: frequency

        self.dim = dim
        self.drop_width = drop_width
        self.stripes_num = stripes_num

    def forward(self, input):
        """input: (batch_size, channels, time_steps, freq_bins)"""

        assert input.ndimension() == 4

        if self.training is False:
            return input

        else:
            batch_size = input.shape[0]
            total_width = input.shape[self.dim]

            for n in range(batch_size):
                self.transform_slice(input[n], total_width)

            return input

    def transform_slice(self, e, total_width):
        """e: (channels, time_steps, freq_bins)"""

        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.drop_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]

            if self.dim == 2:
                e[:, bgn : bgn + distance, :] = 0
            elif self.dim == 3:
                e[:, :, bgn : bgn + distance] = 0   

# Building blocks
def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.0)

def interpolate(x: torch.Tensor, ratio: int):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.

    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled

def pad_framewise_output(framewise_output: torch.Tensor, frames_num: int):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1:, :].repeat(
        1, frames_num - framewise_output.shape[1], 1)
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output

def train_loop(
    manager, args, model, device,
    train_loader, optimizer, scheduler, loss_func
    ):
    """Run minibatch training loop"""
    while not manager.stop_trigger:
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            with manager.run_iteration():
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = loss_func(output, target)
                ppe.reporting.report({'train/loss': loss.item()})
                loss.backward()
                optimizer.step()
        scheduler.step()  # <- call at the end of each epoch


def eval_for_batch(
    args, model, device,
    data, target, loss_func, eval_func_dict={}
    ):
    """
    Run evaliation for valid
    
    This function is applied to each batch of val loader.
    """
    model.eval()
    data, target = data.to(device), target.to(device)
    output = model(data)
    # Final result will be average of averages of the same size
    val_loss = loss_func(output, target).item()
    ppe.reporting.report({'val/loss': val_loss})
    
    for eval_name, eval_func in eval_func_dict.items():
        eval_value = eval_func(target, output["clipwise_output"])
        ppe.reporting.report({"val/{}".format(eval_name): eval_value})


def set_extensions(
    manager, args, model, device, test_loader, optimizer,
    loss_func, eval_func_dict={}
    ):
    """set extensions for PPE"""
        
    my_extensions = [
        # # observe, report
        ppe_extensions.observe_lr(optimizer=optimizer),
        # ppe_extensions.ParameterStatistics(model, prefix='model'),
        # ppe_extensions.VariableStatisticsPlot(model),
        ppe_extensions.LogReport(),
        ppe_extensions.PlotReport(['train/loss', 'val/loss'], 'epoch', filename='loss.png'),
        ppe_extensions.PlotReport(['lr',], 'epoch', filename='lr.png'),
        ppe_extensions.PrintReport([
            'epoch', 'iteration', 'lr', 'train/loss', 'val/loss', "elapsed_time", "val/f1_score"]),
        ppe_extensions.PlotReport(["val/f1_score"], 'epoch', filename='val_f1_score.png'),
        ppe_extensions.ProgressBar(update_interval=10),

        # # evaluation
        (
            ppe_extensions.Evaluator(
                test_loader, model,
                eval_func=lambda data, target:
                    eval_for_batch(args, model, device, data, target, loss_func, eval_func_dict),
                progress_bar=True),
            (1, "epoch"),
        ),
        # # save model snapshot.
        (
            ppe_extensions.snapshot(
                target=model, filename="snapshot_epoch_{.updater.epoch}.pth"),
            ppe.training.triggers.MinValueTrigger(key="val/loss", trigger=(1, 'epoch'))
        ),
        (
            ppe_extensions.snapshot(
                target=model, filename="snapshot_epoch_f1_{.updater.epoch}.pth"),
            ppe.training.triggers.MaxValueTrigger(key="val/f1_score", trigger=(1, 'epoch'))
        ),
    ]
           
    # # set extensions to manager
    for ext in my_extensions:
        if isinstance(ext, tuple):
            manager.extend(ext[0], trigger=ext[1])
        else:
            manager.extend(ext)
        
    return manager


def f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False, threshold=0.5):
    trues = []
    preds = []
    for true, pred in zip(y_true, y_pred):
        pred = torch.sigmoid(pred)
        proba = pred.detach().cpu().numpy()
        events = proba >= threshold
        labels = np.argwhere(events).reshape(-1).tolist()

        if len(labels) == 0:
            label_string = "nocall"
        else:
            labels_str_list = list(map(lambda x: INV_BIRD_CODE[x], labels))
            label_string = " ".join(labels_str_list)
        true = INV_BIRD_CODE[true.detach().cpu().numpy().argmax()]
        preds.append(label_string)
        trues.append(true)
    f1 = row_wise_micro_averaged_f1_score(trues, preds)
    return f1


def row_wise_micro_averaged_f1_score(
        y_true: List[str],
        y_pred: List[str]
) -> float:
    """
    Compute row-wise micro averaged f1 score
    Parameters
    ----------
    y_true : List[str]
        Target list of strings of a space separated birds names
    y_pred : List[str]
        Predicted list of strings of a space separated birds names
    Returns
    -------
    float
        Row-wise micro averaged F1 score
    Examples
    --------
    >>> from evaluations.kaggle_2020 import row_wise_micro_averaged_f1_score
    >>> y_true = [
    ...         'amecro',
    ...         'amecro amerob',
    ...         'nocall',
    ...     ]
    >>> y_pred = [
    ...         'amecro',
    ...         'amecro bird666',
    ...         'nocall',
    ...     ]
    >>> row_wise_micro_averaged_f1_score(y_true, y_pred)
    0.8333333333333333
    """
    n_rows = len(y_true)
    f1_score = 0.
    for true_row, predicted_row in zip(y_true, y_pred):
        f1_score += micro_f1_similarity(true_row, predicted_row) / n_rows
    return f1_score


def micro_f1_similarity(
        y_true: str,
        y_pred: str
) -> float:
    """
    Compute micro f1 similarity for 1 row
    Parameters
    ----------
    y_true : str
        True string of a space separated birds names
    y_pred : str
        Predicted string of a space separated birds names
    Returns
    -------
    float
        Micro F1 similarity
    Examples
    --------
    >>> from evaluations.kaggle_2020 import micro_f1_similarity
    >>> y_true = 'amecro amerob'
    >>> y_pred = 'amecro bird666'
    >>> micro_f1_similarity(y_true, y_pred)
    0.5
    """
    true_labels = y_true.split()
    pred_labels = y_pred.split()

    true_pos, false_pos, false_neg = 0, 0, 0

    for true_elem in true_labels:
        if true_elem in pred_labels:
            true_pos += 1
        else:
            false_neg += 1

    for pred_el in pred_labels:
        if pred_el not in true_labels:
            false_pos += 1

    f1_similarity = 2 * true_pos / (2 * true_pos + false_neg + false_pos)

    return f1_similarity

if __name__ == '__main__':
    train()
