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
import librosa.core
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


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
#     torch.backends.cudnn.deterministic = True  # type: ignore
#     torch.backends.cudnn.benchmark = True  # type: ignore
    
@contextmanager
def timer(name: str) -> None:
    """Timer Util"""
    t0 = time.time()
    print("[{}] start".format(name))
    yield
    print("[{}] done in {:.0f} s".format(name, time.time() - t0))


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
    model = get_model(settings["model"])
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
    loss_func = getattr(nn, settings["loss"]["name"])(**settings["loss"]["params"])
    # eval func
    eval_func_dict = {
        'f1_score': f1_loss
    }
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
        val_loader, optimizer, loss_func, eval_func_dict
        )

    train_loop(
        manager, settings, model, device,
        train_loader, optimizer, scheduler, loss_func)

    log = pd.read_json(output_dir / "log")
    best_epoch = log["val/loss"].idxmin() + 1
    print('... best epoch')
    print(log.iloc[[best_epoch - 1],])

    shutil.copy(output_dir / "snapshot_epoch_{}.pth".format(best_epoch), output_dir / "best_model.pth")

    m = get_model({
    'name': settings["model"]["name"],
    'params': {'pretrained': False, 'n_classes': 264}})
    state_dict = torch.load(output_dir / 'best_model.pth')
    print(m.load_state_dict(state_dict))

    # f1ベスト書き出し
    best_epoch = log["val/f1_score"].idxmax() + 1
    print('... best epoch')
    print(log.iloc[[best_epoch - 1],])

    shutil.copy(output_dir / "snapshot_epoch_f1_{}.pth".format(best_epoch), output_dir / "f1_best_model.pth")

    m = get_model({
    'name': settings["model"]["name"],
    'params': {'pretrained': False, 'n_classes': 264}})
    state_dict = torch.load(output_dir / 'best_model.pth')
    print(m.load_state_dict(state_dict))

    print('... all well done')
    end_time = time.time()
    print('... elapsed time : {} minutes'.format((end_time - start_time)/60))



PERIOD = 5
def mono_to_color(
    X: np.ndarray, mean=None, std=None,
    norm_max=None, norm_min=None, eps=1e-6
    ):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V

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
        self.filterbank = None

    def __len__(self):
        return len(self.file_list)

    def mel(self, sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False,
            norm=1):
        """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins

        Parameters
        ----------
        sr        : number > 0 [scalar]
            sampling rate of the incoming signal

        n_fft     : int > 0 [scalar]
            number of FFT components

        n_mels    : int > 0 [scalar]
            number of Mel bands to generate

        fmin      : float >= 0 [scalar]
            lowest frequency (in Hz)

        fmax      : float >= 0 [scalar]
            highest frequency (in Hz).
            If `None`, use `fmax = sr / 2.0`

        htk       : bool [scalar]
            use HTK formula instead of Slaney

        norm : {None, 1, np.inf} [scalar]
            if 1, divide the triangular mel weights by the width of the mel band
            (area normalization).  Otherwise, leave all the triangles aiming for
            a peak value of 1.0

        Returns
        -------
        M         : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
            Mel transform matrix

        Notes
        -----
        This function caches at level 10.

        Examples
        --------
        >>> melfb = librosa.filters.mel(22050, 2048)
        >>> melfb
        array([[ 0.   ,  0.016, ...,  0.   ,  0.   ],
            [ 0.   ,  0.   , ...,  0.   ,  0.   ],
            ...,
            [ 0.   ,  0.   , ...,  0.   ,  0.   ],
            [ 0.   ,  0.   , ...,  0.   ,  0.   ]])


        Clip the maximum frequency to 8KHz

        >>> librosa.filters.mel(22050, 2048, fmax=8000)
        array([[ 0.  ,  0.02, ...,  0.  ,  0.  ],
            [ 0.  ,  0.  , ...,  0.  ,  0.  ],
            ...,
            [ 0.  ,  0.  , ...,  0.  ,  0.  ],
            [ 0.  ,  0.  , ...,  0.  ,  0.  ]])


        >>> import matplotlib.pyplot as plt
        >>> plt.figure()
        >>> librosa.display.specshow(melfb, x_axis='linear')
        >>> plt.ylabel('Mel filter')
        >>> plt.title('Mel filter bank')
        >>> plt.colorbar()
        >>> plt.tight_layout()
        """

        if fmax is None:
            fmax = float(sr) / 2

        if norm is not None and norm != 1 and norm != np.inf:
            raise ParameterError('Unsupported norm: {}'.format(repr(norm)))

        # Initialize the weights
        n_mels = int(n_mels)
        weights = np.zeros((n_mels, int(1 + n_fft // 2)))

        # Center freqs of each FFT bin
        fftfreqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # 'Center freqs' of mel bands - uniformly spaced between limits
        mel_f = np.array([200,
                200.65839880255413,
                201.37750848432012,
                201.51126380583162,
                201.67386956899037,
                201.87582543688856,
                202.13345527203808,
                202.47362290912503,
                202.94395795945513,
                203.63802229847101,
                204.7696777583669,
                206.9686321525935,
                212.88885671471746,
                217.43048869703566,
                217.67573891228102,
                217.99718059148606,
                218.43718046012455,
                219.07709875345563,
                220.09657180388132,
                221.99293137020203,
                226.94388589196586,
                232.62734674169212,
                235.10231672457329,
                236.44419592188694,
                239.27104445621455,
                247.84189145158842,
                258.1332636574189,
                266.45592538764413,
                287.9890352180489,
                310.53399991020956,
                329.7280481648527,
                365.2636543430171,
                412.1971069913186,
                474.9992071267056,
                543.8012938633804,
                613.9981630065175,
                684.4309008264818,
                746.926859193976,
                824.9481206944652,
                934.1647408605522,
                1059.406578152521,
                1191.4714986828917,
                1315.3661976656404,
                1417.0854269491579,
                1511.4688454929535,
                1612.80056993873,
                1721.6428871330952,
                1838.288761545141,
                1955.3394189452179,
                2064.968083010285,
                2166.8084495171515,
                2260.0916983488605,
                2338.3314875237083,
                2401.906399148228,
                2463.7128966413247,
                2524.8571234176743,
                2580.0580920624607,
                2628.9702036006215,
                2682.628540587129,
                2735.574322816629,
                2782.2949980039793,
                2823.2077006567883,
                2863.515614399661,
                2908.3053773334987,
                2947.9254283746377,
                2987.2898160362897,
                3026.660057289936,
                3066.3058926403455,
                3110.9589651232495,
                3150.8433299423505,
                3190.7647125250114,
                3235.4147868690166,
                3275.3314141881165,
                3315.610520650735,
                3360.6813727382987,
                3400.707128832307,
                3440.631288594453,
                3485.2582072693517,
                3525.048226029132,
                3565.061232929195,
                3610.3530117659466,
                3656.158234266066,
                3702.626404589744,
                3749.713173618707,
                3797.136734159918,
                3844.688148985763,
                3892.179804142279,
                3939.684061309631,
                3992.7908662105892,
                4046.0964639926615,
                4094.048026768339,
                4147.715225077025,
                4201.803328840861,
                4250.415130590165,
                4304.400570488394,
                4358.891311144112,
                4413.626381238448,
                4468.247903671312,
                4523.200949785816,
                4584.385109100807,
                4646.338108408218,
                4708.794686519528,
                4771.600484347028,
                4834.854981582876,
                4904.5553996394865,
                4974.441512229322,
                5045.491312421904,
                5130.717507368683,
                5223.421347621407,
                5317.079138212701,
                5418.7071099135665,
                5535.597836296701,
                5667.909048600521,
                5808.1872217746295,
                5956.457881201861,
                6127.4753815696895,
                6314.274205611619,
                6509.692128073985,
                6736.107119503475,
                7008.663641495503,
                7312.530456313335,
                7655.887319816534,
                8100.341736238419,
                8973.536714095795,
                10144.613928413162,
                11315.69114273053,
                12486.768357047898,
                13657.845571365266,
                14828.922785682633,
                16000])

        fdiff = np.diff(mel_f)
        ramps = np.subtract.outer(mel_f, fftfreqs)

        for i in range(n_mels):
            # lower and upper slopes for all bins
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i+2] / fdiff[i+1]

            # .. then intersect them with each other and zero
            weights[i] = np.maximum(0, np.minimum(lower, upper))

        if norm == 1:
            # Slaney-style mel is scaled to be approx constant energy per channel
            enorm = 2.0 / (mel_f[2:n_mels+2] - mel_f[:n_mels])
            weights *= enorm[:, np.newaxis]

        # Only check weights if f_mel[0] is positive
        if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
            # This means we have an empty channel somewhere
            warnings.warn('Empty filters detected in mel frequency basis. '
                        'Some channels will produce empty responses. '
                        'Try increasing your sampling rate (and fmax) or '
                        'reducing n_mels.')

        return weights


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

        n_fft = self.melspectrogram_parameters["n_fft"]
        if self.filterbank is not None: pass
        else:
            self.filterbank = self.mel(sr, n_fft,
                self.melspectrogram_parameters["n_mels"], 
                self.melspectrogram_parameters["fmin"],
                self.melspectrogram_parameters["fmax"])

        power = 2.0
        S = np.abs(librosa.stft(y, n_fft=n_fft))**power
        melspec = np.dot(self.filterbank, S)
        melspec = librosa.power_to_db(melspec).astype(np.float32)

        if self.spectrogram_transforms:
            melspec = self.spectrogram_transforms(melspec)
        else:
            pass

        image = mono_to_color(melspec)
        height, width, _ = image.shape
        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)

#         labels = np.zeros(len(BIRD_CODE), dtype="i")
        labels = np.zeros(len(self.bird_code), dtype="f")
        labels[self.bird_code[ebird_code]] = 1

        return image, labels


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


def get_model(args: tp.Dict):
    model =getattr(resnest_torch, args["name"])(pretrained=args["params"]["pretrained"])
    # 最終層を書き換える
    del model.fc
    # # use the same head as the baseline notebook.
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, args["params"]["n_classes"]))
    
    return model


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
        eval_value = eval_func(target, output)
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
