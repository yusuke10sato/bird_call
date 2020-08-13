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
def train(config_path):
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
        val_loader, optimizer, loss_func,
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

    print('... all well done')
    end_time = time.time()
    print('... elapsed time : {} minutes'.format((end_time - start_time)/60))



PERIOD = 2
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
        file_list: tp.List[tp.List[str]], bird_code: tp.Dict, img_size=224,
        waveform_transforms=None, spectrogram_transforms=None, melspectrogram_parameters={},
        clip_aroud_peak=False
        ):
        self.file_list = file_list  # list of list: [file_path, ebird_code]
        self.bird_code = bird_code
        self.img_size = img_size
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters
        self.clip_aroud_peak = clip_aroud_peak

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        wav_path, ebird_code = self.file_list[idx]

        y, sr = sf.read(wav_path)

        if self.clip_aroud_peak:
            y = exract_peak_near(y, sr)

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

        melspec = librosa.feature.melspectrogram(y, sr=sr, **self.melspectrogram_parameters)
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
    train_dataset = SpectrogramDataset(train_file_list, bird_code, **args_dataset)
    val_dataset = SpectrogramDataset(val_file_list, bird_code, **args_dataset)
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
        eval_value = eval_func(output, target).item()
        ppe.reporting.report({"val/{}".format(eval_aame): eval_value})


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
            'epoch', 'iteration', 'lr', 'train/loss', 'val/loss', "elapsed_time"]),
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
    ]
           
    # # set extensions to manager
    for ext in my_extensions:
        if isinstance(ext, tuple):
            manager.extend(ext[0], trigger=ext[1])
        else:
            manager.extend(ext)
        
    return manager


def exract_peak_near(x, sr):
    i = np.argmax(x)
    if len(x) < sr * 2:
        start_index = 0
        end_index = len(x) - 1
    # スタート地点が0より前
    elif i - sr * 1 < 0:
        start_index = 0
        end_index = int(sr*2)
    # end_indexがlen_xより大きい
    elif i + sr*1 >= len(x):
        start_index = int(len(x) - sr*2 - 1)
        end_index = int(len(x) - 1)
    else:
        start_index = int(i - sr * 1)
        end_index = int(i + sr * 1)
    return x[start_index:end_index]


if __name__ == '__main__':
    train()
