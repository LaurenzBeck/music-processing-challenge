""" 
# Feature Extraction Stage

this stage heavily relies on madmom's processors and utilities
"""

import os
import pickle
from functools import partial

import madmom
import numpy as np
import yaml
from alive_progress import alive_it
from fastai.data.transforms import get_files
from loguru import logger as log
from scipy import signal, stats


def process_wav_files(wav_files, processors, act, stage, add_file_extension=True):
    for feature_name, processor in processors.items():

        if not os.path.exists(f"data/interim/{stage}/{feature_name}"):
            os.makedirs(f"data/interim/{stage}/{feature_name}")

        io = madmom.processors.IOProcessor([processor], [act])

        bar = alive_it(wav_files[stage])

        for file in bar:
            bar.title = feature_name
            bar.text = file.name
            file_wav = str(file) + ".wav" if add_file_extension else file
            with open(
                f"data/interim/{stage}/{feature_name}/{file.name}.txt", "w"
            ) as feature_file:
                io(file_wav, feature_file)


def main():
    with open("params.yaml", "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    with open("data/processed/train_files.pkl", "rb") as train_files, open(
        "data/processed/val_files.pkl", "rb"
    ) as val_files:
        wav_files = {
            "train": pickle.load(train_files),
            "val": pickle.load(val_files),
            "test": get_files(params["data_dir_test"], extensions=".wav"),
        }

    fs = madmom.audio.signal.FramedSignalProcessor(
        fps=params["featurize"]["fps"],
        frame_size=params["featurize"]["frame_size"],
        origin="right",
    )

    processors = {
        "superflux": madmom.processors.SequentialProcessor(  # alternative: sound_pressure_level
            [
                fs,
                madmom.audio.stft.ShortTimeFourierTransformProcessor(),
                madmom.audio.spectrogram.SpectrogramProcessor(),
                madmom.audio.spectrogram.FilteredSpectrogramProcessor(num_bands=24),
                madmom.audio.spectrogram.LogarithmicSpectrogramProcessor(),
                madmom.audio.spectrogram.SpectrogramDifferenceProcessor(
                    diff_max_bins=3, positive_diffs=True
                ),
                partial(np.mean, axis=1),
            ]
        ),
        "energy": madmom.processors.SequentialProcessor(  # alternative: sound_pressure_level
            [
                fs,
                madmom.audio.signal.energy,
            ]
        ),
        "rms": madmom.processors.SequentialProcessor(
            [
                fs,
                madmom.audio.signal.root_mean_square,
            ]
        ),
        "sound_preassure_level": madmom.processors.SequentialProcessor(
            [
                fs,
                madmom.audio.signal.sound_pressure_level,
            ]
        ),
        "welch": madmom.processors.SequentialProcessor(
            [
                fs,
                lambda s: signal.welch(s)[1],  # only return the spectral density,
                partial(np.mean, axis=1),
            ]
        ),
        "mean": madmom.processors.SequentialProcessor(
            [
                fs,
                partial(np.mean, axis=1),
            ]
        ),
        "kurtosis": madmom.processors.SequentialProcessor(
            [
                fs,
                lambda s: np.array(list(map(stats.kurtosis, s))),
            ]
        ),
        "skew": madmom.processors.SequentialProcessor(
            [
                fs,
                lambda s: np.array(list(map(stats.skew, s))),
            ]
        ),
        "variation": madmom.processors.SequentialProcessor(
            [
                fs,
                lambda s: np.array(list(map(stats.variation, s))),
            ]
        ),
    }

    act = madmom.features.ActivationsProcessor(
        mode="save", fps=params["featurize"]["fps"], sep="\n"
    )

    log.info("start extraction of train features")
    if not os.path.exists("data/interim/train"):
        os.makedirs("data/interim/train")
    process_wav_files(wav_files, processors, act, "train")

    log.info("start extraction of val features")
    if not os.path.exists("data/interim/val"):
        os.makedirs("data/interim/val")
    process_wav_files(wav_files, processors, act, "val")

    log.info("start extraction of test features")
    if not os.path.exists("data/interim/test"):
        os.makedirs("data/interim/test")
    process_wav_files(wav_files, processors, act, "test", add_file_extension=False)


if __name__ == "__main__":
    main()
