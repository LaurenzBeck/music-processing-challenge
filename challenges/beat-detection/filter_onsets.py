"""
# Onset Filtering Stage

This stage is a model based approach to filter the onsets and extract only the beats.
It uses an approach combining ideas from using autocorrelation on pulse trains and
multiple agents approaches to beat detection.
"""

import pandas as pd
import numpy as np
import yaml
import json
from loguru import logger as log
from alive_progress import alive_it
import madmom

from challenges import utils


def gaussian(x, mu, std):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(std, 2.0)))


def gaussian_train(bpm, offset, signal_length, gaus_width):
    x = np.arange(0, signal_length + gaus_width, 1 / 44100, dtype=float)
    num_beats = int(np.ceil((signal_length - offset) * bpm / 60))
    gaus_train = np.zeros_like(x)
    for beat_idx in range(num_beats):
        gaus_train += gaussian(x, offset + beat_idx * 60 / bpm, gaus_width / 4)

    return gaus_train, num_beats


def score(gaus_train, beats_, num_beats):
    return (
        sum([gaus_train[utils.get_index_from_timestamp(beat)] for beat in beats_])
        / num_beats
    )


def main():
    with open("params.yaml", "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    with open(
        "reports/onset-detection/test-onsets.json", "r", encoding="utf-8"
    ) as file:
        test_onsets = json.load(file)

    with open("reports/onset-detection/val-onsets.json", "r", encoding="utf-8") as file:
        val_onsets = json.load(file)

    onsets: dict = {
        "test": test_onsets,
        "val": val_onsets,
    }

    beats: dict = {"test": {}, "val": {}}

    signal_processor = madmom.audio.signal.SignalProcessor(sample_rate=44100)

    for stage in onsets.keys():
        log.info(f"filtering onsets of stage ’{stage}’")
        bar = alive_it(onsets[stage].keys())
        for file in bar:
            bar.text = file

            # calculate signal_length
            signal_length = signal_processor(
                f"data/raw/{stage if stage == 'test' else 'train'}/{file}.wav"
            ).length

            fits = []

            for bpm in range(
                params["beat_detection"]["lower_bpm_bound"],
                params["beat_detection"]["upper_bpm_bound"],
                params["beat_detection"]["bpm_sweep_step_size"],
            ):
                for offset in np.linspace(
                    0.0,
                    params["beat_detection"]["upper_offset_bound"],
                    params["beat_detection"]["num_offsets_for_sweep"],
                ):
                    gaus, num_beats = gaussian_train(
                        bpm,
                        offset,
                        max(signal_length, max(onsets[stage][file]["onsets"])),
                        params["beat_detection"]["width_of_gaussian"],
                    )
                    fit = score(gaus, onsets[stage][file]["onsets"], num_beats)
                    fits.append(
                        {
                            "bpm": bpm,
                            "offset": offset,
                            "gaus": gaus,
                            "num_beats": num_beats,
                            "fit": fit,
                        }
                    )

            fits = pd.DataFrame(fits)
            best_fit = fits[fits["fit"] == fits["fit"].max()]

            train = best_fit["gaus"].to_numpy()[0]

            onsets_idx = list(
                map(utils.get_index_from_timestamp, onsets[stage][file]["onsets"])
            )
            beats_idx = list(
                filter(
                    lambda idx: train[idx] > params["beat_detection"]["beat_threshold"],
                    onsets_idx,
                )
            )

            beats[stage][file] = {
                "beats": list(map(utils.get_timestamp_from_index, beats_idx))
            }

    with open("reports/beat-detection/val-beats.json", "w") as file:
        json.dump(
            beats["val"],
            file,
            indent=2,
        )

    with open("reports/beat-detection/test-beats.json", "w") as file:
        json.dump(
            beats["test"],
            file,
            indent=2,
        )

    targets: dict = {}

    log.info(f"gathering beat targets")
    bar = alive_it(beats["val"].keys())
    for file in bar:
        bar.text = file

        target_beats = list(
            madmom.io.load_events("data/raw/train/" + str(file) + ".beats.gt")
        )

        targets[file] = {"beats": target_beats}

    with open("reports/beat-detection/val-targets.json", "w") as file:
        json.dump(
            targets,
            file,
            indent=2,
        )


if __name__ == "__main__":
    main()
