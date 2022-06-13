"""
# Tempo Estimation Stage
"""

import pandas as pd
import numpy as np
import yaml
import json
from loguru import logger as log
from alive_progress import alive_it
import madmom

from challenges import utils


def main():
    with open("params.yaml", "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    with open("reports/beat-detection/test-beats.json", "r", encoding="utf-8") as file:
        test_beats = json.load(file)

    with open("reports/beat-detection/val-beats.json", "r", encoding="utf-8") as file:
        val_beats = json.load(file)

    beats: dict = {
        "test": test_beats,
        "val": val_beats,
    }

    tempo: dict = {"test": {}, "val": {}}

    signal_processor = madmom.audio.signal.SignalProcessor(sample_rate=44100)

    for stage in beats.keys():
        log.info(f"aggregating beats of stage ’{stage}’")
        bar = alive_it(beats[stage].keys())
        for file in bar:
            bar.text = file

            # calculate signal_length
            signal_length = signal_processor(
                f"data/raw/{stage if stage == 'test' else 'train'}/{file}.wav"
            ).length

            tempo_estimate = len(beats[stage][file]["beats"]) / signal_length * 60

            tempo[stage][file] = {"tempo": [tempo_estimate]}

    with open("reports/tempo-estimation/val-tempo.json", "w") as file:
        json.dump(
            tempo["val"],
            file,
            indent=2,
        )

    with open("reports/tempo-estimation/test-tempo.json", "w") as file:
        json.dump(
            tempo["test"],
            file,
            indent=2,
        )

    targets: dict = {}

    log.info(f"gathering tempo targets")
    bar = alive_it(tempo["val"].keys())
    for file in bar:
        bar.text = file

        target_tempo = list(
            madmom.io.load_events("data/raw/train/" + str(file) + ".tempo.gt")
        )

        targets[file] = {"tempo": target_tempo}

    with open("reports/tempo-estimation/val-targets.json", "w") as file:
        json.dump(
            targets,
            file,
            indent=2,
        )


if __name__ == "__main__":
    main()
