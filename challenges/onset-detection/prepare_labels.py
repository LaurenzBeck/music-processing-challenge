"""
# Prepare Onset labels Stage

This stage creates as csv file with binary onset labels
that are aligned with the frame rate of the FramedSignalProcessor of the features
"""

import os
import pickle

import madmom
import pandas as pd
import yaml
from alive_progress import alive_it
from loguru import logger as log


class OnsetAssignmentProcessor(madmom.processors.Processor):
    def __init__(self, fps):
        self.signal = madmom.audio.signal.FramedSignalProcessor(fps=fps)
        self.fps = fps

    def process(self, file):
        onsets = madmom.io.load_events(str(file) + ".onsets.gt")
        fs = self.signal(str(file) + ".wav")

        labels = []
        for window_index in range(len(fs)):
            window_start_time = window_index / self.fps
            window_end_time = window_index / self.fps + 1.0 / self.fps
            labels.append(
                any([window_start_time < onset <= window_end_time for onset in onsets])
            )
        return labels


class CSVOutputProcessor(madmom.processors.OutputProcessor):
    def process(self, data, output):

        df = pd.DataFrame(data)

        df.to_csv(output, header=False, index=False)

        return df


def main():
    with open("params.yaml", "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    with open("data/processed/train_files.pkl", "rb") as train_files, open(
        "data/processed/val_files.pkl", "rb"
    ) as val_files:
        files = {
            "train": pickle.load(train_files),
            "val": pickle.load(val_files),
        }

    csv = CSVOutputProcessor()

    log.info("preparing labels ...")

    for stage, files in files.items():

        if not os.path.exists(f"data/processed/{stage}/labels"):
            os.makedirs(f"data/processed/{stage}/labels")

        processor = OnsetAssignmentProcessor(params["featurize"]["fps"])
        io = madmom.processors.IOProcessor([processor], [csv])

        bar = alive_it(files)

        for file in bar:
            bar.title = stage
            bar.text = file.name

            with open(
                f"data/processed/{stage}/labels/{file.name}.txt", "w"
            ) as label_file:
                io(file, label_file)


if __name__ == "__main__":
    main()
