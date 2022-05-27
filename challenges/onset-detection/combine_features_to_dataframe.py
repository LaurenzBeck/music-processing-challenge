"""
# Prepare Onset labels Stage

This stage creates as csv file with binary onset labels
that are aligned with the frame rate of the FramedSignalProcessor of the features
"""

import itertools
import os
import pickle

import madmom
import pandas as pd
import yaml
from alive_progress import alive_it
from loguru import logger as log

from challenges import utils


def combine_features(stage, fps, add_labels=True):

    data = {}

    for feature in os.scandir(f"data/interim/{stage}"):
        column_data = []
        file_names = []
        timestamps = []
        bar = alive_it(os.walk(f"data/interim/{stage}/{feature.name}"))
        for (root, _, files) in bar:
            for file in files:
                bar.title = feature.name
                bar.text = file
                features = madmom.features.Activations(
                    os.path.join(root, file), fps=fps, sep="\n"
                )
                column_data.extend(features)
                file_names.extend(itertools.repeat(file[:-4], len(features)))
                timestamps.extend(
                    utils.get_timestamps_for_framerate(fps, len(features))
                )
        data["file"] = file_names
        data[feature.name] = column_data
        data["timestamps"] = timestamps

    df = pd.DataFrame(data)

    if add_labels:
        log.info("adding labels")
        labels = []
        for (root, _, files) in os.walk(f"data/processed/{stage}/labels"):
            bar = alive_it(files)
            for file in bar:
                bar.title = "labels"
                bar.text = file
                labels_list = list(
                    pd.read_csv(os.path.join(root, file), header=None)[0]
                )
                labels.extend(labels_list)

        df[
            "onset"
        ] = labels  # pd.concat(labels, ignore_index=True) #? does this concat in the correct order?

    df.to_csv(f"data/processed/{stage}/data.csv", index=False)

    return df


def main():
    with open("params.yaml", "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    log.info("combining train features")
    if not os.path.exists("data/processed/train"):
        os.makedirs("data/processed/train")
    combine_features("train", fps=params["featurize"]["fps"])

    log.info("combining val features")
    if not os.path.exists("data/processed/val"):
        os.makedirs("data/processed/val")
    combine_features("val", fps=params["featurize"]["fps"])

    log.info("combining test features")
    if not os.path.exists("data/processed/test"):
        os.makedirs("data/processed/test")
    combine_features("test", fps=params["featurize"]["fps"], add_labels=False)


if __name__ == "__main__":
    main()
