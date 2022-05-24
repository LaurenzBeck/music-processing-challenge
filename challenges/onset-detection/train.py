"""
# Model Training Stage

Train a fastai tabular model.
"""

# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""

from fastai.torch_core import defaults
from torch import device

defaults.device = device("cpu")

import pandas as pd
import yaml
from dvclive.fastai import DvcLiveCallback
from fastai.tabular.all import *
from loguru import logger as log


def main():
    with open("params.yaml", "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    set_seed(params["seed"])

    df_train = pd.read_csv("data/processed/train/data.csv", low_memory=False)
    df_val = pd.read_csv("data/processed/val/data.csv", low_memory=False)
    df = pd.concat([df_train, df_val])

    train_len = len(df_train)
    splits = (list(range_of(df)[:train_len]), list(range_of(df)[train_len:]))

    procs = [Categorify, Normalize]

    cont, cat = cont_cat_split(
        df.loc[
            :,
            ~df.columns.isin(
                ["file", "timestamps", "sound_preassure_level", "variation"]
            ),
        ],
        1,
        dep_var="onset",
    )

    to = TabularPandas(
        df,
        procs=procs,
        cont_names=cont,
        y_names="onset",
        splits=splits,
        device=device("cpu"),
    )

    dls = to.dataloaders(bs=params["train"]["batch_size"], device=device("cpu"))

    learn = tabular_learner(dls, metrics=[accuracy], cbs=[])

    learn.fit_one_cycle(
        params["train"]["epochs"],
        params["train"]["learning_rate"],
        cbs=[DvcLiveCallback(model_file=params["train"]["model_file_name"])],
    )


if __name__ == "__main__":
    main()
