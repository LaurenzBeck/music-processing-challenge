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

    df_train = pd.read_csv(
        "data/processed/onset-detection/train/data.csv", low_memory=False
    )
    df_val = pd.read_csv(
        "data/processed/onset-detection/val/data.csv", low_memory=False
    )
    df = pd.concat([df_train, df_val])
    df["onset"] = df["onset"].astype("category")

    train_len = len(df_train)
    splits = (list(range_of(df)[:train_len]), list(range_of(df)[train_len:]))

    procs = [Categorify, FillMissing, Normalize]

    cont, cat = cont_cat_split(
        df.loc[
            :,
            ~df.columns.isin(["file", "timestamps"]),
        ],
        1,
        dep_var="onset",
    )

    dls = TabularDataLoaders.from_df(
        df,
        y_names="onset",
        cat_names=cat,
        cont_names=cont,
        procs=procs,
        splits=splits,
        device=device("cpu"),
        bs=params["onset_detection"]["train"]["batch_size"],
    )

    # construct class weights
    class_count_df = df.groupby("onset").count()
    n_0, n_1 = class_count_df.iloc[0, 0], class_count_df.iloc[1, 0]
    w_0 = (n_0 + n_1) / (2.0 * n_0)
    w_1 = (n_0 + n_1) / (2.0 * n_1)
    class_weights = torch.FloatTensor([w_0, w_1])

    learn = tabular_learner(
        dls,
        loss_func=LabelSmoothingCrossEntropyFlat(weight=class_weights),
        opt_func=Lamb,
        layers=params["onset_detection"]["train"]["layers"],
        config=tabular_config(
            ps=params["onset_detection"]["train"]["dropout_probs"],
            act_cls=Mish(inplace=True),
        ),
        metrics=[
            accuracy,
            F1Score(labels=[0, 1]),
            Precision(labels=[0, 1]),
            Recall(labels=[0, 1]),
        ],
    )

    learn.summary()

    learn.fit_one_cycle(
        params["onset_detection"]["train"]["epochs"],
        params["onset_detection"]["train"]["learning_rate"],
        wd=params["onset_detection"]["train"]["weight_decay"],
        cbs=[
            DvcLiveCallback(
                model_file=params["onset_detection"]["train"]["model_file_name"],
                path="reports/onset-detection/dvclive",
            )
        ],
    )


if __name__ == "__main__":
    main()
