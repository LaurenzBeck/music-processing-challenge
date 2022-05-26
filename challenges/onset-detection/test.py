"""
# Model Testing Stage

Train a fastai tabular model.
"""

from fastai.torch_core import defaults
from torch import device

defaults.device = device("cpu")

import pandas as pd
import yaml
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

    learn = learn.load(params["train"]["model_file_name"])

    test_df = pd.read_csv("data/processed/test/data.csv", low_memory=False)
    dl = learn.dls.test_dl(test_df)
    preds = learn.get_preds(dl=dl)[0]

    onsets = preds.argmax(dim=1)

    test_df["onset probabilty"] = preds[:, 1]
    test_df["onset prediction"] = onsets

    test_df.to_csv(
        "reports/test-df.csv",
        columns=["file", "timestamps", "onset probabilty", "onset prediction"],
        index=False,
    )

    with open("reports/test-onsets.json", "w") as file:
        json.dump(
            {
                file_name[:-4]: {
                    "onsets": list(
                        test_df[
                            (test_df["file"] == file_name)
                            & (test_df["onset prediction"] == 1)
                        ]["timestamps"]
                    )
                }
                for file_name in test_df["file"].unique()
            },
            file,
            indent=2,
        )

    val_df = pd.read_csv("data/processed/val/data.csv", low_memory=False)
    dl = learn.dls.test_dl(val_df)
    preds = learn.get_preds(dl=dl)[0]

    onsets = preds.argmax(dim=1)

    val_df["onset probabilty"] = preds[:, 1]
    val_df["onset prediction"] = onsets

    val_df.to_csv(
        "reports/val-df.csv",
        columns=[
            "file",
            "timestamps",
            "onset",
            "onset probabilty",
            "onset prediction",
            "superflux",
        ],
        index=False,
    )

    with open("reports/val-onsets.json", "w") as file:
        json.dump(
            {
                file_name: {
                    "onsets": list(
                        val_df[
                            (val_df["file"] == file_name)
                            & (val_df["onset prediction"] == 1)
                        ]["timestamps"]
                    )
                }
                for file_name in val_df["file"].unique()
            },
            file,
            indent=2,
        )

    with open("reports/val-targets.json", "w") as file:
        json.dump(
            {
                file_name: {
                    "onsets": list(
                        val_df[(val_df["file"] == file_name) & (val_df["onset"])][
                            "timestamps"
                        ]
                    )
                }
                for file_name in val_df["file"].unique()
            },
            file,
            indent=2,
        )


if __name__ == "__main__":
    main()
