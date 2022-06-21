![image_header](./studio.jpg)

<h1 align="center">🎛️🎧 - Audio and Music Processing - 🎼🎶</h1>

<p align="center">
    Special Topics - JKU Linz
</p>

<p align="center">
    <a href="https://www.repostatus.org/#inactive"><img src="https://www.repostatus.org/badges/latest/inactive.svg" alt="Project Status: Inactive – The project has reached a stable, usable state but is no longer being actively developed; support/maintenance will be provided as time allows." /></a>
</p>

<p align="center">
    <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.9+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>
</p>

---

## Project

This project was part of my master studies in Artificial Intelligence at the Johannes Kepler University in Linz.
During the Special Topics lecture on audio and music processing from Rainer Kelz, I took part in three challenges:

+ Onset Detection
+ Beat Detection
+ Tempo Detection

**Team:** NeuraBeats

## Installation

To install the projects dependencies and create a virtual environment, make sure that your system has python (>=3.9,<3.10) and [poetry](https://python-poetry.org/) installed.

Then `cd` into the projects root directory and call: `$poetry install`.

Alternatively, install the dependencies from the `requirements.txt` file with your python environment manager of choice.

I was not allowed to make the dataset public, which is why one needs to add the challenge dataset in the `data/raw/` directory.

## Project structure

```
.
├── challenges                   # scripts and pipeline descriptions of the three challenges
│   ├── beat-detection
│   ├── onset-detection
│   ├── tempo-estimation
│   ├── train_val_split.py
│   └── utils.py
├── data
│   ├── interim
│   ├── processed
│   └── raw                      # the train and test .wav files and labels are stored here
├── dvc.lock
├── dvc_plots
├── dvc-storage                  # dvc storage backend (not included in this public repo)
├── LICENSE
├── models
├── notebooks                    # exploratory programming
├── params.yaml                  # configuration for the three challenges
├── poetry.lock
├── pyproject.toml               # python environment information
├── README.md
├── reports                      # predictions and project reports
│   ├── beat-detection
│   ├── onset-detection
│   └── tempo-estimation
└── requirements.txt
```

## Running the data pipelines

The three challenges were implemented as [dvc](https://dvc.org/) pipelines, which allows for a complete reprodicibility of every experiment, given that the datbackend storage is available. This is achieved by a git-centric approach, were not only the code is versioned with git, but also the configuration, the data, the artifacts, the models and the metrics. 

The pipelines are defined by the `dvc.yaml` files in the `challenges` directory. To run them all, simply call `$dvc repro -P`.
If you want to execute the scripts manually, you could go through the stages in the `dvc.yaml` files and call the `cmd` value of every stage from the projects root.
