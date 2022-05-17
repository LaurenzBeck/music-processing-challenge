""" 
# Train/Val Split Stage

this stage splits the training set explicitly in a separate stage,
which improves reproducibility.
"""

import yaml
import pickle
from fastai.data.transforms import get_files
from sklearn.model_selection import train_test_split
from loguru import logger as log
import snoop

@snoop()
def main():
    with open("params.yaml", 'r') as f:
        params = yaml.safe_load(f)
    
    wav_files = get_files(params["data_dir_train"], extensions=".wav")

    log.info(f"found {len(wav_files)} .wav files")

    files = list(map(lambda file: file.with_suffix(""), wav_files))

    train_files, val_files = train_test_split(files, train_size=params["train_val_split"], random_state=params["seed"])

    log.info(f"splitted the files into {len(train_files)} training and {len(val_files)} validation samples")

    with open("data/processed/train_files.pkl", "wb") as f:
        pickle.dump(train_files, f)
    
    with open("data/processed/val_files.pkl", "wb") as f:
        pickle.dump(val_files, f)

if __name__ == '__main__':
    main()