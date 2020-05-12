# west_iyte_plausability_news_detection

## Installation
Run in one of virtual environment. If you have a server incompatible with venv, do the [following](https://stackoverflow.com/questions/50777849/from-conda-create-requirements-txt-for-pip3):

- `conda create --name plausible`
- `conda activate plausible`
- `conda install pip`
- `pip freeze > requirements.txt`
- `pip install -r requirements.txt`

## Running the experiments

### Preparing experiment data
You can create random splits from all samples given a seed value or can create kfold splits as train and test set given a seed and a kfold value.

Random split with seed 42 is executed as follows:

`python -m data --seed 42 --training_mode random_split` 

That command creates a folder name as `random_seed_42` and files as `dev.tsv`, and `test.tsv`, `train.tsv`. `train.tsv` is %60 of all data, `dev.tsv` and `test.tsv` are %20 of all data.

10 fold split with seed 10 is executed as follows:

`python -m data --seed 42 --training_mode kfold --kfold 10` 

That command creates a folder name as `kfold_random_seed_42` and creates `kfold_{kfold_id}_train.tsv` and `kfold_{kfold_id}_test.tsv` for each fold.

### Linear Models
To get results of linear models, run the following:

`python -m baselines.linear_models --seed 42`

### Training

an example script how to train the model `python main.py --training_mode random_split --epochs 20`


## Experiments

This table will be updated...

Model | Mode | Acc | F1 | Recall | Precision
| --- | --- | --- | --- | --- | --- |
majority | dev | 0.58 | 0.73 | 1.0 | 0.58
majority | test | 0.6 | 0.75 | 1.0 | 0.6
NBSVM-unigram | dev | 0.6 | 0.75 | 1.0 | 0.6
NBSVM-unigram | test | 0.78 | 0.82 | 0.84 | 0.8
NBSVM-unigram+bigram | dev | 0.78 | 0.82 | 0.84 | 0.8
NBSVM-unigram+bigram | test | 0.79 | 0.84 | 0.88 | 0.79

