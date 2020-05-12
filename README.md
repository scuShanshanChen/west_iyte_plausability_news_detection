# WeST & IYTE Plausibility Detection on News

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

`python -m baselines.linear_models --seed 42 --training_mode random_split --feature headline`

`feature` is input type: you can select `headline`, `body`, `merged` (the headline is joined to the body)

### Training

an example script how to train the model `python main.py --training_mode random_split --epochs 20`


## Experiments

This table will be updated...

Following table is from the experiment with random split with seed 42
 Model | Feature | Mode | Acc | F1 | Recall | Precision
| --- | --- | --- | --- | --- | --- | --- |
majority | any feature | dev | 0.62 | 0.76 | 1.0 | 0.62
majority | any feature | test | 0.56 | 0.72 | 1.0 | 0.56
NBSVM-unigram | headline | dev | 0.56 | 0.72 | 1.0 | 0.56
NBSVM-unigram | headline | test | 0.72 | 0.76 | 0.79 | 0.73
NBSVM-unigram+bigram | headline | dev | 0.72 | 0.76 | 0.79 | 0.73
NBSVM-unigram+bigram | headline | test | 0.7 | 0.76 | 0.84 | 0.69
NBSVM-unigram | body | dev | 0.56 | 0.72 | 1.0 | 0.56
NBSVM-unigram | body | test | 0.75 | 0.79 | 0.84 | 0.75
NBSVM-unigram+bigram | body | dev | 0.75 | 0.79 | 0.84 | 0.75
NBSVM-unigram+bigram | body | test | 0.79 | 0.82 | 0.9 | 0.76
NBSVM-unigram | merged | dev | 0.56 | 0.72 | 1.0 | 0.56
NBSVM-unigram | merged | test | 0.75 | 0.79 | 0.85 | 0.74
NBSVM-unigram+bigram | merged | dev | 0.75 | 0.79 | 0.85 | 0.74
NBSVM-unigram+bigram | merged | test | 0.78 | 0.82 | 0.9 | 0.75

Following table is from the experiment with 10-fold cross validation with seed 42
Model | Feature | Acc_CV | F1_CV | Recall_CV | Precision_CV
| --- | --- | --- | --- | --- | --- |
majority | any feature | 0.58 +/- 0.00 +/- 0.00 | 0.74 +/- 0.00 +/- 0.00 | 1.00 +/- 0.00 +/- 0.00 | 0.58 +/- 0.00 +/- 0.00
NBSVM-unigram | headline | 0.73 +/- 0.03 +/- 0.03 | 0.77 +/- 0.02 +/- 0.02 | 0.81 +/- 0.02 +/- 0.02 | 0.74 +/- 0.02 +/- 0.02
NBSVM-unigram+bigram | headline | 0.72 +/- 0.04 +/- 0.04 | 0.78 +/- 0.03 +/- 0.03 | 0.83 +/- 0.04 +/- 0.04 | 0.73 +/- 0.03 +/- 0.03
NBSVM-unigram | body | 0.80 +/- 0.02 +/- 0.02 | 0.83 +/- 0.02 +/- 0.02 | 0.86 +/- 0.04 +/- 0.04 | 0.80 +/- 0.03 +/- 0.03
NBSVM-unigram+bigram | body | 0.80 +/- 0.03 +/- 0.03 | 0.83 +/- 0.03 +/- 0.03 | 0.88 +/- 0.05 +/- 0.05 | 0.80 +/- 0.03 +/- 0.03
NBSVM-unigram | merged | 0.79 +/- 0.04 +/- 0.04 | 0.83 +/- 0.03 +/- 0.03 | 0.86 +/- 0.05 +/- 0.05 | 0.80 +/- 0.03 +/- 0.03
NBSVM-unigram+bigram | merged | 0.80 +/- 0.03 +/- 0.03 | 0.84 +/- 0.03 +/- 0.03 | 0.88 +/- 0.04 +/- 0.04 | 0.80 +/- 0.03 +/- 0.03