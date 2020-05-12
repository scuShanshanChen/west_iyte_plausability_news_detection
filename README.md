# west_iyte_plausability_news_detection

## Installation
Run in one of virtual environment. If you have a server incompatible with venv, do the [following](https://stackoverflow.com/questions/50777849/from-conda-create-requirements-txt-for-pip3):

- `conda create --name plausible`
- `conda activate plausible`
- `conda install pip`
- `pip freeze > requirements.txt`
- `pip install -r requirements.txt`

## Running the experiments
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

