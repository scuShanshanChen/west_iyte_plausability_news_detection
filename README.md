# west_iyte_plausability_news_detection

## Installation
Run in one of virtual environment. If you have a server incompatible with venv, do the [following](https://stackoverflow.com/questions/50777849/from-conda-create-requirements-txt-for-pip3):

- `conda create --name plausible`
- `conda activate plausible`
- `conda install pip`
- `pip freeze > requirements.txt`
- `pip install -r requirements.txt`

## Running the experiments

### Training

an example script how to train the model `python main.py --training_mode random_split --epochs 20`


## Experiments

This table will be updated...

Model | Mode | Acc | F1 | Recall | Precision
| --- | --- | --- | --- | --- | --- |
majority | dev | 0.58 | 0.73 | 1.0 | 0.58
majority | test | 0.6 | 0.75 | 1.0 | 0.6
