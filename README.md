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
