# Tone Regonition

Train a machine learning model to classify tones in Chinese through audio files.

# Prerequisites

## 1) Hardware
* All experiments in paper were conducted with single RTX 2060 SUPER GPU (8GB).

## 2) Software
* Ubuntu 20.04
* Python 3.8+
  - `pip install -r requirements.txt` 


## 3) Data

```sh
# Download Audio files.
mkdir data/raw
wget https://www.dropbox.com/s/33f8lvgfjphnrzh/Audio.zip?dl=0
unzip "Audio.zip@dl=0"
mv Audio data/raw

# Request more data. Insert api key from https://api.forvo.com/
python make_dataset.py 

# Clean the audio files.
python clean_data.py

# Prepare training data.
python prepare_data.py 
```

# Run
You can see more configurations in [configs](src/configs.py)

## Train
```sh
python train.py --model='ALL' --epochs=50
```

## Evaluation
```sh
python evaluate.py 
