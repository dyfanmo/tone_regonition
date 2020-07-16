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
wget https://www.dropbox.com/s/le1bc0e20sshu4b/Audio.zip?dl=0
unzip Audio.zip
mv Audio data/raw

# Request more data. Insert api key from https://api.forvo.com/
python make_dataset.py --api_key=''

# Clean the audio files.
python clean_data.py

# Prepare training data.
python data_loader.py 
```

# Run
You can see more configurations in [configs](src/configs) folder

## Train
```sh
python train.py --model='ALL' --epochs=50
```

## Evaluation
```sh
python evaluate.py 
