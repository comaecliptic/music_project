import re
import librosa
import librosa.display
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from pathlib import Path
from itertools import product


# Statistics functions
# time-domain
def mean_energy(x):
    # Mean squared signal value
    return np.mean(x**2)

def energy_entropy(x):
    # How its signal is the same over all the treck
    window = 10000
    subframes_energies = []
    for i in range(0, len(x)-window+1, window):
        temp = x[i:i+window]
        prob = sum(temp)/sum(x)
        subframes_energies.append(prob)
    subframes_energies = np.array(subframes_energies)
    return -sum(subframes_energies*np.log2(subframes_energies))

def ZCR(x):
    # How many times it goes through zero
    window = 100
    ZCRS = []
    for i in range(0, len(x)-window+1, window):
        temp = x[i:i+window]
        zero_cross = sum(librosa.zero_crossings(y=temp, pad=False))
        ZCRS.append(zero_cross)
    return np.mean(np.array(ZCRS)), np.std(np.array(ZCRS))

# frequency-domain
def spectral_centroids(x, sr):
    # Kinda frequency expectation
    specter = librosa.feature.spectral_centroid(x, sr=sr)[0]
    to_count = sklearn.preprocessing.minmax_scale(specter, axis=0)
    return np.mean(to_count), np.std(to_count)

def spectral_bandwidths(x, sr):
    #Kinda frequency variance
    to_count = librosa.feature.spectral_bandwidth(x, sr)
    print("done librosa thing")
    return np.mean(to_count), np.std(to_count)

def spectral_flatness(x):
    # How it's horizontal
    to_count = librosa.feature.spectral_flatness(x)[0]
    return np.mean(to_count), np.std(to_count)

def spectral_flux(x, sr):
    # How it's changing
    to_count = librosa.onset.onset_strength(x, sr)
    return np.mean(to_count), np.std(to_count)

def spectral_rolloff(x, sr):
    # Kinda enrichment of high frequencies (where lies 85% of energy?)
    to_count = librosa.feature.spectral_rolloff(x, sr=sr)[0]
    return np.mean(to_count), np.std(to_count)

# Main thing
def MFCC(x, sr):
    return np.array([np.mean(element) for element in librosa.feature.mfcc(x, sr=sr)])



#========= Main code===========

def change_name(name):
    """
    This is a little thing, don't even mind it.
    :param name:
    :return:
    """
    badsymbols = " )-(,;:"
    trantab = str.maketrans(badsymbols, "".join(["_" for i in badsymbols]))
    name = name.translate(trantab)
    for i in range(1, 5):
        name = name.replace("_" * i, "_")
    if name[0] == "_":
        name = name[1:]
    if name[-1] == "_":
        name = name[:-1]
    return name


def extract_features(file, dataframe):
    """
    Eztracts features from a given file
    :param file:
    :param dataframe:
    :return:
    """
    file = Path(file)
    sound = AudioSegment.from_mp3(file)
    dst = file.parents[0] / "wavs" / file.with_suffix(".wav").name
    sound.export(dst, format="wav")
    print(dst)
    print("exported")
    x, sr = librosa.load(dst)
    print("read")
    audio_name = change_name(file.stem)
    features_list = [audio_name, mean_energy(x)]  # energy_entropy(x)
    print("starting first set of functions, ", len(x))
    for func in (ZCR, spectral_flatness):
        print("-", func.__name__, " is processed")
        features_list += list(func(x))
    print("starting second set of functions")
    for func in (spectral_centroids, spectral_bandwidths, spectral_flux, spectral_rolloff):
        print("--", func.__name__, " is processed")
        features_list += list(func(x, sr))
    print("starting MFCC")
    features_list += list(MFCC(x, sr))
    with open("temp_data.csv", "a") as fl:
        fl.write(";".join([str(element) for element in features_list]) + "\n")
    with open("processed.txt", "a") as fl:
        fl.write(file.stem + "\n")
    row = pd.DataFrame([features_list], columns=dataframe.columns)
    new_df = dataframe.append(row, ignore_index=True)
    return new_df


def make_data_frame():
    column_names = ["track_name", "mean_energy"]
    for name in product(["ZCR", "spectral_flatness", "spectral_centroids", "spectral_bandwidths", "spectral_flux",
                         "spectral_rolloff"], ["_mean", "_std"]):
        column_names.append("".join(name))
    for i in range(1, 21):
        column_names.append(f'MFCC_{i}')
    df = pd.DataFrame(columns=column_names)
    return df


def main(folder):
    folder = Path(folder)
    wav_folder = folder / Path("wavs")
    wav_folder.mkdir(exist_ok=True)
    dataframe = make_data_frame()
    files = [element for element in folder.iterdir() if element.is_file()]
    files.sort(key=lambda element: element.stat().st_size)
    file = Path("temp_data.csv")
    if not file.is_file():
        with open(file, "w") as fl:
            fl.write(";".join(dataframe.columns) + "\n")
    file = Path("processed.txt") # This is for multiple runs, for not computing the same thing over and over again.
    processed = []
    if file.is_file():
        with open(file, "r") as fl:
            processed = [line.strip() for line in fl.readlines()]
    ps = 0
    for file in files:
        print(file, file.stat().st_size)
        if file.stem in processed:
            print("passing")
            continue
        ps += 1
        # if ps <= 1:
        #   continue
        dataframe = extract_features(file, dataframe)
    dataframe.to_csv("./Features.csv", index=False, sep=";")
    return dataframe


df = main("/home/dmitry/PycharmProjects/MachineLearning/dataset")