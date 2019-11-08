# Wave-Attention-Net

Implementation of the Wave-Attention-Net for audio source separation.

# Installation

## Requirements

The project is based on Python 2.7 and requires [libsndfile](http://mega-nerd.com/libsndfile/) to be installed.

```shell
sudo  apt-get  install  libsndfile-dev
```

Then, the following Python packages need to be installed:

```
ffmpeg
numpy==1.15.4
sacred==0.7.3
tensorflow-gpu==1.9.0
librosa==0.6.2
soundfile==0.10.2
scikits.audiolab==0.11.0
lxml==4.2.1
musdb==0.2.3
museval==0.2.0
google==2.0.1
protobuf==3.4.0
```

Alternatively to ``tensorflow-gpu`` the CPU version of TF, ``tensorflow`` can be used, if there is no GPU available.
All the above packages are also saved in the file ``requirements.txt`` located in this repository, so you can clone the repository and then execute the following in the downloaded repository's path to install all the required packages at once:

``pip install -r requirements.txt``

To recreate the figures from the paper, use functions in ``Plot.py``. The ``matplotlib<3.0`` package needs to be installed as well in that case.

### Download datasets

#### MUSDB18

Download the [full MUSDB18 dataset](https://sigsep.github.io/datasets/musdb.html) and extract it into a folder of your choice. It should have two subfolders: "test" and "train" as well as a README.md file.

#### CCMixter (only required for vocal separation experiments)

If you want to replicate the vocal separation experiments and not only the multi-instrument experiments, you also need to download the CCMixter vocal separation database from https://members.loria.fr/ALiutkus/kam/. Extract this dataset into a folder of your choice. Its main folder should contain one subfolder for each song.

### Set-up filepaths

Now you need to set up the correct file paths for the datasets and the location where source estimates should be saved.

Open the ``Config.py`` file, and set the ``musdb_path`` entry of the ``model_config`` dictionary to the location of the main folder of the MUSDB18 dataset.
Also set the ``estimates_path`` entry of the same ``model_config`` dictionary to the path pointing to an empty folder where you want the final source estimates of the model to be saved into.

If you use CCMixter, open the ``CCMixter.xml`` in the main repository folder, and replace the given file path tagged as ``databaseFolderPath`` with your path to the main folder of CCMixter.

## Training the  model overview

We give the command needed to start training them:

| Model name (from paper) | Description                                             | Separate vocals or multi-instrument? | Command for training                                |
| ----------------------- | ------------------------------------------------------- | ------------------------------------ | --------------------------------------------------- |
| U7                      | U-Net replication from prior work, audio-based MSE loss | Vocals                               | ``python Training.py with cfg.unet_spectrogram``    |
| U7a                     | Like U7, but with L1 magnitude loss                     | Vocals                               | ``python Training.py with cfg.unet_spectrogram_l1`` |
