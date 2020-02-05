import glob
import json
import subprocess
import librosa
import numpy as np

dfdc_train_wav_path = "./../dataset/fb_audio/"

def audio_altered(fake_path, fake_video, orig_path, orig_video):
    """Finds out if audio of fake_video was altered.

    # Arguments
        fake_path: fake mp4 video path name
        fake_video: fake mp4 video name
        orig_path: original mp4 video path name
        orig_video: original mp4 video name

    # Returns
        True - if audio of fake mp4 video was altered
        False - otherwise
    """
    fake_wav = fake_video[:-4] + '.wav'
    fake_wav_path = dfdc_train_wav_path + fake_wav
    try:
        # in case if .wav has already been extracted
        fake_data, fake_rate = librosa.load(fake_wav_path, sr=None)
    except FileNotFoundError:
        # extract fake_path audio
        # .wav audio format is used because librosa.load() doesn't work with .aac
        command = "./ffmpeg-git-amd64-static/ffmpeg -i %s -vn -f wav %s" % (fake_path, fake_wav_path)
        subprocess.run(command, shell=True) 

        try:
            fake_data, fake_rate = librosa.load(fake_wav_path, sr=None)
        except FileNotFoundError:
            # if fake video has no audio than its audio is not altered
            return False

    orig_wav = orig_video[:-4] + '.wav'
    orig_wav_path = dfdc_train_wav_path + orig_wav
    try:
        # in case if .wav has already been extracted
        orig_data, orig_rate = librosa.load(orig_wav_path, sr=None)
    except FileNotFoundError:
        # extract orig_path audio
        # .wav audio format is used because librosa.load() doesn't work with .aac
        command = "./ffmpeg-git-amd64-static/ffmpeg -i %s -vn -f wav %s" % (orig_path, orig_wav_path)
        subprocess.run(command, shell=True)

        try:
            orig_data, orig_rate = librosa.load(orig_wav_path, sr=None)
        except FileNotFoundError:
            # if original video has no audio but fake video does than audio is altered
            return True

    return fake_rate != orig_rate or not np.array_equal(fake_data, orig_data)