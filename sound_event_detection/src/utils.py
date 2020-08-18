from pydub import AudioSegment
import json
import os
import numpy as np
import librosa

def video_to_wav(file, save_directory='../wavs/', video_format='mp4', ifsave=1):
    audio = AudioSegment.from_file(file, format=video_format)
    if ifsave:
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        audio.export(save_directory+file.split('/')[-1][:-4] + '.wav', format="wav")
    else:
        return audio

def get_msecs(time_str, separator=':'):
    time_list = time_str.split(separator)
    msecs = float(time_list[2])*1000 + float(time_list[1])*60*1000 + float(time_list[0])*3600*1000
    return int(msecs)

def get_data_chunk(wav_file, json_data):
    labeled_wav_list = []
    for descr in json_data['sound_results']:
        st_time = get_msecs(descr['start_time'])
        end_time = get_msecs(descr['end_time'])
        label = descr['sound_type']
        labeled_wav_list.append((wav_file[st_time:end_time+1], label, json_data['file_name'][:-4], st_time, end_time))
    return labeled_wav_list

def get_wav(file):
    audio = AudioSegment.from_file(file, format="wav")
    return (audio, file.split('/')[-1])

def extract_features(X, sr, delta=False):
    mfccs = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40)
    if delta:
        mfccs = librosa.feature.delta(mfccs)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs
    
def read_audio_file(file_path, offset=0, duration=5):
    data, sr = librosa.load(file_path, res_type='kaiser_fast', sr=None, offset=offset)#, duration=duration)
    return data, sr