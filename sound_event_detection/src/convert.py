from pydub import AudioSegment
import tqdm
import json
import os
import statistics
import argparse
from utils import video_to_wav, get_msecs, get_data_chunk, get_wav

def sound_extraction(args):
    video_list = [f for f in  os.listdir('../'+args.video_dir) if f.endswith('.'+args.video_format)]

    print("Audio extraction from the videos:\n", video_list)
    print("Conversion format:", args.video_format, '-> wav')
    print('Conversion...')
    
    print("Saving files to", '../'+args.wav_dir, '...')
    for video in tqdm.tqdm(video_list):
        video_to_wav('../'+args.video_dir+video, save_directory='../'+args.wav_dir, video_format=args.video_format)
    print('Conversion finished!')

def get_audio_file_paths(prefix='../wavs/', name='AnotherMissOh', suffix='', format='wav', min_num=1, max_num=19):
    file_paths = []
    for i in tqdm.tqdm(range(min_num, max_num)):
        zero = ''
        if i < 10: 
            zero += '0'
        file_paths.append(prefix + name + zero + str(i) + suffix + '.' + format)
    return file_paths

def get_labels_file_paths(prefix='../data/AnotherMissOh_Sound/', name='AnotherMissOh', suffix='_sound', format='json', min_num=1, max_num=19):
    file_paths = []
    for i in tqdm.tqdm(range(min_num, max_num)):
        zero = ''
        if i < 10: 
            zero += '0'
        file_paths.append(prefix + name + zero + str(i) + suffix + '.' + format)
    return file_paths

def pre_proc(args):
    labels_file_paths = get_labels_file_paths()
    labels_list = []
    
    print("Loading json label files...")
    for path in labels_file_paths:
        with open(path) as js_file:
            js_data = json.load(js_file)
            print("JS_data of ", js_data['file_name'], 'is loaded')
            labels_list.append(js_data)

    audio_file_paths = get_audio_file_paths(prefix='../'+args.wav_dir, format='wav')
    audio_list = []
    
    print("Loading wav files...")
    for path in audio_file_paths:
        audio_list.append(get_wav(path))
        print(audio_list[len(audio_list)-1][1])
    
    print("Dividing whole videos into chunks with GT labels...")
    all_labeled_wavs = []
    for it in  tqdm.tqdm(range(len(audio_list))):
        lbl_wavs = get_data_chunk(audio_list[it][0], labels_list[it])
        all_labeled_wavs.append(lbl_wavs)

    counter = 0
    lens = []
    folder = './pre_proc/'
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    print("Saving GT chunks into ./pre_proc folder for further feature extraction...")
    for entry in tqdm.tqdm(all_labeled_wavs):
        for sample in entry:
            if len(sample[0]) < 1000:
                continue
            sample[0].export(folder + sample[2]+'_' + '"' + sample[1].replace("/", "-") + '"' + '_' + str(counter)+'.wav', format='wav')
            counter += 1
            lens.append(len(sample[0]))
    print("Dataset size: ", counter)
    print("Min-max-median lengths of samples (ms): {}-{}-{}".format(min(list(lens)), max(list(lens)), statistics.median(lens)))

def parse_args():
	'''
	Parses the arguments.
	'''
	parser = argparse.ArgumentParser(description="Run data preprocessing.")
	parser.add_argument('--wav_dir', nargs='?', default='wavs',
	                    help='Directory for saving audios (wav) extracted from videos.')
	parser.add_argument('--video_dir', nargs='?', default='data/AnotherMissOh_High',
	                    help='Path to the directory with videos.')
	parser.add_argument('--video_format', nargs='?', default='mp4',
	                    help='Format of the videos (def:mp4).')
	return parser.parse_args()

if __name__ == "__main__": 
    args = parse_args()
    if args.video_dir[-1] != '/':
        args.video_dir += '/'
    if args.wav_dir[-1] != '/':
        args.wav_dir += '/'

    sound_extraction(args)
    pre_proc(args)

