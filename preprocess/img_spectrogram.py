import os
import glob
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

SAMPLERATE = 48000
PATH_CSV = '../input/'
PATH_FLAC_TRAIN = '../input/train'

def cut_train(df:pd.DataFrame, data: np.array, sr: int, idx:int, f_min: float, f_max: float)-> np.array:
    """
    df: data,
    data: array flac
    sr: samplerate
    idx: we have duplicate in data(one file flac, diff part in file) and i take all by index df
    f_min/f_max: freq flac

    hoto: 
        find min/max time convert to index
        time mean for all 2.6 sec
        make 6 sec bin
        cuts and convert to melspec
    return mel_spec    
    """

    idx_min = df.iloc[idx].t_min * sr
    idx_max = df.iloc[idx].t_max * sr

    add = 3 * sr     
    center = int(idx_min + idx_max) /2
    
    idx_min = center - add
    if idx_min < 0:
        idx_min = 0
    idx_max = center + add       
    if idx_max > 2880000:
        idx_max = 2880000    
    
    cut_data = data[int(idx_min):int(idx_max)]   
    mel_spec = librosa.feature.melspectrogram(cut_data,
                                              n_fft=2048,
                                              hop_length=512,
                                              sr=sr,
                                              fmin=f_min,
                                              fmax=f_max,
                                              power=1.5
                                             )
    mel_spec = librosa.power_to_db(mel_spec).astype(np.float32)    
    return mel_spec

def cut_test(data: np.array, sr: int, f_min: float, f_max: float)->np.array:
    """
    data: array flac
    sr: samplerate    
    f_min/f_max: freq flac from df

    hoto:
        no time but we have bin 6 sec for train
        bins flac by 6 sec
        one file(flac) get 10 bins
        convert each to mel

    return dim (10, 128, 563)    
    
    """

    temp = []    
    l = sr * 6
    bins = len(data) / l
    for i in range(int(bins)):
        if (i+1) * l > len(data):
            d = data[len(data) - l : len(data)]            
        else:
            d = data[i * l: (i+1) * l]        
        mel_spec = librosa.feature.melspectrogram(d,
                                                  n_fft=2048,
                                                  hop_length=512,
                                                  sr=sr,
                                                  fmin=f_min,
                                                  fmax=f_max,
                                                  power=1.5
                                                 )
        mel_spec = librosa.power_to_db(mel_spec).astype(np.float32)
        temp.append(mel_spec)
    return temp


def gen_mel_img(df: pd.DataFrame, recording_id: str, idx:int, path_save: str,f_min: float, f_max: float, test: bool= None)->None:
    """
    make img Mel Spectrogram    
    and  save npy    
    default param: n_mels = 128
    for test recording_id is path to file, but train is only name without expansion file (flac)
    idx: take all recording_id by index
    """

    if test:
        data, sr = librosa.load(recording_id, sr = SAMPLERATE)
        file_name = os.path.basename(recording_id)
        recording_id = file_name.split('.')[0]
    else:
        data, sr = librosa.load(os.path.join(PATH_FLAC_TRAIN, recording_id +'.flac'), sr = SAMPLERATE)

    if test:
        # dim (10, 128, 563) we cat test for 10 bin
        mel_spec = cut_test(data, sr, f_min, f_max)
    else:
        mel_spec = cut_train(df, data, sr, idx, f_min, f_max)  
    np.save(os.path.join(path_save, f'{recording_id}.npy'), mel_spec)


if __name__ == "__main__":
    tp = pd.read_csv(os.path.join(PATH_CSV, 'train_tp.csv'))
    fp = pd.read_csv(os.path.join(PATH_CSV, 'train_fp.csv'))
    df = pd.concat([tp, fp])
    
    # find min/max freq flacs and scale
    f_min, f_max = df.f_min.min() * 0.8, df.f_max.max() * 1.2

    path_save = '../input/train_npy'    
    Parallel(n_jobs = -1)(delayed(gen_mel_img)(df,
                                                name_id,
                                                idx,
                                                path_save,
                                                f_min,
                                                f_max,
                                                test = None
                                                )
                                                for idx, name_id in tqdm(enumerate(df.recording_id.values))
                                                )# time 6.07, 1.3Gb
    # path_save = '../input/test_npy'
    # test_path = glob.glob('../input/test/*.flac')
    # Parallel(n_jobs = -1)(delayed(gen_mel_img)(df,
    #                                             name_id,
    #                                             idx,
    #                                             path_save,
    #                                             f_min,
    #                                             f_max,
    #                                             True
    #                                             )
    #                                             for idx, name_id in tqdm(enumerate(test_path))
    #                                             ) # time 4.10, 5.7Gb


    