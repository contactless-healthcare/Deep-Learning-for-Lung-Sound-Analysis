import os
import shutil
import numpy as np
import scipy.signal as signal
import librosa
import Config
import pickle
import matplotlib.pyplot as plt
from scipy.io.wavfile import write


def Filter_Denoised(raw_audio, sample_rate, filter_order, filter_lowcut, filter_highcut, btype="bandpass"):
    b, a = 0.0, 0.0
    if btype == "bandpass":
        b, a = signal.butter(filter_order, [filter_lowcut/(sample_rate/2), filter_highcut/(sample_rate/2)], btype=btype)

    if btype == "highpass":
        b, a = signal.butter(filter_order, filter_lowcut, btype=btype, fs=sample_rate)


    audio = signal.lfilter(b, a, raw_audio)

    return audio



def Padding(data, sample_rate, respiratory_cycle, padding_more):

    if len(data) == (sample_rate * respiratory_cycle):
        return data  # the duration of data is equal to the length of the demand
    else:
        padding = None
        if padding_more == "zero":
            padding = np.zeros((data.shape[0]))
        elif padding_more == "sample":
            padding = data.copy()

        while True:  # 反复拼接
            data = np.concatenate([data, padding])
            if len(data) > (sample_rate * respiratory_cycle):
                data = data[:int(sample_rate * respiratory_cycle)]

            if len(data) == (sample_rate * respiratory_cycle):
                return data



def Segmentation(audio, label_dir, sample_rate, respiratory_cycle, overlap, padding_more, show_function=False):
    samples = {
        "signal": [],
        "label": [],
    }

    filelabel = open(f"{label_dir}", "r")
    res = filelabel.readlines()
    filelabel.close()

    for i, cur in enumerate(res):
        cur = cur.strip("\n")
        lStart, lEnd, lcrackle, lwheeze = cur.split("\t")
        lStart = float(lStart)
        lEnd = float(lEnd)
        lcrackle = int(lcrackle)
        lwheeze = int(lwheeze)

        # Label Construction
        label = None  # normal - 0, crackle - 1, wheezes - 2, both - 3
        if lcrackle == 0 and lwheeze == 0:
            label = Config.normal
        elif lcrackle == 1 and lwheeze == 0:
            label = Config.crackle
        elif lcrackle == 0 and lwheeze == 1:
            label = Config.wheezes
        elif lcrackle == 1 and lwheeze == 1:
            label = Config.both

        # Data Construction
        while lStart < lEnd:
            temp_start = lStart
            temp_end = temp_start + respiratory_cycle
            if temp_end > lEnd:
                temp_end = lEnd

            if (lEnd - lStart) < (Config.respiratory_cycle // 2):
                break

            temp_start = int(temp_start * sample_rate)
            temp_end = int(temp_end * sample_rate)

            temp_data = Padding(audio[temp_start:temp_end], sample_rate, respiratory_cycle, padding_more)

            if show_function:
                plt.plot(temp_data)
                plt.show()

            samples["signal"].append(temp_data)
            samples["label"].append(label)

            lStart += overlap

    return samples


def Preprocessing(dir, preprocessed_dir_savesamples):
    print('start Preprcessing')

    # Make dir
    if os.path.exists(f"{preprocessed_dir_savesamples}") and Config.save_for_preprocessing_and_feature_extraction:
        shutil.rmtree(f"{preprocessed_dir_savesamples}")
        os.makedirs(f"{preprocessed_dir_savesamples}")
    else:
        os.makedirs(f"{preprocessed_dir_savesamples}")


    for file_name in os.listdir(dir):
        if ".wav" not in file_name:
            continue

        # path
        data_dir = f"{dir}\\{file_name}"
        label_dir = f"{dir}\\{file_name.split('.')[0]}.txt"

        # load data
        raw_audio, sample_rate = librosa.load(path=data_dir, sr=Config.sample_rate)

        # filter
        audio_data = Filter_Denoised(raw_audio, sample_rate, Config.filter_order,
                                     Config.filter_lowcut, Config.filter_highcut, btype=Config.filter_btype)


        # Segmentation
        samples = Segmentation(audio_data, label_dir, sample_rate, Config.respiratory_cycle, Config.overlap, Config.padding_mode)
        if samples["signal"] == []:
            continue

        # Save to the preprocessed_dir_savesamples
        if Config.save_for_preprocessing_and_feature_extraction:
            for i in range(len(samples["signal"])):
                save_dir = preprocessed_dir_savesamples + '\\' + file_name.split('.')[0] + f"_{i}.dat"
                temp = {
                    "signal": samples["signal"][i],
                    "label": samples["label"][i],
                }
                with open(save_dir, 'wb') as f:
                    pickle.dump(temp, f)

        print(f"{file_name} over")


if __name__ == '__main__':
    Preprocessing(Config.raw_dir, Config.preprocessed_dir_savesamples)



