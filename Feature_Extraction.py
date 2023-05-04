import matplotlib.pyplot as plt
import Config
import librosa
from librosa import display
import numpy as np
import os
import pickle
from TOOL_statistics_feature import feature_extraction, spectrogram


def Data_Acq(fileName):
    file = open(fileName, 'rb')
    sample = pickle.load(file, encoding='latin1')
    file.close()

    return sample


def Dc_normalize(sig_array):
    """Removes DC and normalizes to -1, 1 range"""
    sig_array_norm = sig_array.copy()
    sig_array_norm -= sig_array_norm.mean()
    sig_array_norm /= abs(sig_array_norm).max()
    return sig_array_norm


def Create_mel_spectrogram(data, sample_rate, n_mels=128, f_min=50, f_max=4000, nfft=2048, hop=512, show_function=False):
    S = librosa.feature.melspectrogram(y=data, sr=sample_rate, n_mels=n_mels, fmin=f_min, fmax=f_max,
                                       n_fft=nfft, hop_length=hop)
    S = librosa.power_to_db(S, ref=np.max)
    S = (S - S.min()) / (S.max() - S.min())

    if show_function:
        fig, ax = plt.subplots()
        img = librosa.display.specshow(S, x_axis='time',
                                       y_axis='mel', sr=sample_rate,
                                       fmax=f_max, ax=ax)
        plt.show()

    return S



def Feature_Extraction(file_dir):
    for data_dir in os.listdir(file_dir):
        # Load data
        data_dir = f"{file_dir}\\{data_dir}"
        samples = Data_Acq(data_dir)

        # 特征提取
        data = Dc_normalize(samples["signal"])  # 数据归一化

        # Mel spectrogram
        mel_spectrogram = Create_mel_spectrogram(data, Config.sample_rate, Config.n_mels,
                                                 Config.f_min, Config.f_max, Config.nfft, Config.hop)


        # acoustic_feature
        features, feature_names = feature_extraction(signal=data, sampling_rate=Config.sample_rate,
                                                         window=Config.nfft, step=Config.hop, deltas=False)

        features_mean = np.mean(features, axis=1)
        features_median = np.median(features, axis=1)
        features_std = np.std(features, axis=1)

        acoustic_feature = np.concatenate((features_mean, features_median, features_std))


        # Spectrogram
        spectrogram_feature, time_axis, freq_axis = spectrogram(signal=data,
                                                                sampling_rate=Config.sample_rate,
                                                                window=Config.nfft, step=Config.hop,
                                                                plot=False, show_progress=False)
        spectrogram_feature = (spectrogram_feature - spectrogram_feature.min()) / (spectrogram_feature.max() - spectrogram_feature.min())


        samples["statistics_feature"] = acoustic_feature
        samples["spectrogram"] = spectrogram_feature
        samples["mel_spectrogram"] = mel_spectrogram

        # save to the dir
        if Config.save_for_preprocessing_and_feature_extraction:
            with open(data_dir, 'wb') as f:
                pickle.dump(samples, f)

        print(f"{data_dir} over")


if __name__ == "__main__":
    Feature_Extraction(Config.preprocessed_dir_savesamples)
