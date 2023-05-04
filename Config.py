# file path
raw_dir = "Data\\ICBHI_final_database"
preprocessed_dir_savesamples = "Data\\preprocessing"
savedir_train_and_test = "Data\\"

sample_keys = ["signal", "label", "mel_spectrogram", "statistics_feature", "spectrogram"]

# Label Set
normal = 0
crackle = 1
wheezes = 2
both = 3


diagnosis_file_dir = "Data\\dignosis.txt"
Healthy = 0
URTI = 1
Asthma = 2
COPD = 3
LRTI = 4
Bronchiectasis = 5
Pneumonia = 6
Bronchiolitis = 7




# Save_Function
save_for_preprocessing_and_feature_extraction = True


# signal
sample_rate = 4000
# padding_mode: (str) zero,  sample;
# sample padding ref to the paper "Lung Sound Classification Using Snapshot Ensemble of Convolutional Neural Networks"
padding_mode = "zero"


# filter
filter_lowcut = 50
filter_highcut = 1800
filter_order = 5
filter_btype = "bandpass"           # filter_btype: (str) highpass, bandpass


# samples set
respiratory_cycle = 5   # The length of data as input
overlap = 5


# Mel spectrogram
n_mels = 128
f_min = 50
f_max = sample_rate // 2
nfft = 256
hop = nfft // 2

# For train and test
Num_classes = 4
num_k = 5                   # N-fold cross validation
device = "cuda"             # device
lr = 0.0001                 # learn rate
weight_decay = 0.0001       # l2 normalization
EPOCH = 300
batch_size = 64



