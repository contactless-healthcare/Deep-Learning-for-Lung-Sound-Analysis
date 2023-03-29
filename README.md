# deep-learning-for-lung-sound-analysis

This is code repository for the paper "Deep learning in wireless stethoscope-based lung sound analysis".

It is developed for lung sound recognition based on the Pytorch, Librosa [1], and pyaudioanalysis [2], 
It modularizes the operation of deep learning, including the preprocessing, feature extraction, dataset splitting and classifier training.


For the demonstration, we uses ICBHI 2017 dataset [3] as the example. 

There are six major .py files in this repo.

  1. Config.py: It controls the required parameters in each process.
  2. Preprocessing.py: It contains the basic noise reduction functions and data segmentation.
  3. Feature extraction.py: It transforms the lung sound signal into an input suitable for the model, such as statistical features and spectrogram.
  4. Data splitting.py: It divided the whole dataset into training and testing sets (in the subject-wise way).
  5. Classifier_training_and_testing: It is used to train and test the model.
  6. Model.deep learning.py: It stores the definitions of the model and the training and testing procedures.


To build a model for lung sound analysis, the 2-5 .py files should be performed sequentially, 
and the Config.py and Model.deep learning.py is applied for custom Settings.



[1] McFee, B., Raffel, C., Liang, D., Ellis, D. P., McVicar, M., Battenberg, E., & Nieto, O. (2015, July). librosa: Audio and music signal analysis in python. In Proceedings of the 14th python in science conference (Vol. 8, pp. 18-25).

[2] Giannakopoulos, T. (2015). pyaudioanalysis: An open-source python library for audio signal analysis. PloS one, 10(12), e0144610.

[3] Rocha, B. M., Filos, D., Mendes, L., Serbes, G., Ulukaya, S., Kahya, Y. P., ... & De Carvalho, P. (2019). An open access database for the evaluation of respiratory sound classification algorithms. Physiological measurement, 40(3), 035001.


