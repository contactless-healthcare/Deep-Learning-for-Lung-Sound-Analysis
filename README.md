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

Please cite below paper if the code was used in your research or development.
    
    @article{huang2023deep,
        title={Deep learning-based lung sound analysis for intelligent stethoscope},
        author={Huang, Dong-Min and Huang, Jia and Qiao, Kun and Zhong, Nan-Shan and Lu, Hong-Zhou and Wang, Wen-Jin},
        journal={Military Medical Research},
        volume={10},
        number={1},
        pages={44},
        year={2023},
        publisher={Springer}
    }


