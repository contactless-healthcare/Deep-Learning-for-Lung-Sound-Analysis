import os
import numpy as np
import Config
import pickle
from sklearn.model_selection import StratifiedKFold
import shutil



def Data_Acq(fileName):
    with open(fileName, 'rb') as f:
        sample = pickle.load(f)
    return sample



def CrossValidation(dir, savedir, cross_subject_flag=True):
    dataDir = os.listdir(dir)

    # Cross_subject
    subjectList = []

    if cross_subject_flag:
        for tempDir in dataDir:
            tempDir = tempDir[:3]
            if tempDir not in subjectList:
                subjectList.append(tempDir)
    else:
        for tempDir in dataDir:
            if tempDir not in subjectList:
                subjectList.append(tempDir)

    # 交叉验证
    seed = 2
    np.random.seed(seed)
    num_k = Config.num_k
    kfold = StratifiedKFold(n_splits=num_k, shuffle=True, random_state=seed)

    for count, (trainIndex, testIndex) in enumerate(kfold.split(subjectList, np.zeros((len(subjectList),)))):
        print(count, "Start")
        if not os.path.exists(f"{savedir}\\Fold_{count}"):
            os.makedirs(f"{savedir}\\Fold_{count}")
        else:
            shutil.rmtree(f"{savedir}\\Fold_{count}")
            os.makedirs(f"{savedir}\\Fold_{count}")

        # Train
        with open(f"{savedir}\\Fold_{count}\\train_list.txt", "w") as train_file:
            for tempIndex in trainIndex:
                subjectIndex = subjectList[tempIndex]
                for tempDir in dataDir:
                    if tempDir[:3] == subjectIndex and cross_subject_flag:  # 匹配id，选择数据
                        train_file.write(f"{dir}\\{tempDir}\n")

                    if tempDir == subjectIndex and cross_subject_flag is False:
                        train_file.write(f"{dir}\\{tempDir}\n")


        print(count, "Train  over")

        # Test
        with open(f"{savedir}\\Fold_{count}\\test_list.txt", "w") as test_file:
            for tempIndex in testIndex:
                subjectIndex = subjectList[tempIndex]
                for tempDir in dataDir:
                    if tempDir[:3] == subjectIndex and cross_subject_flag:  # 匹配id，选择数据
                        test_file.write(f"{dir}\\{tempDir}\n")

                    if tempDir == subjectIndex and cross_subject_flag is False:
                        test_file.write(f"{dir}\\{tempDir}\n")

        print(count, "Test  over")
        print(count, "Over")



if __name__ == '__main__':
    CrossValidation(Config.preprocessed_dir_savesamples, Config.savedir_train_and_test, cross_subject_flag=True)
