import os
import Config
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader
import numpy as np
from Model.Deep_learning import DatasetLoad, CNN_Model, Train_one_epoch, Evaluate, \
    Get_mean_and_std, Recod_and_Save_Train_Detial, FNN_Model


import warnings
warnings.filterwarnings("ignore")


def Load_data(dir):
    data_dir = []
    with open(dir, 'r') as file:
        line = file.readline()
        while line:
            data_dir.append(line.strip())
            line = file.readline()

    return data_dir


def Main(feature_name, model_name):
    device = torch.device(Config.device if torch.cuda.is_available() else "cpu")
    if not os.path.exists(f"{Config.savedir_train_and_test}\\Record"):
        os.makedirs(f"{Config.savedir_train_and_test}\\Record")


    senList, speList, aeList, hsList, accList, confusion_matrix_List  = [], [], [], [], [], []
    for count in range(Config.num_k):
        # Data Load
        train_data_dir_list = Load_data(f"{Config.savedir_train_and_test}\\Fold_{count}\\train_list.txt")
        test_data_dir_list = Load_data(f"{Config.savedir_train_and_test}\\Fold_{count}\\test_list.txt")

        # Normalization using mean & std
        mean, std = Get_mean_and_std(DatasetLoad(train_data_dir_list, feature_name))
        input_transform = None
        if "statistics" in feature_name:
            input_transform = [mean, std]
        else:
            input_transform = Compose([ToTensor(), Normalize(mean, std)])
        print(f"Fold: {count}, mean: {mean}, std: {std}")

        # Train & Test
        train_dataloader = DataLoader(
            dataset=DatasetLoad(train_data_dir_list, feature_name, input_transform),
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        test_data = DataLoader(
            dataset=DatasetLoad(test_data_dir_list, feature_name, input_transform),
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Model load
        model = None
        if model_name == "CNN":
            model = CNN_Model(Config.Num_classes).to(device)
        elif model_name == "FNN":
            model = FNN_Model(Config.Num_classes).to(device)

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)

        # train & test
        train_recorder, test_recorder = [], []
        for epoch in range(Config.EPOCH):
            train_info = Train_one_epoch(model, optimizer, train_dataloader, device, epoch)
            train_recorder.append(train_info)

            test_info = Evaluate(model, test_data, device)
            test_recorder.append(test_info)

            print(f"Epoch: {epoch+1}/{Config.EPOCH}: Train Loss: {train_info[0]:.6f}, ACC: {train_info[1]:.6f}, "
                  f"SEN: {train_info[2]:.6f}, SPE: {train_info[3]:.6f}, AE: {train_info[4]:.6f}")
            print(f"Epoch: {epoch + 1}/{Config.EPOCH}: Test  Loss: {test_info[0]:.6f}, ACC: {test_info[1]:.6f}, "
                  f"SEN: {test_info[2]:.6f}, SPE: {test_info[3]:.6f}, AE: {test_info[4]:.6f}\n")

            # save the weights
            if (epoch+1) % 30 == 0 and epoch > 0:
                print(test_info[-2])
                print(test_info[-1])
                torch.save(model.state_dict(),  f"{Config.savedir_train_and_test}\\Record\\Fold_{count}_model_weights_Epoch_{epoch}.pth")

        # Save the final model
        torch.save(model.state_dict(), f"{Config.savedir_train_and_test}\\Record\\Fold_{count}_model_weights_Epoch_Final.pth")

        # Plot picture & save the training record
        Recod_and_Save_Train_Detial(count, Config.savedir_train_and_test,
                                    train_recorder, test_recorder, show_fucntion=False)

        # Test
        test_info = Evaluate(model, test_data, device)
        print(f"Fold: {count}\n"
              f"Final Test  Loss: {test_info[0]:.6f}, ACC: {test_info[1]:.6f}, "
              f"SEN: {test_info[2]:.6f}, SPE: {test_info[3]:.6f}, AE: {test_info[4]:.6f}")
        print(test_info[-2], "\n", test_info[-1], "\n\n\n")

        accList.append(test_info[1])
        senList.append(test_info[2])
        speList.append(test_info[3])
        aeList.append(test_info[4])
        hsList.append(test_info[5])
        confusion_matrix_List.append(test_info[6])

    for i, matrix in enumerate(confusion_matrix_List):
        print(i)
        print(matrix, "\n")

    print("acc:", accList)
    print("sen：", senList)
    print("spe:", speList)
    print("ae :", aeList)
    print("he :", hsList)
    print("mean acc:", np.mean(accList))
    print("mean sen:", np.mean(senList))
    print("mean spe:", np.mean(speList))
    print("mean ae :", np.mean(aeList))
    print("mean he :", np.mean(hsList))


if __name__ == '__main__':
    feature_name_list = ["spectrogram", "mel_spectrogram", "statistics_feature"]
    feature_name = feature_name_list[1]

    model_list = ["CNN", "FNN"]
    mode_name = model_list[0]

    Main(feature_name, mode_name)



