#
# import os
# import numpy as np
# from scipy.io import loadmat
# import einops
# import torch
# import random
# import pickle
# import mat73
# import scipy.io as scio
#
# def random_1D_seed(num):
#
#     rand_lists = []
#     for index in range(num):
#         grand_list = [i for i in range(62)]
#         random.shuffle(grand_list)
#         rand_tensor = torch.tensor(grand_list).view(1, 62)
#         rand_lists.append(rand_tensor)
#
#     rand_torch = torch.cat(tuple(rand_lists), 0)
#     return rand_torch
#
#
# def load_eeg_data(data_path):
#
#     label_path = "/data2/EEG_data/MPED/label.mat"
#
#     label = mat73.loadmat(label_path)["label"].astype(int)-1 # 30
#
#     X = scio.loadmat(data_path)["STFT"][0]
#
#     train_trials = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 22, 24, 26]
#     test_trials = [20, 21, 23, 25, 27, 28, 29]
#
#     trainData = np.array([])  # 存储每一个subject的数组
#     testData = np.array([])
#     trainLabel = np.array([])
#     testLabel = np.array([])
#
#     for i in range(len(label)):
#         if i % 15 == 0:
#             continue
#
#         if i in train_trials:
#             X_ = np.array(einops.rearrange(X[i], "w h c -> h w c"))
#             y_ = np.array([label[i],] * X_.shape[0])
#
#             if trainData.shape[0] == 0:
#                 trainData = X_
#                 trainLabel = y_
#             else:
#                 trainData = np.append(trainData, X_, axis=0)
#                 trainLabel = np.append(trainLabel, y_, axis=0)
#
#         if i in test_trials:
#             X_ = np.array(einops.rearrange(X[i], "w h c -> h w c"))
#             y_ = np.array([label[i],] * X_.shape[0])
#
#             if testData.shape[0] == 0:
#                 testData = X_
#                 testLabel = y_
#             else:
#                 testData = np.append(testData, X_, axis=0)
#                 testLabel = np.append(testLabel, y_, axis=0)
#
#     # 将数据进行重新组合
#     first_21_trials = trainLabel.size
#     emotionData = np.append(trainData, testData, axis=0)
#     emotionLabel = np.append(trainLabel, testLabel, axis=0)
#
#     return emotionData, emotionLabel, first_21_trials
#
#
# def load_mm_data(data_path):
#
#     ECG_features = scio.loadmat(data_path)["ECG_Feature"][0]
#     GSR_features = scio.loadmat(data_path)["GSR_Feature"][0]
#     RSP_features = scio.loadmat(data_path)["RSP_Feature"][0]
#
#     label_path = "/data2/EEG_data/MPED/label.mat"
#     label = mat73.loadmat(label_path)["label"].astype(int) - 1  # 30
#
#     train_trials = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 22, 24, 26]
#     test_trials = [20, 21, 23, 25, 27, 28, 29]
#
#     trainData = np.array([])  # 存储每一个subject的数组
#     testData = np.array([])
#     trainLabel = np.array([])
#     testLabel = np.array([])
#
#     for i in range(len(label)):
#         if i % 15 == 0:
#             continue
#
#         if i in train_trials:
#             X_ECG = np.array(einops.rearrange(ECG_features[i], "w h -> h w"))
#             X_GSR = np.array(einops.rearrange(GSR_features[i], "w h -> h w"))
#             X_RSP = np.array(einops.rearrange(RSP_features[i], "w h -> h w"))
#             # y_ = np.array([label[i],] * X_ECG.shape[0])
#
#             X_MM = np.concatenate((X_ECG, X_GSR, X_RSP), axis=1)
#
#             if trainData.shape[0] == 0:
#                 trainData = X_MM
#             else:
#                 trainData = np.append(trainData, X_MM, axis=0)
#
#         if i in test_trials:
#             X_ECG = np.array(einops.rearrange(ECG_features[i], "w h -> h w"))
#             X_GSR = np.array(einops.rearrange(GSR_features[i], "w h -> h w"))
#             X_RSP = np.array(einops.rearrange(RSP_features[i], "w h -> h w"))
#
#             X_MM = np.concatenate((X_ECG, X_GSR, X_RSP), axis=1)
#
#             if testData.shape[0] == 0:
#                 testData = X_MM
#             else:
#                 testData = np.append(testData, X_MM, axis=0)
#
#     # 将数据进行重新组合
#     emotionData = np.append(trainData, testData, axis=0)
#
#     return emotionData
# #
#
#
# def extend_normal(sample):
#     for i in range(len(sample)):
#
#         features_min = np.min(sample[i])
#         features_max = np.max(sample[i])
#         sample[i] = (sample[i] - features_min) / (features_max - features_min)
#     return sample
#
#
#
# def mm_normalized_data():
#
#     rawData_EEG = "/data2/EEG_data/MPED/STFT/"
#     rawData_MM = "/data2/EEG_data/MPED/GSR_RSP_ECG_features/"
#     normalizedData = "/data2/EEG_data/MPED/MM_Normalized/"
#     x_list, y_list, subject_list = [], [], []
#
#     for subject in os.listdir(rawData_EEG):
#         x_eeg, y_, dependent_index = load_eeg_data(os.path.join(rawData_EEG, str(subject)))
#         x_eeg = np.array(einops.rearrange(x_eeg, "w h c -> w (h c)"))
#
#         x_mm = load_mm_data(os.path.join(rawData_MM, str(subject)))
#         x_mm = np.log2(1000000*(abs(x_mm) + 1))  # 先按对数归一化，降低数据间的差异
#         x_MM = np.append(x_eeg, x_mm, axis=1)
#
#         x_ = extend_normal(x_MM)
#
#         dict = {'sample': x_, 'label': y_, 'dependent_index': dependent_index}
#         np.save(os.path.join(normalizedData, (str(subject) + ".npy")), dict)
#
#
#
# if __name__ == "__main__":
#
#     # save_normalized_data()
#     mm_normalized_data()



