#-------------------------------------
# SEED 数据的预处理代码 并将SEED的数据打包到npy文件中
# Date: 2022.11.2
# Author: Ming Jin
# All Rights Reserved
#-------------------------------------


import os
import numpy as np
from scipy.io import loadmat
import einops
import torch
import random
import pickle
from sklearn import svm
# from einops import rearrange, reduce, repeat



def load_data_3fold(data_path):

    clip1 = None
    clip2 = None

    emotionData = np.array([])  # 存储每一个subject的数组
    emotionLabel = np.array([])

    data = np.load(data_path)
    X = pickle.loads(data['data'])
    y = pickle.loads(data['label'])

    x_fold1 = np.array([])
    y_fold1 = np.array([])

    x_fold2 = np.array([])
    y_fold2 = np.array([])

    x_fold3 = np.array([])
    y_fold3 = np.array([])

    for i in range(45):

        metaData = np.array(X[i])
        metaLabel = np.array(y[i])
        metaLabel = metaLabel.astype(int)

        if i%15 in [0, 1, 2, 3, 4]:
            if x_fold1.shape[0] == 0:
                x_fold1 = metaData
                y_fold1 = metaLabel
            else:
                x_fold1 = np.append(x_fold1, metaData, axis=0)
                y_fold1 = np.append(y_fold1, metaLabel, axis=0)

        if i%15 in [5, 6, 7, 8, 9]:
            if x_fold2.shape[0] == 0:
                x_fold2 = metaData
                y_fold2 = metaLabel
            else:
                x_fold2 = np.append(x_fold2, metaData, axis=0)
                y_fold2 = np.append(y_fold2, metaLabel, axis=0)

        if i%15 in [10, 11, 12, 13, 14]:
            if x_fold3.shape[0] == 0:
                x_fold3 = metaData
                y_fold3 = metaLabel
            else:
                x_fold3 = np.append(x_fold3, metaData, axis=0)
                y_fold3 = np.append(y_fold3, metaLabel, axis=0)

    clip1 = y_fold1.size
    clip2 = y_fold1.size + y_fold2.size

    emotionData = np.append(x_fold1, x_fold2, axis=0)
    emotionLabel = np.append(y_fold1, y_fold2, axis=0)
    emotionData = np.append(emotionData, x_fold3, axis=0)
    emotionLabel = np.append(emotionLabel, y_fold3, axis=0)


    return emotionData, emotionLabel, clip1, clip2


def eeg_eye_data_3fold(eegData, eyeData, mmData):
    """
    讲脑电数据与眼动数据组织到一个向量中，方便后续的处理
    将重新组织的数据保存成npy格式，方便后续的使用
    :return:
    """
    x_list, y_list, subject_list = [], [], []

    for subject in os.listdir(eegData):
        x_eeg, y_eeg, clip1, clip2 = load_data_3fold(os.path.join(eegData, str(subject)))

        # eye movement data
        x_eye, y_eye, eyeclip1, eyeclip2 = load_data_3fold(os.path.join(eyeData, str(subject)))
        x_eye_log = np.log(x_eye+1) # Reduced discrepancies between eye movement data

        assert y_eeg.all() == y_eye.all() # 判断两组数据一一对应

        x_ = np.append(x_eeg, x_eye_log, axis=1)

        x_ = extend_normal(x_) # normalize all data

        # 将预处理文件保存到字典中，方便下次使用时直接加载
        normalizedDataPath = os.path.join(mmData)
        dict = {'sample': x_, 'label': y_eeg, 'clip1': clip1, 'clip2': clip2}
        np.save(os.path.join(normalizedDataPath, (str(subject) + ".npy")), dict)

        # # 读取保存好的EEG数据
        # dict_load = np.load(os.path.join(normalizedDataPath,(str(subject)+".npy")), allow_pickle=True)
        # sample_ = dict_load[()]['sample']
        # label_ = dict_load[()]['label']
        #
        #
        # x_list.append(x_)
        # y_list.append(y_)
        # subject_list.append(str(subject))


########################################################
####### 取随机数
########################################################
def random_1D_seed(num):

    rand_lists = []
    for index in range(num):
        grand_list = [i for i in range(62)]
        random.shuffle(grand_list)
        rand_tensor = torch.tensor(grand_list).view(1, 62)
        rand_lists.append(rand_tensor)

    rand_torch = torch.cat(tuple(rand_lists), 0)
    return rand_torch

def extend_normal(sample):
    """
    对训练集和测试集进行归一化
    对于EEG和aux数据进行分开归一化，考虑到两种数据存在较大差异
    :param sample:
    :return:
    """
    for i in range(len(sample)):

        EEG_features_min = np.min(sample[i][:310])
        EEG_features_max = np.max(sample[i][:310])

        aux_features_min = np.min(sample[i][310:])
        aux_features_max = np.max(sample[i][310:])

        sample[i][:310] = (sample[i][:310] - EEG_features_min) / (EEG_features_max - EEG_features_min)
        sample[i][310:] = (sample[i][310:] - aux_features_min) / (aux_features_max - aux_features_min)

    return sample


# def log_normal(sample):
#     """
#     对训练集和测试集进行归一化
#     :param sample:
#     :return:
#     """
#     for i in range(len(sample)):
#
#         features_min = np.min(sample[i])
#         features_max = np.max(sample[i])
#         sample[i] = (sample[i] - features_min) / (features_max - features_min)
#     return sample


# 按session对数据进行划分
# def load_data(data_path, session):
#
#     emotionData = np.array([])  # 存储每一个subject的数组
#     emotionLabel = np.array([])
#     first_10_trials = 0
#
#     data = np.load(data_path)
#     X = pickle.loads(data['data'])
#     y = pickle.loads(data['label'])
#
#     for i in range(15):
#         index = i + 15 * (int(session) - 1)
#
#         metaData = np.array(X[index])
#         metaLabel = np.array(y[index])
#         metaLabel = metaLabel.astype(int)
#
#         if emotionData.shape[0] == 0:
#             emotionData = metaData
#             emotionLabel = metaLabel
#         else:
#             emotionData = np.append(emotionData, metaData, axis=0)
#             emotionLabel = np.append(emotionLabel, metaLabel, axis=0)
#
#         if i == 9:
#             first_10_trials = emotionLabel.size
#     return emotionData, emotionLabel, first_10_trials




# def eeg_eye_data():
#     """
#     讲脑电数据与眼动数据组织到一个向量中，方便后续的处理
#     将重新组织的数据保存成npy格式，方便后续的使用
#     :return:
#     """
#     eegData = "/data2/EEG_data/SEED5/EEG_DE_features/"
#     eyeData = "/data2/EEG_data/SEED5/Eye_movement_features/"
#
#     normalizedData = "/data2/EEG_data/SEED5/Normal_eeg_eye/" # 存储合并的两种数据
#     session = "3"  # or 2,3
#     x_list, y_list, subject_list = [], [], []
#
#     for subject in os.listdir(eegData):
#         x_eeg, y_eeg, eeg_dependent_index = load_data_3fold(os.path.join(eegData, str(subject)))
#         # x_eeg = extend_normal(x_eeg)
#
#         # 加入眼动数据，并将eeg和eye数据整合到一起
#         x_eye, y_eye, eye_dependent_index = load_data_3fold(os.path.join(eyeData, str(subject)))
#         x_eye_log = np.log(x_eye+1) # 先按对数归一化，降低数据间的差异
#         # x_eye = extend_normal(x_eye_log)
#
#         assert y_eeg.all() == y_eye.all() # 判断两组数据一一对应
#
#         x_ = np.append(x_eeg, x_eye_log, axis=1)
#
#         x_ = extend_normal(x_)
#
#         # 将预处理文件保存到字典中，方便下次使用时直接加载
#         normalizedDataPath = os.path.join(normalizedData, session)
#         dict = {'sample': x_, 'label': y_eeg, 'dependent_index': eeg_dependent_index}
#         np.save(os.path.join(normalizedDataPath, (str(subject) + ".npy")), dict)
#
#         # # 读取保存好的EEG数据
#         # dict_load = np.load(os.path.join(normalizedDataPath,(str(subject)+".npy")), allow_pickle=True)
#         # sample_ = dict_load[()]['sample']
#         # label_ = dict_load[()]['label']
#         #
#         #
#         # x_list.append(x_)
#         # y_list.append(y_)
#         # subject_list.append(str(subject))


def return_coordinates():

    m1 = [(-2.285379, 10.372299, 4.564709),
          (0.687462, 10.931931, 4.452579),
          (3.874373, 9.896583, 4.368097),
          (-2.82271, 9.895013, 6.833403),
          (4.143959, 9.607678, 7.067061),

          (-6.417786, 6.362997, 4.476012),
          (-5.745505, 7.282387, 6.764246),
          (-4.248579, 7.990933, 8.73188),
          (-2.046628, 8.049909, 10.162745),
          (0.716282, 7.836015, 10.88362),
          (3.193455, 7.889754, 10.312743),
          (5.337832, 7.691511, 8.678795),
          (6.842302, 6.643506, 6.300108),
          (7.197982, 5.671902, 4.245699),

          (-7.326021, 3.749974, 4.734323),
          (-6.882368, 4.211114, 7.939393),
          (-4.837038, 4.672796, 10.955297),
          (-2.677567, 4.478631, 12.365311),
          (0.455027, 4.186858, 13.104445),
          (3.654295, 4.254963, 12.205945),
          (5.863695, 4.275586, 10.714709),
          (7.610693, 3.851083, 7.604854),
          (7.821661, 3.18878, 4.400032),

          (-7.640498, 0.756314, 4.967095),
          (-7.230136, 0.725585, 8.331517),
          (-5.748005, 0.480691, 11.193904),
          (-3.009834, 0.621885, 13.441012),
          (0.341982, 0.449246, 13.839247),
          (3.62126, 0.31676, 13.082255),
          (6.418348, 0.200262, 11.178412),
          (7.743287, 0.254288, 8.143276),
          (8.214926, 0.533799, 4.980188),

          (-7.794727, -1.924366, 4.686678),
          (-7.103159, -2.735806, 7.908936),
          (-5.549734, -3.131109, 10.995642),
          (-3.111164, -3.281632, 12.904391),
          (-0.072857, -3.405421, 13.509398),
          (3.044321, -3.820854, 12.781214),
          (5.712892, -3.643826, 10.907982),
          (7.304755, -3.111501, 7.913397),
          (7.92715, -2.443219, 4.673271),

          (-7.161848, -4.799244, 4.411572),
          (-6.375708, -5.683398, 7.142764),
          (-5.117089, -6.324777, 9.046002),
          (-2.8246, -6.605847, 10.717917),
          (-0.19569, -6.696784, 11.505725),
          (2.396374, -7.077637, 10.585553),
          (4.802065, -6.824497, 8.991351),
          (6.172683, -6.209247, 7.028114),
          (7.187716, -4.954237, 4.477674),

          (-5.894369, -6.974203, 4.318362),
          (-5.037746, -7.566237, 6.585544),
          (-2.544662, -8.415612, 7.820205),
          (-0.339835, -8.716856, 8.249729),
          (2.201964, -8.66148, 7.796194),
          (4.491326, -8.16103, 6.387415),
          (5.766648, -7.498684, 4.546538),

          (-6.387065, -5.755497, 1.886141),
          (-3.542601, -8.904578, 4.214279),
          (-0.080624, -9.660508, 4.670766),
          (3.050584, -9.25965, 4.194428),
          (6.192229, -6.797348, 2.355135),
          ]

    m1 = (m1 - np.min(m1)) / (np.max(m1) - np.min(m1))

    m1 = np.float32(np.array(m1))


    return m1



if __name__ == "__main__":

    eeg_data_path = "/data2/EEG_data/SEED5/EEG_DE_features/"
    eye_data_path = "/data2/EEG_data/SEED5/Eye_movement_features/"
    mm_data_path = "/data2/Ming/G2G/seed5_3fold_multimodal/" # 存储合并的两种数据
    eeg_eye_data_3fold(eeg_data_path, eye_data_path, mm_data_path)



