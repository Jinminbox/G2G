
# import os
# import numpy as np
# from scipy.io import loadmat
# import einops
# import torch
# import random
# # from einops import rearrange, reduce, repeat
#
#
# def load_data(folder_path, frequency):
#     """
#     读取数据，返回样本与标签
#     :param folder_path:
#     :param frequency:
#     :return:
#     """
#
#     emotionData = np.array([]) # 存储每一个subject的数组
#     emotionLabel = np.array([])
#     first_9_trials = 0
#
#     emotion = [2,1,0,0,1,2,0,1,2,2,1,0,1,2,0] #情绪标签序列，每一个mat文件的情绪标签顺序相同
#
#     for i in range(15):
#         dataKey =frequency + str(i+1) #读取的数据对应的Key
#
#         metaData = np.array((loadmat(folder_path,verify_compressed_data_integrity=False)[dataKey])).astype('float') #读取到原始的三维元数据
#
#         trMetaData = einops.rearrange(metaData, 'w h c -> h w c') #(235,62,5)
#         subArrayLength = trMetaData.shape[0]  # 读取每一个trial的时间量（非固定值）
#         trMetaData = np.array(trMetaData)
#         trMetaLabel = np.array([emotion[i],]*subArrayLength)
#
#         if emotionData.shape[0] == 0:
#             emotionData = trMetaData
#             emotionLabel = trMetaLabel
#         else:
#             emotionData = np.append(emotionData, trMetaData, axis=0)
#             emotionLabel = np.append(emotionLabel, trMetaLabel, axis=0)
#
#         if i < 9:
#             first_9_trials+=subArrayLength
#
#     return emotionData, emotionLabel, first_9_trials
#
#
# def extend_normal(sample):
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
#
#
# def data_independent(data_path, frequency):
#
#     x_, y_, dependent_index = load_data(data_path, frequency)
#
#     x_ = extend_normal(x_)
#
#     return x_, y_, dependent_index
#
#
# def save_normalized_data():
#     """
#     将重新组织的数据保存成npy格式，方便后续的使用
#     :return:
#     """
#     # SEED
#     rawData = "/data2/EEG_data/SEED/ExtractedFeatures/"
#     normalizedData = "/data2/EEG_data/SEED/Normalized/"
#     session = "1"  # or 2,3
#
#     dataPath = os.path.join(rawData, session)
#     x_list, y_list, subject_list = [], [], []
#
#     for subject in os.listdir(dataPath):
#         x_, y_, dependent_index = data_independent(os.path.join(dataPath, str(subject)), "de_LDS")
#
#         # 将预处理文件保存到字典中，方便下次使用时直接加载
#         normalizedDataPath = os.path.join(normalizedData, session)
#         dict = {'sample': x_, 'label': y_, 'dependent_index': dependent_index}
#         np.save(os.path.join(normalizedDataPath, (str(subject) + ".npy")), dict)
#
#
# if __name__ == "__main__":
#
#     save_normalized_data()



