import os
import logging
import random
import torch
import shutil
import datetime
import torch.nn as nn
from torch.nn import functional as F
import numpy as np




class CE_Label_Smooth_Loss(nn.Module):
    def __init__(self, classes=4, epsilon=0.14, ):
        super(CE_Label_Smooth_Loss, self).__init__()

        self.classes = classes
        self.epsilon = epsilon


    def forward(self, input, target):

        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.epsilon / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.epsilon))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss



def set_logging_config(logdir):
    """
    logging configuration
    :param logdir:
    :return:
    """
    def beijing(sec, what):
        beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
        return beijing_time.timetuple()


    if not os.path.exists(logdir):
        os.makedirs(logdir)

    logging.Formatter.converter = beijing

    logging.basicConfig(format="[%(asctime)s] [%(name)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, ("log.txt"))),
                                  logging.StreamHandler(os.sys.stdout)])


def save_checkpoint(state, is_best, exp_name):
    """
    save the checkpoint during training stage
    :param state: content to be saved
    :param is_best: if model's performance is the best at current step
    :param exp_name: experiment name
    :return: None
    """
    torch.save(state, os.path.join('{}'.format(exp_name), 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join('{}'.format(exp_name), 'checkpoint.pth.tar'),
                        os.path.join('{}'.format(exp_name), 'model_best.pth.tar'))


## for dependent experiment
def de_train_test_split(data, label, index, config):

    if str(config["dataset_name"]) == "SEED": # 无需添加多模态数据
        new_data = data
    elif str(config["dataset_name"]) == "SEED5":
        new_data = split_eye_data(data, config["sup_node_num"])
    elif str(config["dataset_name"]) == "MPED":
        new_data = split_eye_data(data, config["sup_node_num"])

    x_train = new_data[:index]
    x_test = new_data[index:]
    y_train = label[:index]
    y_test = label[index:]

    data_and_label = {"x_train": x_train,
                      "x_test": x_test,
                      "y_train": y_train,
                      "y_test": y_test}

    return data_and_label


## for dependent experiment
def de_train_test_split_3fold(data, label, index1, index2, config):

    x_train = np.array([])
    x_test = np.array([])
    y_train = np.array([])
    y_test = np.array([])

    if str(config["dataset_name"]) == "SEED5":
        new_data = split_eye_data(data, config["sup_node_num"])


    x1 = new_data[:index1]
    x2 = new_data[index1:index2]
    x3 = new_data[index2:]

    y1 = label[:index1]
    y2 = label[index1:index2]
    y3 = label[index2:]

    if config["cfold"] == 1:
        x_train = np.append(x2, x3, axis=0)
        x_test = x1
        y_train = np.append(y2, y3, axis=0)
        y_test = y1

    elif config["cfold"] == 2:
        x_train = np.append(x1, x3, axis=0)
        x_test = x2
        y_train = np.append(y1, y3, axis=0)
        y_test = y2

    else:
        x_train = np.append(x1, x2, axis=0)
        x_test = x3
        y_train = np.append(y1, y2, axis=0)
        y_test = y3


    data_and_label = {"x_train": x_train,
                      "x_test": x_test,
                      "y_train": y_train,
                      "y_test": y_test}

    return data_and_label


## for independent experiment
def inde_train_test_split(dataList, labelList, subject_index, config):

    x_train, x_test, y_train, y_test = np.array([]), np.array([]),\
                                       np.array([]), np.array([])

    for j in range(len(dataList)):
        if j == subject_index:
            x_test = dataList[j]
            y_test = labelList[j]
        else:
            if x_train.shape[0] == 0:
                x_train = dataList[j]
                y_train = labelList[j]
            else:
                x_train = np.append(x_train, dataList[j], axis=0)
                y_train = np.append(y_train, labelList[j], axis=0)

    if str(config["dataset_name"]) == "SEED5":
        x_train = split_eye_data(x_train, config["sup_node_num"])
        x_test = split_eye_data(x_test, config["sup_node_num"])
    elif str(config["dataset_name"]) == "MPED":
        x_train = split_eye_data(x_train, config["sup_node_num"])
        x_test = split_eye_data(x_test, config["sup_node_num"])

    data_and_label = {"x_train": x_train,
                      "x_test": x_test,
                      "y_train": y_train,
                      "y_test": y_test}

    return data_and_label


## for independent experiment
def transfer_train_test_split(dataList, labelList, dataList_tar, labelList_tar, index, config):

    x_train, x_test, y_train, y_test = np.array([]), np.array([]),\
                                       np.array([]), np.array([])


    x_train = dataList[index]
    y_train = labelList[index]
    x_test = dataList_tar[index]
    y_test = labelList_tar[index]

    # 在数据中插入零
    x_train = split_eye_data(x_train, config["sup_node_num"])
    x_test = split_eye_data(x_test, config["sup_node_num"])


    data_and_label = {"x_train": x_train,
                      "x_test": x_test,
                      "y_train": y_train,
                      "y_test": y_test}

    return data_and_label


# seed-v and mped eye movement data split
def split_eye_data(eeg_eye_data, mode2_node_num):
    zero_index = []

    if mode2_node_num == 5: # MPED
        zero_index = [318, 319,
                      328, 329,
                      332, 333, 334, 335, 336, 337, 338, 339,
                      344, 345, 346, 347, 348, 349,
                      356, 357, 358, 359,
                      ]
    elif mode2_node_num == 6: # (6,6,4,4,4,9) SEED5
        zero_index =  [316, 317, 318, 319,
                       326, 327, 328, 329,
                       334, 335, 336, 337, 338, 339,
                       344, 345, 346, 347, 348, 349,
                       354, 355, 356, 357, 358, 359,
                       369
                       ]
    elif mode2_node_num == 7: # (8,8,2,4,2,2,2) MPED
        zero_index =  [318, 319,
                       328, 329,
                       332, 333, 334, 335, 336, 337, 338, 339,
                       344, 345, 346, 347, 348, 349,
                       352, 353, 354, 355, 356, 357, 358, 359,
                       362, 363, 364, 365, 366, 367, 368, 369,
                       372, 373, 374, 375, 376, 377, 378, 379,
                       ]
    else:
        print("Wrong eye movement data arrangement")

    for i in range(len(zero_index)):
        eeg_eye_data = np.insert(eeg_eye_data, zero_index[i], 0, axis=1)


    return eeg_eye_data


def return_coordinates():
    """
    Node location for SEED, SEED4, SEED5, MPED
    """
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


def random_1D_node(num, node_num):

    rand_lists = []
    for index in range(num):
        grand_list = [i for i in range(node_num)]
        random.shuffle(grand_list)
        rand_tensor = torch.tensor(grand_list).view(1, node_num)
        rand_lists.append(rand_tensor)

    rand_torch = torch.cat(tuple(rand_lists), 0)
    return rand_torch