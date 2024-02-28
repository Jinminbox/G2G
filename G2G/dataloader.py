import os
import numpy as np


SEED_datapath = ""


class SEED:

    def __init__(self, config):
        # 读取已经预处理好的EEG数据，以list组织，每一个list中的数据为[3394,62,5]
        self.config = config

        self.normalizedDataPath =os.path.join(self.config["dataset_path"], str(self.config["session"]))
        self.sampleList, self.labelList, self.subjects, self.split_index = [], [], [], []

        for subject in os.listdir(self.normalizedDataPath):
            dict_load = np.load(os.path.join(self.normalizedDataPath, (str(subject))), allow_pickle=True)
            sample_ = dict_load[()]['sample']
            label_ = dict_load[()]['label']
            dependent_index = dict_load[()]["dependent_index"]

            self.sampleList.append(sample_)
            self.labelList.append(label_)
            self.split_index.append(dependent_index)
            self.subjects.append(str(subject))


    def __getitem__(self, item):
        pass

    def __len__(self):
        pass



class SEED5:
    def __init__(self, config):
        #
        self.config = config

        self.normalizedDataPath = self.config["dataset_path"]
        self.sampleList, self.labelList, self.subjects, self.clip1, self.clip2 = [], [], [], [], []

        for subject in os.listdir(self.normalizedDataPath):
            dict_load = np.load(os.path.join(self.normalizedDataPath, (str(subject))), allow_pickle=True)
            sample_ = dict_load[()]['sample']
            label_ = dict_load[()]['label']
            clip1_ = dict_load[()]["clip1"]
            clip2_ = dict_load[()]["clip2"]

            self.sampleList.append(sample_)
            self.labelList.append(label_)
            self.clip1.append(clip1_)
            self.clip2.append(clip2_)
            self.subjects.append(str(subject))

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass


class MPED:
    def __init__(self, config):
        # 读取已经预处理好的EEG数据，以list组织，每一个list中的数据为[3394,62,5]
        self.config = config

        self.normalizedDataPath = self.config["dataset_path"]
        self.sampleList, self.labelList, self.subjects, self.split_index = [], [], [], []

        for subject in os.listdir(self.normalizedDataPath):
            dict_load = np.load(os.path.join(self.normalizedDataPath, (str(subject))), allow_pickle=True)
            sample_ = dict_load[()]['sample']
            label_ = dict_load[()]['label']
            dependent_index = dict_load[()]["dependent_index"]

            self.sampleList.append(sample_)
            self.labelList.append(label_)
            self.split_index.append(dependent_index)
            self.subjects.append(str(subject))


    def __getitem__(self, item):
        pass

    def __len__(self):
        pass




