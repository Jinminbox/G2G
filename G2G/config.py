

def load_config(dataset):

    config = None
    if dataset == "SEED":
        config = SEED_config
    elif dataset == "SEED5":
        config = SEED5_config
    elif dataset == "MPED":
        config = MPED_config
    else:
        raise Exception("Wrong dataset!")

    return config


SEED_config = {
    "dataset_name": "SEED",
    "session": 1,
    "num_class": 3,
    "epsilon": 0.01,  # 0.14
    "eeg_node_num": 62,
    "input_size": 5,
    "location_size": 3,
    "expand_size": 10,   # attention feature expend size
    "dataset_path": "/data2/EEG_data/SEED/Normalized/",
}


SEED5_config = {
    "dataset_name": "SEED5",
    "mode": 'unimodal',
    "session": 1,
    "tar_session": 3,
    "cfold": 3,
    "num_class": 5,
    "sup_node_num": 6, # Additional nodes constructed from eye movement data
    "epsilon": 0.01,  # 0.08
    "eeg_node_num": 62,
    "input_size": 5,
    "location_size": 3,
    "expand_size": 10,  # attention feature expend size
    "dataset_path": "/data2/Ming/G2G/seed5_3fold_multimodal/",
    # "dataset_path": "/data2/EEG_data/SEED5/sub7_multimodal/", # cross session
}



MPED_config = {
    "dataset_name": "MPED",
    # "session": 1,
    "num_class": 7,
    "sup_node_num": 5, # Additional nodes constructed from eye movement data
    "epsilon": 0.05,
    "eeg_node_num": 62,
    "input_size": 5,
    "location_size": 3,
    "expand_size": 10,   # attention feature expend size
    "dataset_path": "/data2/EEG_data/MPED/MM_Normalized/",
}