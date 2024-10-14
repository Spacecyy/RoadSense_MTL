import os
from collections import Counter
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from dataset import *


def join_path(*a):
    return os.path.join(*a)

def get_dloader_RSCD(data_name, re_weighting,  ratio, configure):
    source_dataset, source_dataset_test, source_dataset_validation = [], [], []
    train_loader, valid_loader, test_loader = [], [], []
    batch_size = configure[data_name]['batch_size']
    valid_batch_size = configure[data_name]['valid_batch_size']
    test_batch_size = configure[data_name]['test_batch_size']
    ratio = str(int(ratio * 100))
    need_balance = re_weighting

    for tsk in configure[data_name]['task_list']:
        list_train = configure[data_name]['data_list']['train'][tsk][ratio]
        list_valid = configure[data_name]['data_list']['valid'][tsk][ratio]
        list_test = configure[data_name]['data_list']['test'][tsk][ratio]

        source_dataset.append(FileListDataset(list_path=list_train, transform=configure[data_name]['train_transform'], 
                                              filter=(lambda x: x in range(configure[data_name]['num_classes']))))
        source_dataset_validation.append(FileListDataset(list_path=list_valid, transform=configure[data_name]['train_transform'], 
                                              filter=(lambda x: x in range(configure[data_name]['num_classes']))))
        source_dataset_test.append(FileListDataset(list_path=list_test, transform=configure[data_name]['test_transform'], 
                                              filter=(lambda x: x in range(configure[data_name]['num_classes']))))
    if re_weighting:
        for i in range(len(source_dataset)):
            source_classes = source_dataset[i].labels
            source_freq = Counter(source_classes)
            source_class_weight = {x: 1.0 / source_freq[x] if need_balance else 1.0 for x in source_freq}
            source_weights = [source_class_weight[x] for x in source_dataset[i].labels]
            source_sampler = WeightedRandomSampler(source_weights, len(source_dataset[i].labels))
            train_loader.append(
                DataLoader(source_dataset[i], batch_size=batch_size, sampler=source_sampler, drop_last=True,
                           num_workers=8))
            # test_loader.append(
            #     DataLoader(source_dataset_test[i], batch_size=test_batch_size, shuffle=True, num_workers=8))
    else:
        train_loader = [DataLoader(source_dataset[t], batch_size=batch_size, shuffle=True, 
                                num_workers=8, drop_last=True) for t in range(len(source_dataset))]
    valid_loader = [DataLoader(source_dataset_validation[t], batch_size=valid_batch_size, shuffle=True, 
                                num_workers=8, drop_last=True) for t in range(len(source_dataset_validation))]
    test_loader = [DataLoader(source_dataset_test[t], batch_size=test_batch_size, shuffle=True, 
                                num_workers=8, drop_last=True) for t in range(len(source_dataset_test))]
    
    return train_loader, valid_loader, test_loader