import os
import torch
import numpy as np
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset


def join_path(*a):
    return os.path.join(*a)

class BaseImageDataset(Dataset):
    """
    base image dataset
    for image dataset, ``__getitem__`` usually reads an image from a given file path
    the image is guaranteed to be in **RGB** mode
    subclasses should fill ``datas`` and ``labels`` as they need.
    """

    def __init__(self, transform=None, return_id=False):
        self.return_id = return_id
        self.transform = transform or (lambda x : x)
        self.datas = []
        self.labels = []

    def __getitem__(self, index):
        if index< len(self.datas):
            idx = index
            im = Image.open(self.datas[index]).convert('RGB')
            im = self.transform(im)
        else:
            re_index = index % len(self.datas)
            idx = re_index
            im = Image.open(self.datas[re_index]).convert('RGB')
            im = self.transform(im)    

        if not self.return_id:
            return im, torch.tensor(self.labels[idx])
        return im, torch.tensor(self.labels[idx]), idx

    def __len__(self):
        return len(self.datas)

class FileListDataset(BaseImageDataset):
    """
    dataset that consists of a file which has the structure of :
    image_path label_id
    image_path label_id
    ......
    i.e., each line contains an image path and a label id
    """

    def __init__(self, list_path, path_prefix='', transform=None, return_id=False, num_classes=None, filter=None):
        """
        :param str list_path: absolute path of image list file (which contains (path, label_id) in each line) **avoid space in path!**
        :param str path_prefix: prefix to add to each line in image list to get the absolute path of image,
            esp, you should set path_prefix if file path in image list file is relative path
        :param int num_classes: if not specified, ``max(labels) + 1`` is used
        :param int -> bool filter: filter out the data to be used
        """
        super(FileListDataset, self).__init__(transform=transform, return_id = return_id)
        self.list_path = list_path
        self.path_prefix = path_prefix
        filter = filter or (lambda x : True)

        with open(self.list_path, 'r') as f:
            data = []
            for line in f.readlines():
                
                line = line.strip() 
                
                if line: # avoid empty lines
                    ans = line.split()
                    # print('ans:', ans)
                    if len(ans) == 1:
                        # no labels provided
                        data.append([ans[0], '0'])
                    elif len(ans) >= 2:
                        # add support for spaces in file path
                        label = ans[-1]
                        file = line[:-len(label)].strip()
                        data.append([file, label])
            self.datas = [join_path(self.path_prefix, x[0]) for x in data]
            try:
                self.labels = [int(x[1]) for x in data]
            except ValueError as e:
                print('invalid label number, maybe there is a space in the image path?')
                raise e

        ans = [(x, y) for (x, y) in zip(self.datas, self.labels) if filter(y)]
        self.datas, self.labels = zip(*ans)


        def return_datas(self):
            
            return self.datas       

class MultiLabelFileListDataset(BaseImageDataset):
    """
    dataset that consists of a file which has the structure of :
    image_path label_id
    image_path label_id
    ......
    i.e., each line contains an image path and a label id
    """

    def __init__(self, list_path, path_prefix='', transform=None, return_id=False, num_classes=None, filter=None):
        """
        :param str list_path: absolute path of image list file (which contains (path, label_id) in each line) **avoid space in path!**
        :param str path_prefix: prefix to add to each line in image list to get the absolute path of image,
            esp, you should set path_prefix if file path in image list file is relative path
        :param int num_classes: if not specified, ``max(labels) + 1`` is used
        :param int -> bool filter: filter out the data to be used
        """
        super(MultiLabelFileListDataset, self).__init__(transform=transform, return_id = return_id)
        self.list_path = list_path
        self.path_prefix = path_prefix
        self.total_classes = ['dry', 'wet', 'water', 'fresh_snow', 'melted_snow', 'ice', 'asphalt',
                             'concrete', 'mud', 'gravel', 'smooth', 'slight', 'severe']
        filter = filter or (lambda x : True)

        with open(self.list_path, 'r') as f:
            data = []
            for line in f.readlines():
                
                line = line.strip() # delete the empty space before and behind the text
                
                if line: # avoid empty lines
                    ans = line.split()
                    # print('ans:', ans)
                    if len(ans) == 1:
                        # no labels provided
                        data.append([ans[0], '0'])
                    elif len(ans) >= 2:
                        # add support for spaces in file path
                        label = ans[1:]
                        one_hot_label = self.label_one_hot(label)
                        # print(one_hot_label)
                        file = ans[0].strip()
                        # print([file, label])
                        data.append([file, one_hot_label])
                        
           
            self.datas = [join_path(self.path_prefix, x[0]) for x in data]
            try:
                self.labels = [x[1] for x in data]
                # print(len(self.datas), len(self.labels))
            except ValueError as e:
                print('invalid label number, maybe there is a space in the image path?')
                raise e

        ans = [(x, y) for (x, y) in zip(self.datas, self.labels)]

        self.datas, self.labels = zip(*ans)
        
    
    def label_one_hot(self, sample):
        one_hot_mtx = [0 for i in range(len(self.total_classes))]

        for label in sample:
            label_index = self.total_classes.index(label)
            one_hot_mtx[label_index] = 1

        return one_hot_mtx
    
    def return_datas(self):
        
        return self.datas
 

